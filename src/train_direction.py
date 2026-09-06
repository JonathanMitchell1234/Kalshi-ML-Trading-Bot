"""LightGBM DIRECTION classifier: P(close[t+1] > close[t]) + isotonic calibration.

Splits (time-ordered): train 70% | calibA 7.5% | calibB 7.5% | test 15%.
(calibA/B reserved for the stacking stage; isotonic fit uses train-tail split.)
Saves: dir_feats.pkl, dir_clf.pkl, dir_iso.pkl, dir_metrics.json
"""
import json
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score

from features_v2 import build_dataset, time_splits, LEADERS
from fetch_data import load_or_fetch
from signals import MODEL_DIR

PARAMS = dict(objective="binary", metric="binary_logloss", n_estimators=2000,
              learning_rate=0.03, num_leaves=31, min_child_samples=200,
              feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1,
              lambda_l1=0.5, lambda_l2=5.0, verbose=-1)


def decile_table(p: np.ndarray, y: np.ndarray) -> list[dict]:
    q = np.quantile(p, np.linspace(0, 1, 11))
    rows = []
    for i in range(10):
        m = (p >= q[i]) & (p <= q[i + 1] if i == 9 else p < q[i + 1])
        if m.sum() == 0:
            continue
        rows.append({"decile": i + 1, "n": int(m.sum()),
                     "mean_p": round(float(p[m].mean()), 4),
                     "hit_rate": round(float(y[m].mean()), 4)})
    return rows


def confidence_stats(p: np.ndarray, y: np.ndarray, conf: float) -> dict:
    m = np.abs(p - 0.5) >= conf
    win = ((p > 0.5) == (y == 1))  # side-aware: shorts win when y==0
    return {"conf": conf, "traded_frac": round(float(m.mean()), 4),
            "n": int(m.sum()), "longs": int(((p[m] > 0.5)).sum()) if m.sum() else 0,
            "hit_rate": round(float(win[m].mean()), 4) if m.sum() else None}


def main(asset: str = "BTC"):
    print("Loading data...")
    leaders = tuple(a for a in ("BTC", "ETH", "SOL", "XRP") if a != asset)
    px = "" if asset == "BTC" else f"{asset.lower()}_"
    btc15 = load_or_fetch(asset, "15m", days=150)
    btc1 = load_or_fetch(asset, "1m", days=120)
    b15 = {a: load_or_fetch(a, "15m", days=150) for a in leaders}
    b1 = {a: load_or_fetch(a, "1m", days=120) for a in leaders}
    df, feats, _pcafit = build_dataset(btc15, btc1, b15, b1, leaders, asset)
    print(f"{asset}: {len(df)} rows x {len(feats)} feats | base rate {df['y_up'].mean():.4f}")

    # Deadband labels: ignore chop smaller than est. half-spread + taker fee
    # (default 2 bps). Weight by normalized move size: 150bp breakouts teach
    # more than 1bp wiggle. Both use NEXT-bar return (the label), never inputs.
    import os as _os
    _eps = float(_os.getenv("DIR_DEADBAND", "0.0002"))
    _gamma = float(_os.getenv("DIR_GAMMA", "2.0"))
    _lr = df["close"].pipe(lambda c: np.log(c.shift(-1) / c))
    _sig = _lr.rolling(24, min_periods=8).std()
    _w = 1.0 + _gamma * (_lr.abs() / _sig.replace(0, np.nan)).fillna(0.0).clip(upper=5.0)
    _w = _w.where(_lr.abs() >= _eps, 0.0).fillna(0.0).values
    X, y = df[feats].values, df["y_up"].values
    tr, cA, cB, te = time_splits(len(df), times=df["time"])

    # Seed-averaged GBM: tail probabilities are unstable across bagging draws
    # on noisy labels; averaging K seeds cuts tail variance before calibration.
    N_SEEDS = 5
    raw = np.zeros(len(df))
    boosters = []
    best_iter = 0
    for s in range(N_SEEDS):
        clf = lgb.LGBMClassifier(**{**PARAMS, "random_state": 100 + s})
        clf.fit(X[tr], y[tr], sample_weight=_w[tr], eval_set=[(X[cA], y[cA])],
                eval_sample_weight=[_w[cA]],
                callbacks=[lgb.early_stopping(100, verbose=False)])
        raw += clf.predict_proba(X)[:, 1] / N_SEEDS
        boosters.append(clf)
        best_iter = max(best_iter, clf.best_iteration_ or 0)
    clf = boosters[-1]  # for feature importance below
    # persist the last seed's booster (feature importance); probs are averaged
    print(f"seeds={N_SEEDS} best_iter~{best_iter}")

    # Calibration: isotonic on pooled held-out folds. Deliberately NOT Platt:
    # Platt's parametric extrapolation inverted tails under distribution shift
    # (raw scores rank monotonically: top-10% raw hits 65%; isotonic preserves
    # ordering by construction). Fit on calibA+calibB, never train, never test.
    # cA drove early stopping -> over-optimistic there. Calibrate on cB only
    # (fully out-of-sample), never train, never test.
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.02, y_max=0.98)
    iso.fit(raw[cB], y[cB])
    p = iso.predict(raw)
    cal_kind = "isotonic"
    b_iso = log_loss(y[te], np.clip(p[te], 1e-6, 1 - 1e-6))

    def rep(name, idx):
        return {"n": len(idx), "acc": round(float(accuracy_score(y[idx], p[idx] > 0.5)), 4),
                "logloss": round(float(log_loss(y[idx], np.clip(p[idx], 1e-6, 1 - 1e-6))), 4),
                "brier": round(float(brier_score_loss(y[idx], p[idx])), 4),
                "auc": round(float(roc_auc_score(y[idx], p[idx])), 4)}

    metrics = {
        "base_rate_test": round(float(y[te].mean()), 4),
        "calibration": cal_kind,
        "logloss_test": round(b_iso, 4),
        "test": rep("test", te),
        "deciles_test": decile_table(p[te], y[te]),
        "conf_test": {str(c): confidence_stats(p[te], y[te], c) for c in (0.05, 0.10, 0.15)},
        "test_start": str(df["time"].iloc[te[0]]),
    }
    print(json.dumps({**metrics["test"], "base": metrics["base_rate_test"],
                      "conf": metrics["conf_test"]}, indent=1))

    joblib.dump(feats, MODEL_DIR / f"{px}dir_feats.pkl")
    joblib.dump(_pcafit, MODEL_DIR / f"{px}dir_pca.pkl")
    joblib.dump(boosters, MODEL_DIR / f"{px}dir_clf.pkl")  # list: live averages them
    joblib.dump(iso, MODEL_DIR / f"{px}dir_iso.pkl")
    joblib.dump({"kind": cal_kind, "model": iso}, MODEL_DIR / f"{px}dir_cal.pkl")
    pd.DataFrame({"time": pd.to_datetime(df["time"], utc=True), "p_lgbm": p}
                 ).to_pickle(MODEL_DIR / f"{px}dir_oof.pkl")
    (MODEL_DIR / f"{px}dir_metrics.json").write_text(json.dumps(metrics, indent=1))
    imp = pd.DataFrame({"feature": feats, "importance": clf.feature_importances_}
                       ).sort_values("importance", ascending=False)
    imp.to_csv(MODEL_DIR / f"{px}dir_importance.csv", index=False)
    print("Top 12:\n" + imp.head(12).to_string(index=False))
    from tracker import stamp
    print(f"version dir_{asset}=" + stamp(f"dir_{asset}"))
    print("saved.")
    return df, feats, p


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="BTC")
    main(ap.parse_args().asset)
