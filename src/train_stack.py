"""Stack LGBM + GRU via logistic regression, then isotonic-calibrate.

Fit stack on calibA slice, isotonic on calibB slice, evaluate on test slice.
Saves: stack_logreg.pkl, stack_iso.pkl, stack_metrics.json
"""
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score

from fetch_data import load_or_fetch
from signals import MODEL_DIR


def _clip(p):
    return np.clip(p, 1e-6, 1 - 1e-6)


def main():
    lgbm = pd.read_pickle(MODEL_DIR / "dir_oof.pkl")
    gru = pd.read_pickle(MODEL_DIR / "gru_oof.pkl")
    df = lgbm.merge(gru, on="time", how="inner").sort_values("time").reset_index(drop=True)

    btc15 = load_or_fetch("BTC", "15m", days=150)
    btc15["time"] = pd.to_datetime(btc15["time"], utc=True)
    ymap = dict(zip(btc15["time"],
                    (btc15["close"].shift(-1) > btc15["close"]).astype(int)))
    df["y"] = df["time"].map(ymap)
    df = df.dropna().reset_index(drop=True)

    n = len(df)
    a, b, c = int(n * 0.70), int(n * 0.775), int(n * 0.85)
    tr, cA, cB, te = np.arange(n)[:a], np.arange(n)[a:b], np.arange(n)[b:c], np.arange(n)[c:]
    X = df[["p_lgbm", "p_gru"]].values
    y = df["y"].values.astype(int)

    # Proper stacking: K-fold OOF over all non-test rows -> fit isotonic on OOF,
    # refit blender on all non-test. (Fitting iso on one 860-row slice overfits.)
    from sklearn.model_selection import TimeSeriesSplit
    oof = np.full(n, np.nan)
    tss = TimeSeriesSplit(n_splits=5)
    non_test = np.r_[tr, cA, cB]
    for tri, vai in tss.split(X[non_test]):
        m = LogisticRegression().fit(X[non_test[tri]], y[non_test[tri]])
        oof[non_test[vai]] = m.predict_proba(X[non_test[vai]])[:, 1]
    lr = LogisticRegression().fit(X[non_test], y[non_test])
    m = ~np.isnan(oof[non_test])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.02, y_max=0.98).fit(
        oof[non_test][m], y[non_test][m])
    p_stack = iso.predict(lr.predict_proba(X)[:, 1])

    def rep(p, idx, name):
        return {name: {"acc": round(float(accuracy_score(y[idx], p[idx] > 0.5)), 4),
                       "logloss": round(float(log_loss(y[idx], _clip(p[idx]))), 4),
                       "brier": round(float(brier_score_loss(y[idx], p[idx])), 4),
                       "auc": round(float(roc_auc_score(y[idx], p[idx])), 4)}}

    out = {"n": n, "base_rate_test": round(float(y[te].mean()), 4), "test_start": str(df["time"].iloc[te[0]]),
           **rep(df["p_lgbm"].values, te, "lgbm"),
           **rep(df["p_gru"].values, te, "gru"),
           **rep(p_stack, te, "stack"),
           "coef": {"p_lgbm": round(float(lr.coef_[0][0]), 4),
                    "p_gru": round(float(lr.coef_[0][1]), 4),
                    "intercept": round(float(lr.intercept_[0]), 4)}}
    # conviction slices on test
    for name, p in (("lgbm", df["p_lgbm"].values), ("gru", df["p_gru"].values), ("stack", p_stack)):
        m = np.abs(p[te] - 0.5) >= 0.10
        out[f"{name}_conf10"] = {"traded_frac": round(float(m.mean()), 4),
                                 "hit_rate": round(float(y[te][m].mean()), 4) if m.sum() else None}
    print(json.dumps(out, indent=1))

    joblib.dump(lr, MODEL_DIR / "stack_logreg.pkl")
    joblib.dump(iso, MODEL_DIR / "stack_iso.pkl")
    pd.DataFrame({"time": df["time"], "p_stack": p_stack}).to_pickle(MODEL_DIR / "stack_oof.pkl")
    (MODEL_DIR / "stack_metrics.json").write_text(json.dumps(out, indent=1))
    print("saved.")


if __name__ == "__main__":
    main()
