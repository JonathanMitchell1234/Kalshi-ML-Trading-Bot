"""Train LightGBM GBMs for BTC 15m-ahead prediction.

Models:
  - reg_model:  predicts log-return to next 15m close -> converted to price
  - clf_model:  predicts P(up) for direction / Kalshi edge estimation

Validation: TimeSeriesSplit (3 folds) + final holdout (last 10%).
Saves to MODEL_DIR (default trained_models/).
"""
import os
import json
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, log_loss
from dotenv import load_dotenv

from fetch_data import load_or_fetch
from features import add_features, get_feature_cols

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent
_MD = Path(os.getenv("MODEL_DIR", str(ROOT / "trained_models")))
MODEL_DIR = _MD if _MD.is_absolute() else ROOT / _MD
MODEL_DIR.mkdir(parents=True, exist_ok=True)

REG_PARAMS = dict(
    objective="regression", metric="rmse",
    n_estimators=2000, learning_rate=0.03,
    num_leaves=63, max_depth=-1, min_child_samples=40,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    lambda_l1=0.1, lambda_l2=1.0, verbose=-1,
)
CLF_PARAMS = dict(
    objective="binary", metric="binary_logloss",
    n_estimators=2000, learning_rate=0.03,
    num_leaves=63, max_depth=-1, min_child_samples=60,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    lambda_l1=0.1, lambda_l2=1.0, verbose=-1,
)


def directional_accuracy(y_true_ret, y_pred_ret) -> float:
    return float(np.mean(np.sign(y_true_ret) == np.sign(y_pred_ret)))


def main(days: int = 365, test_frac: float = 0.10):
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] Loading data ({days}d)...")
    raw = load_or_fetch(days=days)
    df = add_features(raw).dropna().reset_index(drop=True)
    feats = get_feature_cols(df)
    print(f"Rows after features/dropna: {len(df)} | features: {len(feats)}")

    X = df[feats].values
    y_reg = df["target_logret"].values
    y_clf = df["target_up"].values
    close = df["close"].values

    n = len(df)
    split = int(n * (1 - test_frac))
    Xtr, Xte = X[:split], X[split:]
    yreg_tr, yreg_te = y_reg[:split], y_reg[split:]
    yclf_tr, yclf_te = y_clf[:split], y_clf[split:]
    close_te = close[split:]
    print(f"Train {len(Xtr)} | Holdout {len(Xte)} (cut at {df['time'].iloc[split]})")

    # --- CV on train portion ---
    tss = TimeSeriesSplit(n_splits=3)
    cv_reg, cv_clf = [], []
    for i, (tri, vai) in enumerate(tss.split(Xtr), 1):
        rm = lgb.LGBMRegressor(**REG_PARAMS)
        rm.fit(Xtr[tri], yreg_tr[tri], eval_set=[(Xtr[vai], yreg_tr[vai])],
               callbacks=[lgb.early_stopping(100, verbose=False)])
        p = rm.predict(Xtr[vai])
        rmse = float(np.sqrt(mean_squared_error(yreg_tr[vai], p)))
        cv_reg.append(rmse)
        cm = lgb.LGBMClassifier(**CLF_PARAMS)
        cm.fit(Xtr[tri], yclf_tr[tri], eval_set=[(Xtr[vai], yclf_tr[vai])],
               callbacks=[lgb.early_stopping(100, verbose=False)])
        acc = float(accuracy_score(yclf_tr[vai], cm.predict(Xtr[vai])))
        cv_clf.append(acc)
        print(f"  fold {i}: reg RMSE(logret)={rmse:.6f} | clf acc={acc:.4f}")
    print(f"CV mean: reg RMSE={np.mean(cv_reg):.6f} | clf acc={np.mean(cv_clf):.4f}")

    # --- Final fit on full train ---
    reg = lgb.LGBMRegressor(**REG_PARAMS)
    reg.fit(Xtr, yreg_tr, eval_set=[(Xte, yreg_te)], callbacks=[lgb.early_stopping(100, verbose=False)])
    clf = lgb.LGBMClassifier(**CLF_PARAMS)
    clf.fit(Xtr, yclf_tr, eval_set=[(Xte, yclf_te)], callbacks=[lgb.early_stopping(100, verbose=False)])

    # --- Holdout evaluation (in price space too) ---
    pred_logret = reg.predict(Xte)
    pred_close = close_te * np.exp(pred_logret)
    true_close = df["target_close"].values[split:]
    rmse_px = float(np.sqrt(mean_squared_error(true_close, pred_close)))
    mae_px = float(mean_absolute_error(true_close, pred_close))
    r2 = float(r2_score(true_close, pred_close))
    mape_bps = float(np.mean(np.abs(true_close - pred_close) / true_close) * 1e4)
    dir_acc = directional_accuracy(yreg_te, pred_logret)
    naive_rmse = float(np.sqrt(mean_squared_error(true_close, close_te)))  # persistence baseline

    p_up = clf.predict_proba(Xte)[:, 1]
    clf_acc = float(accuracy_score(yclf_te, (p_up > 0.5).astype(int)))
    try:
        clf_ll = float(log_loss(yclf_te, np.clip(p_up, 1e-6, 1 - 1e-6)))
    except Exception:
        clf_ll = None
    base_up_rate = float(yclf_te.mean())

    metrics = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": n, "n_features": len(feats),
        "train_end_time": str(df["time"].iloc[split - 1]),
        "holdout_start": str(df["time"].iloc[split]),
        "cv_reg_rmse_logret": cv_reg, "cv_clf_acc": cv_clf,
        "holdout": {
            "rmse_price": rmse_px, "mae_price": mae_px, "r2_price": r2,
            "mape_bps": mape_bps, "directional_accuracy": dir_acc,
            "naive_persistence_rmse": naive_rmse,
            "beat_naive": rmse_px < naive_rmse,
            "clf_accuracy": clf_acc, "clf_logloss": clf_ll,
            "base_up_rate": base_up_rate,
            "last_close": float(close_te[-1]),
        },
    }
    print("\n==== HOLDOUT ====")
    print(json.dumps(metrics["holdout"], indent=2))

    # --- Save ---
    joblib.dump(reg, MODEL_DIR / "gbm_reg.pkl")
    joblib.dump(clf, MODEL_DIR / "gbm_clf.pkl")
    joblib.dump(feats, MODEL_DIR / "feature_cols.pkl")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    imp = pd.DataFrame({"feature": feats, "importance": reg.feature_importances_}).sort_values("importance", ascending=False)
    imp.to_csv(MODEL_DIR / "feature_importance.csv", index=False)
    print(f"\nSaved models to {MODEL_DIR}/")
    print("Top 15 features:\n" + imp.head(15).to_string(index=False))
    return metrics


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365)
    a = ap.parse_args()
    main(days=a.days)
