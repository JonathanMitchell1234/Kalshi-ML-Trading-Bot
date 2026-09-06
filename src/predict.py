"""Inference: predict BTC price 15 min ahead + Kalshi edge helper.

Usage:
    python predict.py                       # live: fetch latest bars, predict next close
    python predict.py --strike 97000 --side yes --kalshi_yes_price 0.55
"""
import os
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from fetch_data import fetch_history
from features import add_features

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent
_MD = Path(os.getenv("MODEL_DIR", str(ROOT / "trained_models")))
MODEL_DIR = _MD if _MD.is_absolute() else ROOT / _MD


def load_models():
    reg = joblib.load(MODEL_DIR / "gbm_reg.pkl")
    clf = joblib.load(MODEL_DIR / "gbm_clf.pkl")
    feats = joblib.load(MODEL_DIR / "feature_cols.pkl")
    return reg, clf, feats


def predict_next():
    reg, clf, feats = load_models()
    raw = fetch_history(days=8, verbose=False)
    df = add_features(raw).dropna(subset=feats).reset_index(drop=True)
    row = df.iloc[[-1]]
    X = row[feats].values
    logret = float(reg.predict(X)[0])
    p_up = float(clf.predict_proba(X)[0, 1])
    last_close = float(row["close"].iloc[0])
    pred_close = last_close * float(np.exp(logret))
    return {
        "asof": str(row["time"].iloc[0]),
        "last_close": last_close,
        "pred_logret": logret,
        "pred_close_next_15m": pred_close,
        "p_up": p_up,
        "expected_move_bps": logret * 1e4,
    }


def kalshi_edge(pred_close: float, p_up_model: float, strike: float, kalshi_yes_price: float, side: str = "yes"):
    """Rough edge check for a Kalshi above/below-strike market.

    Model-implied P(close > strike): use direction prob as base and shift by
    distance-to-strike in units of recent 15m volatility. This is a heuristic —
    replace with a calibrated residual distribution once you log live errors.
    """
    import math
    # fallback: logistic map of (pred - strike) scaled by ~typical 15m move ($150 default)
    scale = 150.0
    p_above = 1 / (1 + math.exp(-(pred_close - strike) / scale))
    # blend with classifier direction signal
    p_above = 0.5 * p_above + 0.5 * p_up_model if abs(pred_close - strike) < scale else p_above
    fair = p_above if side == "yes" else 1 - p_above
    mkt = kalshi_yes_price if side == "yes" else 1 - kalshi_yes_price
    edge = fair - mkt
    return {"p_above": p_above, "fair": fair, "market": mkt, "edge": edge,
            "bet": edge >= float(os.getenv("MIN_EDGE_THRESHOLD", "0.05"))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--strike", type=float, default=None)
    ap.add_argument("--side", type=str, default="yes")
    ap.add_argument("--kalshi_yes_price", type=float, default=None)
    a = ap.parse_args()
    out = predict_next()
    print(out)
    if a.strike is not None and a.kalshi_yes_price is not None:
        print(kalshi_edge(out["pred_close_next_15m"], out["p_up"], a.strike, a.kalshi_yes_price, a.side))
