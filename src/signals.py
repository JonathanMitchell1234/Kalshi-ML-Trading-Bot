"""Model signals mapped onto Kalshi brackets.

Fair value: Normal(pred_price, sigma) mass inside each bracket, where
sigma = holdout 15-min RMSE scaled by sqrt(minutes_to_expiry / 15).
This is a transparent heuristic — drift beyond the 15-min horizon is unknown.
"""
import json
import math
import os
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent


def _resolve(p: Path) -> Path:
    return p if p.is_absolute() else ROOT / p


MODEL_DIR = _resolve(Path(os.getenv("MODEL_DIR", "trained_models")))
DATA_DIR = _resolve(Path(os.getenv("DATA_DIR", "data_cache")))

_models = {}


def load_models():
    if "reg" not in _models:
        _models["reg"] = joblib.load(MODEL_DIR / "gbm_reg.pkl")
        _models["clf"] = joblib.load(MODEL_DIR / "gbm_clf.pkl")
        _models["feats"] = joblib.load(MODEL_DIR / "feature_cols.pkl")
        _models["metrics"] = json.loads((MODEL_DIR / "metrics.json").read_text())
    return _models["reg"], _models["clf"], _models["feats"], _models["metrics"]


def sigma_for_holdout() -> float:
    _, _, _, metrics = load_models()
    return float(metrics["holdout"]["rmse_price"])  # 15-min horizon


def latest_prediction() -> dict:
    """Fresh 15-min-ahead prediction from latest Binance bars (cached CSV ok)."""
    from fetch_data import load_or_fetch
    from features import add_features
    reg, clf, feats, _ = load_models()
    raw = load_or_fetch(days=8)
    df = add_features(raw).dropna(subset=feats).reset_index(drop=True)
    row = df.iloc[[-1]]
    X = row[feats].values
    logret = float(reg.predict(X)[0])
    p_up = float(clf.predict_proba(X)[0, 1])
    spot = float(row["close"].iloc[0])
    out = {
        "asof": str(row["time"].iloc[0]),
        "spot": spot,
        "pred_price": spot * float(np.exp(logret)),
        "p_up": p_up,
    }
    try:
        out.update(stacked_prediction())
    except Exception as e:
        out.update({"p_stack": None, "stack_error": str(e)[:120]})
    return out


def _px(asset: str) -> str:
    return "" if asset == "BTC" else f"{asset.lower()}_"


def asset_signal(asset: str = "BTC") -> dict:
    """Per-asset signal for the 15m up/down trader.

    BTC: full prediction (regression center + direction). Others: spot +
    direction probability only (no regression model).
    """
    if asset == "BTC":
        return latest_prediction()
    from fetch_data import load_or_fetch
    d = stacked_prediction(asset)
    try:
        bars = load_or_fetch(asset, "15m", days=2)
        spot = float(bars["close"].iloc[-1])
        asof = str(bars["time"].iloc[-1])
    except Exception:
        spot, asof = None, None
    return {"asset": asset, "spot": spot, "pred_price": spot, "asof": asof, **d}


def load_direction_stack(asset: str = "BTC"):
    """Production direction signal: isotonic-calibrated LightGBM P(up).

    (GRU+stack evaluated on BTC and added no value — see stack_metrics.json.
    ETH ships LGBM-only.) Per-asset artifacts: {eth_}dir_{feats,clf,iso}.pkl
    """
    key = f"stack_{asset}"
    if key not in _models:
        try:
            px = _px(asset)
            try:
                cal = joblib.load(MODEL_DIR / f"{px}dir_cal.pkl")
            except Exception:
                cal = {"kind": "isotonic",
                       "model": joblib.load(MODEL_DIR / f"{px}dir_iso.pkl")}
            try:
                _pca = joblib.load(MODEL_DIR / f"{px}dir_pca.pkl")
            except Exception:
                _pca = None
            S = {"feats": joblib.load(MODEL_DIR / f"{px}dir_feats.pkl"),
                 "pca": _pca,
                 "clf": joblib.load(MODEL_DIR / f"{px}dir_clf.pkl"),
                 "cal": cal,
                 "gru_meta": None}
            if asset == "BTC":
                try:
                    S["gru_meta"] = json.loads((MODEL_DIR / "gru_meta.json").read_text())
                except Exception:
                    pass
            _models[key] = S
        except Exception as e:
            _models[key] = {"error": str(e)[:200]}
    s = _models[key]
    return None if "error" in s else s


def stacked_prediction(asset: str = "BTC") -> dict:
    """Calibrated P(up) from production direction model on latest bars."""
    from fetch_data import load_or_fetch
    from features_v2 import latest_row
    leaders = tuple(a for a in ("BTC", "ETH", "SOL", "XRP") if a != asset)
    S = load_direction_stack(asset)
    if S is None:
        return {"p_stack": None}
    btc15 = load_or_fetch(asset, "15m", days=30)
    btc1 = load_or_fetch(asset, "1m", days=3)
    b15 = {a: load_or_fetch(a, "15m", days=30) for a in leaders}
    b1 = {a: load_or_fetch(a, "1m", days=3) for a in leaders}
    # trim: the live row needs ~100 bars of history for trailing windows, not months
    btc15 = btc15.tail(500).reset_index(drop=True)
    btc1 = btc1.tail(4000).reset_index(drop=True)
    b15 = {a: df.tail(500).reset_index(drop=True) for a, df in b15.items()}
    b1 = {a: df.tail(4000).reset_index(drop=True) for a, df in b1.items()}

    row = latest_row(btc15, btc1, b15, b1, S["feats"], leaders, asset, S.get("pca"))
    X1 = row[S["feats"]].values.reshape(1, -1)
    clfs = S["clf"]
    clfs = clfs if isinstance(clfs, list) else [clfs]
    raw = sum(c.predict_proba(X1)[:, 1] for c in clfs) / len(clfs)
    cal = S["cal"]
    if cal["kind"] == "platt":
        p_lgbm = float(cal["model"].predict_proba(raw.reshape(-1, 1))[:, 1][0])
    else:
        p_lgbm = float(cal["model"].predict(raw)[0])

    p_gru = None  # GRU retired 2026-09: AUC 0.50 across all runs, TCN skipped
    # (low-SNR 60m windows; LGBM+flow dominates). train_gru.py kept for record.
    # production = calibrated LGBM (stack evaluated, added no value)
    return {"p_lgbm": round(p_lgbm, 4), "p_gru": p_gru, "p_stack": round(p_lgbm, 4)}


def _gru_proba_via_subprocess(btc_tail: pd.DataFrame, b1: dict, meta: dict,
                              timeout: int = 120) -> float | None:
    """GRU forward in a child process.

    torch segfaults LightGBM predictions in-process on this platform, so the
    GRU (display/monitoring only) is isolated. Any failure -> None;
    production LGBM signal is unaffected.
    """
    import subprocess
    import sys
    import tempfile
    try:
        b = btc_tail.sort_values("time")
        seq = b.tail(meta["seq"])[["time"]].copy()
        seq = seq.merge(b[["time", "close", "volume", "high", "low"]], on="time")
        seq["ret"] = np.log(seq["close"] / seq["close"].shift(1))
        seq["volr"] = (seq["volume"] / seq["volume"].rolling(1440, min_periods=20)
                       .median().replace(0, np.nan))
        seq["rng"] = (seq["high"] - seq["low"]) / seq["close"]
        for a in ("ETH", "SOL", "XRP"):
            d = b1[a].sort_values("time")[["time", "close"]].copy()
            d["ret"] = np.log(d["close"] / d["close"].shift(1))
            seq = pd.merge_asof(seq.sort_values("time"),
                                d[["time", "ret"]].rename(columns={"ret": f"{a}_r"}),
                                on="time", direction="backward")
        stds = meta["stds"]
        arr = np.column_stack([
            seq["ret"].fillna(0) / stds["BTC"], (seq["ret"].fillna(0) / stds["BTC"]).abs(),
            seq["volr"].fillna(1.0), seq["rng"].fillna(0.0),
            seq["ETH_r"].fillna(0) / stds["ETH"], seq["SOL_r"].fillna(0) / stds["SOL"],
            seq["XRP_r"].fillna(0) / stds["XRP"],
            np.sin(2 * np.pi * seq["time"].dt.minute / 60),
            np.cos(2 * np.pi * seq["time"].dt.minute / 60),
        ]).astype(np.float32)[-meta["seq"]:]
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            np.save(f.name, arr)
            path = f.name
        script = (
            "import sys, json, math, numpy as np, torch\n"
            "sys.path.insert(0, %r)\n"
            "from train_gru import GRU\n"
            "md = json.load(open(%r))\n"
            "net = GRU(d_in=len(md['feats']), h=md.get('hidden', 32))\n"
            "net.load_state_dict(torch.load(%r, map_location='cpu'))\n"
            "net.eval()\n"
            "arr = np.load(%r)\n"
            "with torch.no_grad():\n"
            "    z = net(torch.from_numpy(arr).unsqueeze(0)).item()\n"
            "print(1 / (1 + math.exp(-z)))\n"
        ) % (str(ROOT / "src"), str(MODEL_DIR / "gru_meta.json"),
             str(MODEL_DIR / "gru_state.pt"), path)
        r = subprocess.run([sys.executable, "-c", script], capture_output=True,
                           text=True, timeout=timeout)
        return round(float(r.stdout.strip().split()[-1]), 4) if r.returncode == 0 else None
    except Exception:
        return None


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _t_cdf(x: float, nu: float = 5.0) -> float:
    from scipy.stats import t as _t
    return float(_t.cdf(x, nu))


def gk_sigma(minutes: float = 15.0) -> float | None:
    """Garman-Klass realized vol from trailing 1m BTC bars, scaled to horizon.

    Falls back to None when bars unavailable (caller uses holdout RMSE).
    """
    try:
        from fetch_data import load_or_fetch
        m1 = load_or_fetch("BTC", "1m", days=2)
        m1 = m1.tail(max(int(minutes), 10) + 5).dropna(subset=["high", "low", "open", "close"])
        if len(m1) < 5:
            return None
        import numpy as _np
        import pandas as _pd
        h, l, o, c = (m1["high"].values, m1["low"].values, m1["open"].values, m1["close"].values)
        hl = _np.log(h / l)
        co = _np.log(c / o)
        var_per_min = float(_np.mean(0.5 * hl * hl - (2 * _np.log(2) - 1) * co * co))
        if not _np.isfinite(var_per_min) or var_per_min <= 0:
            return None
        spot = float(m1["close"].iloc[-1])
        return spot * float(_np.sqrt(var_per_min * max(minutes, 5)))
    except Exception:
        return None


def bracket_fair_values(pred_price: float, minutes_to_expiry: float, brackets: list[dict]) -> dict[str, float]:
    """ticker -> fair P(settles in bracket). Tails use CDF; 'between' uses mass.

    Dynamic vol (Garman-Klass on trailing 1m, floored at half the holdout
    RMSE) + fat-tailed Student-t (nu=5): wings stop being underpriced when
    regimes shift. Set WX_USE_DYNAMIC_VOL=0 for the legacy static-Normal.
    """
    use_dyn = os.getenv("USE_DYNAMIC_VOL", "1") == "1"
    nu = float(os.getenv("T_NU", "5"))
    sigma15 = sigma_for_holdout()
    if use_dyn:
        gk = gk_sigma(minutes_to_expiry)
        base = sigma15 * math.sqrt(max(minutes_to_expiry, 5) / 15.0)
        sigma = max(gk or 0.0, 0.5 * base) if (gk or 0) > 0 else base
    else:
        sigma = max(sigma15 * math.sqrt(max(minutes_to_expiry, 5) / 15.0), 1e-6)
    sigma = max(sigma, 1e-6)
    cdf = (lambda z: _t_cdf(z, nu)) if use_dyn else _norm_cdf
    # Student-t has variance nu/(nu-2): rescale so sigma stays a std-dev
    s = sigma * math.sqrt((nu - 2) / nu) if use_dyn and nu > 2 else sigma
    fairs: dict[str, float] = {}
    for b in brackets:
        st = b["strike_type"]
        if st == "less":
            cap = float(b["cap"] or b["floor"] or 0)
            fairs[b["ticker"]] = cdf((cap - pred_price) / s)
        elif st == "greater":
            fl = float(b["floor"] or 0)
            fairs[b["ticker"]] = 1.0 - cdf((fl - pred_price) / s)
        else:
            fl = float(b["floor"] or 0)
            cap = float(b.get("cap") or fl + 100)
            fairs[b["ticker"]] = cdf((cap - pred_price) / s) - cdf((fl - pred_price) / s)
    return fairs


def minutes_to_close(close_time: str | None) -> float | None:
    if not close_time:
        return None
    try:
        dt = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
        return max((dt - datetime.now(timezone.utc)).total_seconds() / 60.0, 0.5)
    except Exception:
        return None


def top_edges(pred_price: float, minutes_to_expiry: float, brackets: list[dict],
              min_edge: float = 0.05, top_n: int = 5) -> list[dict]:
    """Rank brackets by fair-minus-ask edge (YES side). Prices in cents here."""
    fairs = bracket_fair_values(pred_price, minutes_to_expiry, brackets)
    rows = []
    for b in brackets:
        if b["status"] != "active" or b["yes_ask"] is None:
            continue
        fair = fairs[b["ticker"]]
        ask = b["yes_ask"] / 100.0
        rows.append({**b, "fair": round(fair, 4), "edge": round(fair - ask, 4)})
    rows.sort(key=lambda r: r["edge"], reverse=True)
    return [r for r in rows[:top_n]]
