"""Policy backtest on settled KXBTC15M up/down binaries.

Per settled event: p = calibrated P(up) from dir_oof at window open
(our 15m grid aligns exactly with Kalshi windows). Policy: trade iff
|p-.5| >= conf; side = YES if p>.5 else NO. Win = side matches result.
No historical prices exist, so this measures policy hit-rate, not P&L.
"""
import pandas as pd
from datetime import datetime

from kalshi_client import get_15m, recent_15m_tickers
from signals import MODEL_DIR
from tracker import save_backtest_run


def run_backtest_15m(n_events: int = 200, conf: float = 0.10, progress=None, asset: str = "BTC") -> dict:
    px = "" if asset == "BTC" else f"{asset.lower()}_"
    oof = pd.read_pickle(MODEL_DIR / f"{px}dir_oof.pkl")
    oof["time"] = pd.to_datetime(oof["time"], utc=True)
    pmap = dict(zip(oof["time"], oof["p_lgbm"]))

    picks, scanned = [], 0
    for t in recent_15m_tickers(n_events + 10, asset):
        if len(picks) >= n_events:
            break
        scanned += 1
        try:
            m = get_15m(t)
        except Exception:
            continue
        if m["status"] not in ("finalized", "settled") or m["result"] not in ("yes", "no"):
            continue
        try:
            close = datetime.fromisoformat(str(m["close_time"]).replace("Z", "+00:00"))
        except Exception:
            continue
        opent = close - pd.Timedelta(minutes=15)
        key = pd.Timestamp(opent)
        key = key.tz_localize("UTC") if key.tzinfo is None else key.tz_convert("UTC")
        p = pmap.get(key)
        if p is None:  # nearest grid fallback (<=1 min)
            cand = oof.iloc[(oof["time"] - key).abs().argsort()[:1]]
            if cand.empty or abs((cand["time"].iloc[0] - key).total_seconds()) > 60:
                continue
            p = float(cand["p_lgbm"].iloc[0])
        y = 1 if m["result"] == "yes" else 0
        traded = abs(p - 0.5) >= conf
        picks.append({
            "event_ticker": t,
            "expiry_ts": close.isoformat(timespec="seconds"),
            "signal_time": key.isoformat(timespec="seconds"),
            "spot": m["target"], "pred_price": round(p, 4),
            "pred_bracket": ("yes" if p > 0.5 else "no") if traded else "skip",
            "winner_bracket": m["result"],
            "hit": int((p > 0.5) == bool(y)) if traded else None,
            "err15": round((p - y) ** 2, 4),
            "expiry_spot": m.get("expiration_value"),
        })
        if progress:
            progress(len(picks), n_events)
    rid = save_backtest_run(n_events=len(picks), lead_min=0, picks=picks,
                            note=f"{asset} 15m up/down policy conf={conf}, scanned={scanned}")
    tr = [p for p in picks if p["hit"] is not None]
    return {"run_id": rid, "n": len(picks), "traded": len(tr),
            "hit_rate": round(sum(p["hit"] for p in tr) / len(tr), 4) if tr else None}


if __name__ == "__main__":
    import sys
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=200)
    ap.add_argument("--asset", default="BTC")
    ap.add_argument("--conf", type=float, default=0.10)
    a = ap.parse_args()
    print(run_backtest_15m(n_events=a.n, conf=a.conf, asset=a.asset,
                           progress=lambda x, y: print(f"  {x}/{y}", end="\r")))
