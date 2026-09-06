"""Backtest the 15m GBM on settled Kalshi KXBTC hourly events.

Protocol per settled event (all real data):
  - signal_time = expiry - lead_min (default 30 min before hourly expiry)
  - features from Binance 15m bars available strictly before signal_time
  - model predicts price at signal_time + 15 min
  - predicted bracket = $100 bracket containing the prediction
  - winner bracket = Kalshi market with result == 'yes' (ground truth)
  - hit = predicted bracket == winner; err15 = |pred - actual T+15m spot|

Caveats (shown in UI): Kalshi settles on CF BRTI, we use Binance.US spot
(small divergence can flip edge brackets); entry prices unknown historically,
so this measures bracket hit-rate, not P&L. P&L comes from live paper trades.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

from kalshi_client import get_event, event_ticker_for, _et_now
from signals import load_models
from tracker import save_backtest_run


def candidate_event_tickers(n_hours: int) -> list[str]:
    """Past hourly close tickers in ET (closing hour), newest first."""
    now_et = _et_now().replace(tzinfo=None)
    out = []
    # current ET hour is in progress -> its close is the coming hour; skip it
    base = now_et.replace(minute=0, second=0, microsecond=0)
    for h in range(1, n_hours + 1):
        out.append(event_ticker_for(base - timedelta(hours=h - 1)))
    return out


def bracket_for_price(brackets: list[dict], price: float) -> str | None:
    for b in brackets:
        st = b["strike_type"]
        if st == "less" and price < float(b["cap"] or b["floor"] or 0):
            return b["ticker"]
        if st == "greater" and price >= float(b["floor"] or 0):
            return b["ticker"]
        if st == "between":
            fl = float(b["floor"] or 0)
            cap = float(b.get("cap") or fl + 100)
            if fl <= price < cap:
                return b["ticker"]
    return None


def run_backtest(n_events: int = 48, lead_min: int = 30, progress=None) -> dict:
    from kalshi_client import parse_bracket
    from fetch_data import load_or_fetch
    from features import add_features

    reg, clf, feats, _ = load_models()
    raw = load_or_fetch(days=30)
    raw["time"] = pd.to_datetime(raw["time"], utc=True)
    feat = add_features(raw).reset_index(drop=True)
    feat["time"] = pd.to_datetime(feat["time"], utc=True)
    closes = raw.set_index("time")["close"]

    picks, checked = [], 0
    for t in candidate_event_tickers(n_events * 2 + 6):
        if len(picks) >= n_events:
            break
        checked += 1
        try:
            ev = get_event(t)
        except Exception:
            continue
        mk = ev.get("markets", [])
        if not mk:
            continue
        if any((m.get("status") not in ("finalized", "settled")) for m in mk):
            continue
        winners = [m["ticker"] for m in mk if (m.get("result") or "") == "yes"]
        if len(winners) != 1:
            continue
        # close_time = true hourly expiry; expiration_value = CF BRTI settlement
        exp_ts = mk[0].get("close_time")
        try:
            expiry = datetime.fromisoformat(str(exp_ts).replace("Z", "+00:00"))
        except Exception:
            continue
        settle = None
        try:
            settle = float(mk[0].get("expiration_value"))
        except (TypeError, ValueError):
            pass
        signal = expiry - timedelta(minutes=lead_min)
        # last fully CLOSED 15m candle (candle at T covers [T, T+15))
        hist = feat[feat["time"] <= signal - timedelta(minutes=15)]
        if len(hist) < 200:
            continue
        row = hist.iloc[[-1]]
        pred = float(row["close"].iloc[0]) * float(np.exp(reg.predict(row[feats].values)[0]))
        spot = float(row["close"].iloc[0])
        brackets = [parse_bracket(m) for m in mk]
        pred_b = bracket_for_price(brackets, pred)
        # actual spot 15 min after signal (model horizon) for err15
        try:
            actual15 = float(closes[closes.index <= pd.Timestamp(signal) + pd.Timedelta(minutes=15)].iloc[-1])
            err15 = abs(pred - actual15)
        except Exception:
            err15 = None
        # expiry spot: prefer Kalshi's own settlement; else Binance proxy
        expiry_spot = settle
        if expiry_spot is None:
            try:
                expiry_spot = float(closes[closes.index <= pd.Timestamp(expiry)].iloc[-1])
            except Exception:
                expiry_spot = None
        picks.append({
            "event_ticker": t,
            "expiry_ts": expiry.isoformat(timespec="seconds"),
            "signal_time": signal.isoformat(timespec="seconds"),
            "spot": spot, "pred_price": round(pred, 2),
            "pred_bracket": pred_b, "winner_bracket": winners[0],
            "hit": 1 if pred_b == winners[0] else 0,
            "err15": round(err15, 2) if err15 is not None else None,
            "expiry_spot": expiry_spot,
        })
        if progress:
            progress(len(picks), n_events)
    rid = save_backtest_run(n_events=len(picks), lead_min=lead_min, picks=picks,
                            note=f"lead={lead_min}m, scanned={checked}")
    scored = [p for p in picks if p["hit"] is not None]
    return {"run_id": rid, "n": len(picks),
            "hit_rate": round(sum(p["hit"] for p in picks) / len(picks), 4) if picks else None}


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    print(run_backtest(n_events=n, progress=lambda a, b: print(f"  {a}/{b}")))
