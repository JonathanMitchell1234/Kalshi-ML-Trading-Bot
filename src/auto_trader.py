"""Auto paper-trader: every cycle, pick the best-edge bracket and paper-buy it.

Selection (YES side only, v1):
  - live KXBTC event, brackets with status=active and ask in [2, 98] cents
  - edge = model_fair - ask >= threshold (default from MIN_EDGE_THRESHOLD)
  - skip if < min_minutes_to_expiry left, or max open trades on this event,
    or the top ticker is already held open
  - contracts fixed per cycle (paper sizing; Kelly comes later with real money)

Every cycle is logged to SQLite (auto_cycles) whether it buys or skips, so a
24h paper run produces a full audit trail. Open trades auto-settle vs Kalshi.

Standalone:
    python src/auto_trader.py --once                 # single cycle now
    python src/auto_trader.py --interval 20          # loop every 20 min
Also importable by app.py (AutoTrader thread with start/stop).
"""
import os
import time
import argparse
import threading
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

load_dotenv()

import kalshi_client as K
import signals as S
import tracker as T
from economics import net_edge, all_in_ask_cents
from sizing import size_contracts

TRADE_ASSETS = ("BTC", "ETH")

CYCLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS auto_config(
  id INTEGER PRIMARY KEY CHECK(id=1),
  enabled INTEGER NOT NULL DEFAULT 0,
  interval_min INTEGER NOT NULL DEFAULT 15,
  threshold REAL NOT NULL DEFAULT 0.03,
  contracts INTEGER NOT NULL DEFAULT 10,  -- Kelly cap per trade
  max_per_event INTEGER NOT NULL DEFAULT 2,
  min_minutes_to_expiry REAL NOT NULL DEFAULT 5
);
CREATE TABLE IF NOT EXISTS auto_cycles(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL, event_ticker TEXT, action TEXT NOT NULL,  -- buy | skip | error
  detail TEXT, trade_id INTEGER,
  spot REAL, pred_price REAL, minutes_to_expiry REAL
);
"""


def _db():
    import sqlite3
    c = sqlite3.connect(T.DB_PATH)
    c.row_factory = sqlite3.Row
    c.executescript(T.SCHEMA + CYCLE_SCHEMA)
    for col, default in (("min_confidence", "0.10"), ("max_minutes_in", "5")):
        try:
            c.execute(f"ALTER TABLE auto_config ADD COLUMN {col} REAL NOT NULL DEFAULT {default}")
        except Exception:
            pass
    return c


def get_config() -> dict:
    with _db() as c:
        r = c.execute("SELECT * FROM auto_config WHERE id=1").fetchone()
        if not r:
            c.execute("INSERT INTO auto_config(id) VALUES(1)")
            r = c.execute("SELECT * FROM auto_config WHERE id=1").fetchone()
        d = dict(r)
    d["threshold"] = float(os.getenv("MIN_EDGE_THRESHOLD", d["threshold"]))
    return d


def set_config(**kw) -> dict:
    allowed = {"enabled", "interval_min", "threshold", "contracts",
               "max_per_event", "min_minutes_to_expiry", "min_confidence",
               "max_minutes_in"}
    kw = {k: v for k, v in kw.items() if k in allowed}
    if not kw:
        return get_config()
    with _db() as c:
        c.execute("INSERT INTO auto_config(id) VALUES(1) ON CONFLICT(id) DO NOTHING")
        c.execute("UPDATE auto_config SET " + ",".join(f"{k}=?" for k in kw) +
                  " WHERE id=1", tuple(kw.values()))
    return get_config()


def log_cycle(action: str, detail: str = "", trade_id=None, event=None,
              spot=None, pred=None, mins=None):
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _db() as c:
        c.execute("INSERT INTO auto_cycles(ts,event_ticker,action,detail,trade_id,"
                  "spot,pred_price,minutes_to_expiry) VALUES(?,?,?,?,?,?,?,?)",
                  (ts, event, action, detail, trade_id, spot, pred, mins))


def pick_position(pred_price: float, mins_left: float, brackets: list[dict],
                  threshold: float, spot: float, p_stack: float,
                  min_conf: float) -> dict | None:
    """LEGACY bracket picker (hourly KXBTC, no longer used by run_cycle)."""
    if abs(p_stack - 0.5) < min_conf:
        return None
    side = "above" if p_stack > 0.5 else "below"
    fairs = S.bracket_fair_values(pred_price, mins_left, brackets)
    cands = []
    for b in brackets:
        if b["status"] != "active" or b["yes_ask"] is None:
            continue
        if not 2 <= b["yes_ask"] <= 98:
            continue
        fl = b["floor"]
        if side == "above":
            if not (b["strike_type"] == "greater" or (fl is not None and fl >= spot - 50)):
                continue
        else:
            if not (b["strike_type"] == "less" or (fl is not None and fl < spot + 50)):
                continue
        edge = net_edge(fairs.get(b["ticker"], 0.0), b["yes_ask"])
        if edge >= threshold:
            cands.append({"bracket": b, "fair": round(fairs[b["ticker"]], 4),
                          "edge": round(edge, 4), "side": side})
    cands.sort(key=lambda x: (x["edge"], x["bracket"].get("volume_24h") or 0), reverse=True)
    return cands[0] if cands else None


def pick_15m(p_stack: float, m: dict, threshold: float, conf: float) -> dict | None:
    """15-min up/down pick. Returns {side, paid, fair, edge} or None.

    fair YES = p_stack, fair NO = 1 - p_stack. Cost = ask + Kalshi taker fee.
    Only the conviction side is ever considered (60/40 rule).
    """
    if abs(p_stack - 0.5) < conf:
        return None
    side = "yes" if p_stack > 0.5 else "no"
    paid = m["yes_ask"] if side == "yes" else m["no_ask"]
    if paid is None or not 2 <= paid <= 98:
        return None
    fair = p_stack if side == "yes" else 1 - p_stack
    edge = net_edge(fair, paid)
    if edge < threshold:
        return None
    return {"side": side, "paid": paid, "fair": round(fair, 4), "edge": round(edge, 4)}


def run_cycle(cfg: dict | None = None) -> dict:
    """One 15-min up/down cycle across TRADE_ASSETS. Never raises."""
    cfg = cfg or get_config()
    max_in = float(cfg.get("max_minutes_in", 5))
    results: list[dict] = []
    try:
        T.resolve_open_trades()
        import risk as R
        dp = T.day_pnl()
        halt, reason = R.check_daily_halt(dp["total_cents"])
        if halt:
            T.trip_halt("all", reason)
            set_config(enabled=0)
            log_cycle("halt", reason)
            return {"action": "halt", "reason": reason, "results": results}
        tp = T.trailing_perf(R.DRIFT_N, "crypto")
        halt, reason = R.check_drift(tp["wins"], tp["n"], tp["mean_implied"])
        if halt:
            T.trip_halt("crypto", reason)
            set_config(enabled=0)
            log_cycle("halt", reason)
            return {"action": "halt", "reason": reason, "results": results}
        if _error_streak(3):
            reason = "3 consecutive cycle errors"
            T.trip_halt("crypto", reason)
            set_config(enabled=0)
            log_cycle("halt", reason)
            return {"action": "halt", "reason": reason, "results": results}
        try:
            from fetch_data import refresh_all
            refresh_all()  # force-fresh bars so p is for the just-closed candle
        except Exception:
            pass
        conf = float(cfg.get("min_confidence", 0.10))
        for asset in TRADE_ASSETS:
            try:
                results.append(_eval_asset(asset, cfg, conf, max_in))
            except Exception as e:
                log_cycle("error", f"{asset}: {type(e).__name__}: {e}")
                results.append({"asset": asset, "action": "error", "reason": str(e)[:100]})
        buys = [r for r in results if r["action"] == "buy"]
        return {"action": "buy" if buys else results[0]["action"] if results else "skip",
                "results": results}
    except Exception as e:
        log_cycle("error", f"{type(e).__name__}: {e}")
        return {"action": "error", "reason": str(e)}


def _eval_asset(asset: str, cfg: dict, conf: float, max_in: float) -> dict:
    pred = S.asset_signal(asset)
    p_stack = pred.get("p_stack")
    base = dict(event=None, spot=pred.get("spot"), pred=pred.get("pred_price"), mins=None)
    if p_stack is None:
        log_cycle("skip", f"{asset}: direction model not trained (no p_stack)", **base)
        return {"asset": asset, "action": "skip", "reason": "no_stack_signal", **base}

    event = K.current_15m_ticker(asset)
    try:
        m = K.get_15m(event)
    except Exception as e:
        log_cycle("error", f"{event}: {e}", **{**base, "event": event})
        return {"asset": asset, "action": "error", "reason": str(e)[:100], **{**base, "event": event}}
    # stale-signal breaker: bars older than 30m invalidate the decision
    try:
        from datetime import datetime, timezone as _tz
        asof = pred.get("asof")
        age = (datetime.now(_tz.utc) - datetime.fromisoformat(str(asof))).total_seconds() / 60 if asof else None
    except Exception:
        age = None
    import risk as R
    halt, reason = R.check_fresh(age)
    if halt:
        log_cycle("skip", f"{asset}: {reason}", **{**base, "event": event})
        return {"asset": asset, "action": "skip", "reason": "stale_data", **{**base, "event": event}}
    mins_left = S.minutes_to_close(m["close_time"]) or 0.0
    mins_in = round(15 - mins_left, 1)
    base = dict(event=event, spot=pred.get("spot"), pred=pred.get("pred_price"), mins=round(mins_left, 1))
    if m["status"] != "active":
        log_cycle("skip", f"{event} status={m['status']}", **base)
        return {"asset": asset, "action": "skip", "reason": "not_active", **base}
    if mins_in > max_in:
        log_cycle("skip", f"{asset} {mins_in}m into window (>{max_in}m, signal stale)", **base)
        return {"asset": asset, "action": "skip", "reason": "window_stale", "p_stack": p_stack, **base}

    if abs(p_stack - 0.5) < conf:
        log_cycle("skip", f"{asset} low conviction p_up={p_stack:.3f} (need |p-.5|≥{conf})", **base)
        return {"asset": asset, "action": "skip", "reason": "low_conviction", "p_stack": p_stack, **base}

    with _db() as c:
        n_open = c.execute(
            "SELECT COUNT(*) n FROM paper_trades WHERE status='open' AND event_ticker=?",
            (event,)).fetchone()["n"]
    if n_open >= 1:  # one market per 15m event; size via contracts
        log_cycle("skip", f"already holding {event}", **base)
        return {"asset": asset, "action": "skip", "reason": "already_holding", "p_stack": p_stack, **base}

    pick = pick_15m(p_stack, m, cfg["threshold"], conf)
    if not pick:
        log_cycle("skip", f"{asset} p_up={p_stack:.3f} but no fee-aware edge ≥ {cfg['threshold']}", **base)
        return {"asset": asset, "action": "skip", "reason": "below_threshold", "p_stack": p_stack, **base}

    from economics import all_in_ask_cents
    fair_p = pick["fair"] if pick["side"] == "yes" else 1 - pick["fair"]
    sizing = size_contracts(fair_p, all_in_ask_cents(pick["paid"]),
                            max_contracts=cfg["contracts"])
    n = sizing["contracts"] or 1
    halt, reason = R.check_exposure(T.day_pnl()["n_open"], n * pick["paid"])
    if halt:
        log_cycle("skip", f"{asset}: exposure cap: {reason}", **base)
        return {"asset": asset, "action": "skip", "reason": "exposure_cap", **base}
    tid = T.record_paper_trade(
        event_ticker=event, market_ticker=m["ticker"], side=pick["side"],
        contracts=n, price_paid_cents=pick["paid"],
        pred_price=pred.get("pred_price"), spot=pred.get("spot"), p_up=p_stack,
        edge=pick["edge"], minutes_to_expiry=round(mins_left, 1),
        model_version=T.model_version(f"dir_{asset}"))
    detail = (f"{asset} p_up={p_stack:.3f} | {pick['side'].upper()} {m['ticker']} "
              f"x{n} @ {pick['paid']}¢ fair={pick['fair']} edge_net={pick['edge']} "
              f"kelly={sizing['kelly_f']}")
    log_cycle("buy", detail, trade_id=tid, **base)
    return {"asset": asset, "action": "buy", "trade_id": tid, "ticker": m["ticker"], **pick,
            "p_stack": p_stack, "contracts": n, **base}


def _error_streak(k: int = 3) -> bool:
    with _db() as c:
        rows = [r[0] for r in c.execute(
            "SELECT action FROM auto_cycles ORDER BY id DESC LIMIT ?", (k,))]
    return len(rows) >= k and all(a == "error" for a in rows)


def recent_cycles(limit: int = 50) -> list[dict]:
    with _db() as c:
        return [dict(r) for r in c.execute(
            "SELECT * FROM auto_cycles ORDER BY id DESC LIMIT ?", (limit,))]


def secs_to_next_window(offset_s: int = 45) -> float:
    """Seconds until next :00/:15/:30/:45 UTC + offset (fresh candle + open book)."""
    now = datetime.now(timezone.utc)
    minute = (now.minute // 15 + 1) * 15
    nxt = now.replace(second=0, microsecond=0)
    if minute >= 60:
        nxt = (nxt.replace(minute=0) + timedelta(hours=1))
    else:
        nxt = nxt.replace(minute=minute)
    return max((nxt - now).total_seconds() + offset_s, 1)


class AutoTrader(threading.Thread):
    """Background loop for app.py. Phase-aligned to 15-min windows."""

    def __init__(self):
        super().__init__(daemon=True)
        self._stop_event = threading.Event()
        self.state = {"running": False, "last": None, "next_in_s": None}

    def run(self):
        self.state["running"] = True
        self._stop_event.wait(secs_to_next_window())
        while not self._stop_event.is_set():
            cfg = get_config()
            if not cfg["enabled"]:
                break
            t0 = time.monotonic()
            try:
                self.state["last"] = run_cycle(cfg)
            except Exception as e:
                self.state["last"] = {"action": "error", "reason": str(e)}
            wait = cfg["interval_min"] * 60 - (time.monotonic() - t0)
            self.state["next_in_s"] = int(max(wait, 0))
            self._stop_event.wait(max(wait, 0))
        self.state["running"] = False
        self.state["next_in_s"] = None

    def stop(self):
        set_config(enabled=0)
        self._stop_event.set()


_trader: AutoTrader | None = None


def start_trader(**cfg) -> dict:
    global _trader
    set_config(enabled=1, **cfg)
    if _trader is None or not _trader.is_alive():
        _trader = AutoTrader()
        _trader.start()
    return status()


def stop_trader() -> dict:
    global _trader
    set_config(enabled=0)
    if _trader and _trader.is_alive():
        _trader._stop_event.set()
    return status()


def status() -> dict:
    cfg = get_config()
    running = bool(_trader and _trader.is_alive() and cfg["enabled"])
    return {"config": cfg, "thread_running": running,
            "last": _trader.state["last"] if _trader else None,
            "next_in_s": _trader.state["next_in_s"] if _trader else None,
            "paper": T.paper_stats()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=int, default=15)
    ap.add_argument("--threshold", type=float, default=float(os.getenv("MIN_EDGE_THRESHOLD", "0.05")))
    ap.add_argument("--contracts", type=int, default=1)
    a = ap.parse_args()
    set_config(interval_min=a.interval, threshold=a.threshold, contracts=a.contracts)
    if a.once:
        print(run_cycle())
        return
    set_config(enabled=1)
    print(f"auto-trader aligned to 15m windows, every {a.interval} min (Ctrl-C to stop)")
    try:
        time.sleep(secs_to_next_window())
        while True:
            t0 = time.monotonic()
            print(run_cycle(), flush=True)
            time.sleep(max(a.interval * 60 - (time.monotonic() - t0), 0))
    except KeyboardInterrupt:
        set_config(enabled=0)
        print("stopped")


if __name__ == "__main__":
    main()
