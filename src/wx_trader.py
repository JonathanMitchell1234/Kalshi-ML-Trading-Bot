"""Auto paper-trader for Kalshi daily-high weather brackets (NYC + Chicago).

Cadence: 4x/day at 00/06/12/18Z + 40 min (post-GFS). Each cycle, per city,
for each open event (today + tomorrow): Bayesian bracket probabilities ->
best fee-aware YES edge >= threshold -> at most 1 open position per event.
All decisions logged (buys + skip reasons). Settles vs Kalshi `result`.
Standalone:  python src/wx_trader.py --once | --loop
"""
import os
import time
import argparse
import threading
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

load_dotenv()

import pandas as pd
import kalshi_client as K
import tracker as T
from economics import net_edge, all_in_ask_cents
from sizing import size_contracts
from wx_data import CITIES
from wx_signals import city_probs

CYCLE = """
CREATE TABLE IF NOT EXISTS wx_config(
  id INTEGER PRIMARY KEY CHECK(id=1),
  enabled INTEGER NOT NULL DEFAULT 0,
  threshold REAL NOT NULL DEFAULT 0.03,
  contracts INTEGER NOT NULL DEFAULT 10  -- Kelly cap per trade
);
CREATE TABLE IF NOT EXISTS wx_cycles(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL, city TEXT, event_ticker TEXT, action TEXT NOT NULL,
  detail TEXT, trade_id INTEGER, pred REAL, sigma REAL
);
"""


def _db():
    import sqlite3
    c = sqlite3.connect(T.DB_PATH)
    c.row_factory = sqlite3.Row
    c.executescript(T.SCHEMA + CYCLE)
    for _col, _dflt in (("max_gfs_gap", "6.0"), ("threshold_t1", "0.05")):
        try:
            c.execute(f"ALTER TABLE wx_config ADD COLUMN {_col} REAL NOT NULL DEFAULT {_dflt}")
        except Exception:
            pass
    return c


def get_config() -> dict:
    with _db() as c:
        r = c.execute("SELECT * FROM wx_config WHERE id=1").fetchone()
        if not r:
            c.execute("INSERT INTO wx_config(id) VALUES(1)")
            r = c.execute("SELECT * FROM wx_config WHERE id=1").fetchone()
        return dict(r)


def set_config(**kw) -> dict:
    allowed = {"enabled", "threshold", "threshold_t1", "contracts", "max_gfs_gap"}
    kw = {k: v for k, v in kw.items() if k in allowed}
    with _db() as c:
        c.execute("INSERT INTO wx_config(id) VALUES(1) ON CONFLICT(id) DO NOTHING")
        if kw:
            c.execute("UPDATE wx_config SET " + ",".join(f"{k}=?" for k in kw) + " WHERE id=1",
                      tuple(kw.values()))
    return get_config()


def log_cycle(action, detail="", trade_id=None, city=None, event=None, pred=None, sigma=None):
    with _db() as c:
        c.execute("INSERT INTO wx_cycles(ts,city,event_ticker,action,detail,trade_id,pred,sigma)"
                  " VALUES(?,?,?,?,?,?,?,?)",
                  (datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   city, event, action, detail, trade_id, pred, sigma))


def _parse(m: dict) -> dict:
    def _c(x):
        try:
            return int(round(float(x) * 100))
        except (TypeError, ValueError):
            return None
    return {"ticker": m.get("ticker"), "subtitle": m.get("subtitle"),
            "strike_type": m.get("strike_type"), "floor": m.get("floor_strike"),
            "cap": m.get("cap_strike"), "status": m.get("status"),
            "yes_bid": _c(m.get("yes_bid_dollars")), "yes_ask": _c(m.get("yes_ask_dollars")),
            "close_time": m.get("close_time")}


def _event_for(series: str, date) -> str:
    d = date if hasattr(date, "strftime") else datetime.strptime(str(date)[:10], "%Y-%m-%d")
    return f"{series}-{d.strftime('%y').upper()}{d.strftime('%b').upper()}{d.day:02d}"


def _eval_city_date(city: str, target, cfg: dict) -> dict:
    series = CITIES[city]["series"]
    event = _event_for(series, target)
    base = dict(city=city, event=event, pred=None, sigma=None)
    try:
        ev = K.get_event(event)
    except Exception as e:
        log_cycle("skip", f"{event} not found", city=city, event=event)
        return {"action": "skip", "reason": "no_event", **base}
    brackets = [_parse(m) for m in ev.get("markets", []) if m.get("status") == "active"]
    if not brackets:
        from collections import Counter as _C
        stat = dict(_C(m.get("status", "?") for m in ev.get("markets", [])))
        n = len(ev.get("markets", []))
        if n and set(stat) <= {"initialized"}:
            detail = f"{event}: {n} brackets listed but not open yet {stat} — Kalshi activates them closer to the day; retry next cycle"
        else:
            detail = f"{event}: no active brackets (found {stat or 'no markets'})"
        log_cycle("skip", detail, city=city, event=event)
        return {"action": "skip", "reason": "not_active", **base}
    try:
        pred, fairs = city_probs(city, target, brackets)
    except Exception as e:
        log_cycle("error", f"{city} {target}: {e}", city=city, event=event)
        return {"action": "error", "reason": str(e)[:100], **base}
    base.update(pred=pred["pred"], sigma=pred["sigma"])
    ref = pred.get("ndfd_max") or pred.get("gfs_max")  # NWS official first
    ref_name = "NDFD" if pred.get("ndfd_max") is not None else "GFS"
    gap = float(cfg.get("max_gfs_gap", 6.0))
    if ref is not None and abs(pred["pred"] - ref) > gap:
        log_cycle("skip", f"model {pred['pred']}F vs {ref_name} {ref}F: NWP regime disagreement, stand down", **base)
        return {"action": "skip", "reason": "nwp_disagree", **base}
    # T+1+ forecast skill decays: demand a bigger edge for tomorrow
    from wx_data import CITIES as _CC
    thresh = cfg["threshold"]
    try:
        if pd.Timestamp(target).tz_localize(None).date() > _local_today(_CC[city]["tz"]):
            thresh = max(thresh, float(cfg.get("threshold_t1", 0.05)))
    except Exception:
        pass
    with _db() as c:
        n_open = c.execute("SELECT COUNT(*) n FROM paper_trades WHERE status='open' AND event_ticker=?",
                           (event,)).fetchone()["n"]
    if n_open >= 1:
        log_cycle("skip", f"already holding {event}", **base)
        return {"action": "skip", "reason": "already_holding", **base}
    cands = []
    for b in brackets:
        if b["yes_ask"] is None or not 2 <= b["yes_ask"] <= 98:
            continue
        edge = net_edge(fairs.get(b["ticker"], 0.0), b["yes_ask"])
        if edge >= thresh:
            cands.append((edge, b, fairs[b["ticker"]]))
    if not cands:
        best = max(((fairs.get(b["ticker"], 0.0) - (b["yes_ask"] or 100) / 100.0)
                    for b in brackets if b["yes_ask"]), default=None)
        log_cycle("skip", f"pred={pred['pred']}F σ={pred['sigma']} gfs={pred['gfs_max']} best_net~{best:.3f}" if best is not None else "no priced brackets", **base)
        return {"action": "skip", "reason": "below_threshold", **base}
    cands.sort(reverse=True)
    edge, b, fair = cands[0]
    sizing = size_contracts(fair, all_in_ask_cents(b["yes_ask"]),
                                max_contracts=cfg["contracts"])
    n = sizing["contracts"] or 1
    tid = T.record_paper_trade(
        event_ticker=event, market_ticker=b["ticker"], side="yes",
        contracts=n, price_paid_cents=b["yes_ask"],
        pred_price=pred["pred"], spot=pred["gfs_max"], p_up=fair,
        edge=round(edge, 4), minutes_to_expiry=None,
        model_version=T.model_version(f"wx_{city}"))
    detail = (f"pred={pred['pred']}F σ={pred['sigma']} gfs={pred['gfs_max']} | YES {b['ticker']} "
              f"({b['subtitle'] or b['floor']}-{b['cap']}) x{n} @ {b['yes_ask']}¢ "
              f"fair={fair:.3f} edge_net={edge:.3f} kelly={sizing['kelly_f']}")
    log_cycle("buy", detail, trade_id=tid, **base)
    return {"action": "buy", "trade_id": tid, "ticker": b["ticker"],
            "paid": b["yes_ask"], "fair": round(fair, 4), "edge": round(edge, 4), **base}


def _local_today(tz: str):
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo(tz)).date()
    except Exception:
        return datetime.now(timezone.utc).date()


def open_targets() -> list[tuple[str, object]]:
    """(city, date) pairs with plausibly-open events: today + tomorrow local."""
    out = []
    for city in ("NYC", "CHI"):
        local_today = _local_today(CITIES[city]["tz"])
        out.append((city, local_today))
        out.append((city, local_today + timedelta(days=1)))
    return out


def run_cycle(cfg: dict | None = None) -> dict:
    cfg = cfg or get_config()
    T.resolve_open_trades()
    results = []
    for city, target in open_targets():
        try:
            results.append({"city": city, "target": str(target),
                            **_eval_city_date(city, target, cfg)})
        except Exception as e:
            log_cycle("error", f"{city} {target}: {e}", city=city)
            results.append({"city": city, "action": "error", "reason": str(e)[:100]})
    buys = [r for r in results if r["action"] == "buy"]
    return {"action": "buy" if buys else "skip", "results": results}


def recent_cycles(limit: int = 50) -> list[dict]:
    with _db() as c:
        return [dict(r) for r in c.execute(
            "SELECT * FROM wx_cycles ORDER BY id DESC LIMIT ?", (limit,))]


def secs_to_next_gfs(offset_s: int = 2400) -> float:
    now = datetime.now(timezone.utc)
    nxt = (now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
    while nxt.hour not in (0, 6, 12, 18) or nxt <= now:
        nxt += timedelta(hours=1)
    return max((nxt - now).total_seconds() + offset_s, 1)


_trader = None


class WxTrader(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._stop_event = threading.Event()
        self.state = {"running": False, "last": None, "next_in_s": None}

    def run(self):
        self.state["running"] = True
        self._stop_event.wait(secs_to_next_gfs())
        while not self._stop_event.is_set():
            cfg = get_config()
            if not cfg["enabled"]:
                break
            t0 = time.monotonic()
            try:
                self.state["last"] = run_cycle(cfg)
            except Exception as e:
                self.state["last"] = {"action": "error", "reason": str(e)}
            wait = 6 * 3600 - (time.monotonic() - t0)
            self.state["next_in_s"] = int(max(wait, 0))
            self._stop_event.wait(max(wait, 0))
        self.state["running"] = False
        self.state["next_in_s"] = None

    def stop(self):
        set_config(enabled=0)
        self._stop_event.set()


def start_trader(**cfg) -> dict:
    global _trader
    set_config(enabled=1, **cfg)
    if _trader is None or not _trader.is_alive():
        _trader = WxTrader()
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
            "cycles": recent_cycles(20)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.03)
    ap.add_argument("--contracts", type=int, default=1)
    a = ap.parse_args()
    set_config(threshold=a.threshold, contracts=a.contracts)
    if a.once or not a.loop:
        import json
        print(json.dumps(run_cycle(), indent=1, default=str))
        return
    set_config(enabled=1)
    print("wx paper-trader on GFS cadence (Ctrl-C to stop)")
    try:
        time.sleep(secs_to_next_gfs())
        while True:
            print(run_cycle(), flush=True)
            cfg = get_config()
            if not cfg["enabled"]:
                break
            time.sleep(6 * 3600)
    except KeyboardInterrupt:
        set_config(enabled=0)


if __name__ == "__main__":
    main()
