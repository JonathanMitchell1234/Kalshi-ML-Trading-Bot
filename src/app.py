"""BitBot dashboard: live Kalshi BTC brackets + GBM signals + paper-trade W/L.

Run:  python src/app.py   (or: uvicorn src.app:app --reload)
Open: http://127.0.0.1:8000
"""
import time
import threading
from pathlib import Path
from datetime import datetime as _dt
from datetime import timedelta as _td

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import kalshi_client as K
import signals as S
import tracker as T
import auto_trader as A
import wx_trader as W
from economics import net_edge, all_in_ask_cents

_STARTED = time.monotonic()

app = FastAPI(title="BitBot Dashboard")
STATIC = Path(__file__).resolve().parent / "static"

_cache: dict = {}
_backtest_job: dict = {"state": "idle"}


def cached(key: str, ttl: int, fn):
    now = time.time()
    if key in _cache and now - _cache[key][0] < ttl:
        return _cache[key][1]
    val = fn()
    _cache[key] = (now, val)
    return val


def overview() -> dict:
    pred = S.latest_prediction()
    event = K.current_event_ticker()
    meta, brackets = K.get_brackets(event, near_floor=pred["spot"], window=100000)
    close = next((b["close_time"] for b in brackets if b["close_time"]), None)
    mins = S.minutes_to_close(close)
    act = [b for b in brackets if b["status"] == "active"]
    d = {
        "spot": pred["spot"], "pred_price": round(pred["pred_price"], 2),
        "p_up": round(pred["p_up"], 4), "asof": pred["asof"],
        "p_stack": pred.get("p_stack"), "p_lgbm": pred.get("p_lgbm"),
        "p_gru": pred.get("p_gru"),
        "event": event, "event_title": meta.get("title"),
        "close_time": close, "minutes_to_close": round(mins, 1) if mins else None,
        "n_active": len(act),
    }
    try:
        ud = {}
        for _asset in ("BTC", "ETH"):
            try:
                u = K.get_15m(K.current_15m_ticker(_asset))
            except Exception as e:
                ud[_asset] = {"error": str(e)[:150]}
                continue
            umins = S.minutes_to_close(u["close_time"])
            p = (pred.get("p_stack") if _asset == "BTC"
                 else S.asset_signal(_asset).get("p_stack"))
            uy, un = None, None
            if p is not None and u["yes_ask"] is not None:
                uy = {"fair": round(p, 4), "edge": round(net_edge(p, u["yes_ask"]), 4)}
            if p is not None and u["no_ask"] is not None:
                un = {"fair": round(1 - p, 4), "edge": round(net_edge(1 - p, u["no_ask"]), 4)}
            ud[_asset] = {**u, "minutes_to_close": round(umins, 1) if umins else None,
                          "p_stack": round(p, 4) if p is not None else None,
                          "yes": uy, "no": un}
        d["updown"] = ud
    except Exception as e:
        d["updown"] = {"error": str(e)[:150]}
    return d


@app.get("/")
def index():
    return FileResponse(STATIC / "dashboard.html")


@app.get("/api/overview")
def api_overview():
    try:
        return cached("overview", 60, overview)
    except Exception as e:
        raise HTTPException(502, f"overview failed: {e}")


@app.get("/api/markets")
def api_markets():
    def _fn():
        ov = cached("overview", 60, overview)
        _, brackets = K.get_brackets(ov["event"], near_floor=ov["spot"], window=100000)
        mins = ov["minutes_to_close"] or 30
        fairs = S.bracket_fair_values(ov["pred_price"], mins, brackets)
        rows = []
        for b in brackets:
            if b["status"] != "active":
                continue
            fair = round(fairs.get(b["ticker"], 0.0), 4)
            ask = b["yes_ask"]
            rows.append({**b, "fair": fair,
                         "ask_allin": round(all_in_ask_cents(ask), 2) if ask is not None else None,
                         "edge": round(net_edge(fair, ask), 4) if ask is not None else None})
        # near-money slice: model-implied mass or a real two-sided price
        def _liquid(r):
            ask, bid = r["yes_ask"], r["yes_bid"]
            return (ask is not None and 2 <= ask <= 98) or (bid is not None and bid >= 2)
        rows = [r for r in rows if r["fair"] > 0.005 or _liquid(r)]
        rows.sort(key=lambda r: (r["floor"] is None, r["floor"] or 0))
        return {"event": ov["event"], "pred_price": ov["pred_price"],
                "minutes_to_close": mins, "brackets": rows}
    try:
        return cached("markets", 20, _fn)
    except Exception as e:
        raise HTTPException(502, f"markets failed: {e}")


class PaperOrder(BaseModel):
    market_ticker: str
    side: str = "yes"
    contracts: int = 1


@app.post("/api/paper_trade")
def api_paper_trade(o: PaperOrder):
    if o.side not in ("yes", "no"):
        raise HTTPException(400, "side must be yes/no")
    if not 1 <= o.contracts <= 1000:
        raise HTTPException(400, "contracts 1..1000")
    try:
        if "15M" in o.market_ticker:
            m = K.get_market(o.market_ticker)
            b = K.parse_updown(m, m.get("event_ticker", ""))
            if b["status"] != "active":
                raise HTTPException(400, f"market not active (status={b['status']})")
            px = b["yes_ask"] if o.side == "yes" else b["no_ask"]
            if px is None:
                raise HTTPException(400, "no price available")
            ov = cached("overview", 60, overview)
            p = ov.get("p_stack")
            fair = (p if o.side == "yes" else 1 - p) if p is not None else 0.5
            tid = T.record_paper_trade(
                event_ticker=b["event"], market_ticker=b["ticker"], side=o.side,
                contracts=o.contracts, price_paid_cents=px,
                pred_price=ov["pred_price"], spot=ov["spot"], p_up=p,
                edge=round(net_edge(fair, px), 4),
                minutes_to_expiry=(ov.get("updown") or {}).get("minutes_to_close"))
            return {"trade_id": tid, "paid_cents": px, "fair": round(fair, 4)}
        m = K.get_market(o.market_ticker)
        b = K.parse_bracket(m)
    except Exception as e:
        raise HTTPException(502, f"kalshi fetch failed: {e}")
    if b["status"] != "active":
        raise HTTPException(400, f"market not active (status={b['status']})")
    px = b["yes_ask"] if o.side == "yes" else (100 - (b["yes_bid"] or 0))
    if px is None:
        raise HTTPException(400, "no price available")
    ov = cached("overview", 60, overview)
    fairs = S.bracket_fair_values(ov["pred_price"], ov["minutes_to_close"] or 30, [b])
    fair = fairs.get(b["ticker"], 0.5)
    fair_side = fair if o.side == "yes" else 1 - fair
    tid = T.record_paper_trade(
        event_ticker=m.get("event_ticker", ""), market_ticker=b["ticker"], side=o.side,
        contracts=o.contracts, price_paid_cents=px if o.side == "yes" else 100 - (b["yes_bid"] or 0),
        pred_price=ov["pred_price"], spot=ov["spot"], p_up=ov["p_up"],
        edge=round(fair_side - px / 100.0, 4), minutes_to_expiry=ov["minutes_to_close"])
    return {"trade_id": tid, "paid_cents": px, "fair": round(fair_side, 4)}


@app.get("/api/trades")
def api_trades(limit: int = 100):
    T.resolve_open_trades()
    return {"trades": T.list_trades(limit)}


@app.get("/api/stats")
def api_stats():
    return {"paper": T.paper_stats(), "backtest": T.backtest_stats()}


@app.post("/api/resolve")
def api_resolve():
    return {"updated": T.resolve_open_trades(), "stats": T.paper_stats()}


class BacktestReq(BaseModel):
    n_events: int = 48
    lead_min: int = 30


def _run_backtest_bg(n_events: int, lead_min: int):
    _backtest_job.update(state="running", done=0, total=n_events)
    try:
        from backtest import run_backtest
        res = run_backtest(n_events=n_events, lead_min=lead_min,
                           progress=lambda a, b: _backtest_job.update(done=a, total=b))
        _backtest_job.update(state="done", result=res)
    except Exception as e:
        _backtest_job.update(state="error", error=str(e))


class Backtest15Req(BaseModel):
    n_events: int = 200
    asset: str = "BTC"


def _run_backtest15_bg(n_events: int, asset: str):
    _backtest_job.update(state="running", done=0, total=n_events)
    try:
        from backtest_15m import run_backtest_15m
        res = run_backtest_15m(n_events=n_events,
                               progress=lambda a, b: _backtest_job.update(done=a, total=b),
                               asset=asset)
        _backtest_job.update(state="done", result=res)
    except Exception as e:
        _backtest_job.update(state="error", error=str(e))


@app.post("/api/backtest15")
def api_backtest15(req: Backtest15Req):
    if _backtest_job.get("state") == "running":
        raise HTTPException(409, "backtest already running")
    if req.asset not in ("BTC", "ETH"):
        raise HTTPException(400, "asset BTC|ETH")
    if not 1 <= req.n_events <= 400:
        raise HTTPException(400, "n_events 1..400")
    th = threading.Thread(target=_run_backtest15_bg,
                          args=(req.n_events, req.asset), daemon=True)
    th.start()
    return {"started": True, "asset": req.asset}


@app.post("/api/backtest")
def api_backtest(req: BacktestReq):
    if _backtest_job.get("state") == "running":
        raise HTTPException(409, "backtest already running")
    if not 1 <= req.n_events <= 200:
        raise HTTPException(400, "n_events 1..200")
    th = threading.Thread(target=_run_backtest_bg, args=(req.n_events, req.lead_min), daemon=True)
    th.start()
    return {"started": True}


@app.get("/api/backtest/status")
def api_backtest_status():
    return {**_backtest_job, "latest": T.backtest_stats()}


@app.get("/api/kalshi/balance")
def api_balance():
    try:
        return K.KalshiAuth().balance()
    except Exception as e:
        raise HTTPException(502, f"auth/balance failed: {e}")


class AutoCfg(BaseModel):
    interval_min: int = 15
    threshold: float = 0.05
    contracts: int = 1
    max_per_event: int = 2
    min_minutes_to_expiry: float = 5
    min_confidence: float = 0.10
    max_minutes_in: float = 5


@app.get("/api/autotrade/status")
def api_autotrade_status():
    return {**A.status(), "cycles": A.recent_cycles(20)}


@app.post("/api/autotrade/start")
def api_autotrade_start(cfg: AutoCfg):
    if not 5 <= cfg.interval_min <= 60:
        raise HTTPException(400, "interval_min 5..60")
    if not 0 <= cfg.threshold <= 0.9:
        raise HTTPException(400, "threshold 0..0.9")
    if not 1 <= cfg.contracts <= 100:
        raise HTTPException(400, "contracts 1..100")
    if not 0 <= cfg.min_confidence <= 0.45:
        raise HTTPException(400, "min_confidence 0..0.45")
    if not 1 <= cfg.max_minutes_in <= 14:
        raise HTTPException(400, "max_minutes_in 1..14")
    return A.start_trader(interval_min=cfg.interval_min, threshold=cfg.threshold,
                          contracts=cfg.contracts, max_per_event=cfg.max_per_event,
                          min_minutes_to_expiry=cfg.min_minutes_to_expiry,
                          min_confidence=cfg.min_confidence,
                          max_minutes_in=cfg.max_minutes_in)


@app.post("/api/autotrade/stop")
def api_autotrade_stop():
    return A.stop_trader()


@app.post("/api/autotrade/cycle")
def api_autotrade_cycle():
    """Run one cycle immediately (manual trigger / testing)."""
    return A.run_cycle()


@app.get("/api/eval")
def api_eval():
    import evaluate as E
    return E.summary()


@app.get("/api/health")
def api_health():
    """Single status snapshot for the UI shell: freshness, regime, versions."""
    import time as _t
    from datetime import datetime, timezone
    out: dict = {"server_time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "kalshi_ok": None, "vol": None, "data_age_min": {},
                 "versions": {}, "traders": {}}
    try:
        K.get_market("KXBTC15M-26SEP042115-15")
        out["kalshi_ok"] = True
    except Exception:
        try:
            K.get_event("KXBTC-26SEP0421")
            out["kalshi_ok"] = True
        except Exception:
            out["kalshi_ok"] = False
    try:
        from fetch_data import load_or_fetch
        import pandas as _pd
        b = load_or_fetch("BTC", "1m", days=8)
        b["r"] = b["close"].pct_change()
        now = b["time"].max()
        v24 = float(b[b["time"] > now - _pd.Timedelta(hours=24)]["r"].rolling(15).std().mean() * 1e4)
        v7d = float(b["r"].rolling(15).std().mean() * 1e4)
        out["vol"] = {"bps_24h": round(v24, 1), "bps_7d": round(v7d, 1),
                      "label": "dead calm" if v24 < 2 else ("thin" if v24 < 3 else
                               ("normal" if v24 < 6 else "volatile")),
                      "asof": str(now)}
        out["data_age_min"] = {"btc_1m": round((datetime.now(timezone.utc) - now).total_seconds() / 60, 1)}
    except Exception as e:
        out["vol"] = {"error": str(e)[:100]}
    try:
        out["versions"] = __import__("json").loads((S.MODEL_DIR / "versions.json").read_text())
    except Exception:
        pass
    for name, mod in (("crypto", A), ("wx", W)):
        try:
            s = mod.status()
            out["traders"][name] = {"running": s["thread_running"], "next_in_s": s["next_in_s"],
                                    "last": s["last"]}
        except Exception as e:
            out["traders"][name] = {"running": False, "error": str(e)[:100]}
    out["uptime_s"] = int(_t.monotonic() - _STARTED)
    return out


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.on_event("startup")
def _warm():
    def _run():
        try:
            overview()
        except Exception as e:
            print(f"warmup failed: {e}", flush=True)
    threading.Thread(target=_run, daemon=True).start()


# ------------------------------------------------------- weather tab
class WxCfg(BaseModel):
    threshold: float = 0.03
    contracts: int = 1


@app.get("/weather")
def weather():
    return FileResponse(STATIC / "weather.html")


def _wx_parse(m: dict) -> dict:
    def _c(x):
        try:
            return int(round(float(x) * 100))
        except (TypeError, ValueError):
            return None
    return {"ticker": m.get("ticker"), "subtitle": m.get("subtitle"),
            "strike_type": m.get("strike_type"), "floor": m.get("floor_strike"),
            "cap": m.get("cap_strike"), "status": m.get("status"),
            "yes_bid": _c(m.get("yes_bid_dollars")), "yes_ask": _c(m.get("yes_ask_dollars")),
            "close_time": m.get("close_time"), "result": m.get("result") or None,
            "expiration_value": m.get("expiration_value")}


def _wx_event(series: str, date) -> str:
    d = date if hasattr(date, "strftime") else _dt.strptime(str(date)[:10], "%Y-%m-%d")
    return f"{series}-{d.strftime('%y').upper()}{d.strftime('%b').upper()}{d.day:02d}"


@app.get("/api/wx/overview")
def api_wx_overview():
    from wx_data import CITIES
    from wx_signals import city_probs
    out = {}
    for city in ("NYC", "CHI"):
        series = CITIES[city]["series"]
        city_out = {"city": city, "dates": {}}
        try:
            loc = W._local_today(CITIES[city]["tz"])
        except Exception:
            loc = _dt.now().date()
        for target in (loc, loc + _td(days=1)):
            t = _wx_event(series, target)
            try:
                ev = K.get_event(t)
                brackets = [_wx_parse(m) for m in ev.get("markets", []) if m.get("status") == "active"]
                pred, fairs = city_probs(city, target, brackets)
                rows = []
                for b in brackets:
                    ask = b["yes_ask"]
                    rows.append({**b, "fair": round(fairs.get(b["ticker"], 0.0), 4),
                                 "edge": round(net_edge(fairs.get(b["ticker"], 0.0), ask), 4)
                                 if ask is not None else None})
                city_out["dates"][str(target)] = {"event": t, "pred": pred, "brackets": rows}
            except Exception as e:
                city_out["dates"][str(target)] = {"event": t, "error": str(e)[:150]}
        out[city] = city_out
    return out


@app.post("/api/wx/backtest")
def api_wx_backtest(payload: dict = {}):
    from wx_backtest import run_wx_backtest
    city = (payload or {}).get("city", "NYC")
    n = int((payload or {}).get("n_days", 60))
    if city not in ("NYC", "CHI"):
        raise HTTPException(400, "city NYC|CHI")
    return run_wx_backtest(city, min(n, 200))


@app.get("/api/wx/auto")
def api_wx_auto():
    return W.status()


@app.post("/api/wx/auto/start")
def api_wx_auto_start(cfg: WxCfg):
    return W.start_trader(threshold=cfg.threshold, contracts=cfg.contracts)


@app.post("/api/wx/auto/stop")
def api_wx_auto_stop():
    return W.stop_trader()


@app.post("/api/wx/auto/cycle")
def api_wx_auto_cycle():
    return W.run_cycle()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
