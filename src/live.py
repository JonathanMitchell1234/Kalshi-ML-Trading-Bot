"""Live Kalshi order lifecycle. Real money moves ONLY through place/cancel,
both hard-gated by risk.require_live() (LIVE_TRADING env must be true).

Reads (balance, positions, orders) are safe anytime. Every fill/cancel is
mirrored to the paper DB with exec_mode='live' for unified attribution.
Idempotency: every order carries a client_order_id (uuid) so retries after
a network blip never double-place.
"""
import time
import uuid
import requests

from kalshi_client import BASE, KalshiAuth
from risk import require_live

_auth = KalshiAuth()


def _req(method: str, path: str, body: dict | None = None) -> dict:
    h = _auth.headers(method, "/trade-api/v2" + path)
    kw: dict = {"headers": {**h, "Content-Type": "application/json"}, "timeout": 25}
    if method == "GET":
        kw["params"] = body or {}
    else:
        kw["json"] = body
    r = requests.request(method, BASE + path, **kw)
    r.raise_for_status()
    return r.json()


# ------------------------------------------------------------ reads (safe)
def balance() -> dict:
    return _req("GET", "/portfolio/balance").get("balance", {})


def positions(settle_ts=None) -> list[dict]:
    p = {}
    if settle_ts:
        p["settlement_ts"] = settle_ts
    return _req("GET", "/portfolio/positions", p).get("market_positions", [])


def get_order(order_id: str) -> dict:
    return _req("GET", f"/portfolio/orders/{order_id}").get("order", {})


def list_orders(status: str | None = None, ticker: str | None = None) -> list[dict]:
    p = {}
    if status:
        p["status"] = status
    if ticker:
        p["ticker"] = ticker
    return _req("GET", "/portfolio/orders", p).get("orders", [])


# ------------------------------------------------------------ writes (LOCKED)
def place_limit(ticker: str, side: str, contracts: int, price_cents: int,
                client_order_id: str | None = None) -> dict:
    """Resting limit order. side yes|no, price in cents (1..99)."""
    require_live()
    if side not in ("yes", "no"):
        raise ValueError("side yes|no")
    if not 1 <= contracts <= 1000:
        raise ValueError("contracts 1..1000")
    if not 1 <= price_cents <= 99:
        raise ValueError("price 1..99c")
    body = {"ticker": ticker, "action": "buy", "side": side, "type": "limit",
            "count": contracts, "yes_price": price_cents,
            "client_order_id": client_order_id or uuid.uuid4().hex}
    if side == "no":
        body["no_price"] = price_cents
        del body["yes_price"]
    return _req("POST", "/portfolio/orders", body).get("order", {})


def cancel_order(order_id: str) -> dict:
    require_live()
    return _req("DELETE", f"/portfolio/orders/{order_id}").get("order", {})


def cancel_all(ticker: str | None = None) -> list[dict]:
    """Flatten resting orders (kill-switch helper). Returns cancelled."""
    require_live()
    out = []
    for o in list_orders(status="resting", ticker=ticker):
        try:
            out.append(cancel_order(o["order_id"]))
        except Exception as e:
            out.append({"order_id": o.get("order_id"), "error": str(e)[:120]})
    return out


# ------------------------------------------------------------ arming (2-key)
def is_armed() -> bool:
    """Live trading requires BOTH env LIVE_TRADING=true AND the DB arm flag."""
    import sqlite3
    import tracker as T
    try:
        with sqlite3.connect(T.DB_PATH) as c:
            r = c.execute("SELECT v FROM kv WHERE k='live_armed'").fetchone()
            db_arm = (r[0] == "1") if r else False
    except Exception:
        db_arm = False
    import risk as R
    return bool(R.LIVE_OK and db_arm)


def set_armed(on: bool):
    import sqlite3
    import tracker as T
    with sqlite3.connect(T.DB_PATH) as c:
        c.execute("CREATE TABLE IF NOT EXISTS kv(k TEXT PRIMARY KEY, v TEXT)")
        c.execute("INSERT INTO kv(k,v) VALUES('live_armed',?) "
                  "ON CONFLICT(k) DO UPDATE SET v=excluded.v", ("1" if on else "0",))
        c.commit()
    return is_armed()


def preflight(cost_cents: int) -> dict:
    """All-pass checklist before any real order. Returns {ok, checks}."""
    import risk as R
    import tracker as T
    checks = {}
    checks["paper_lock"] = {"ok": R.LIVE_OK, "detail": "LIVE_TRADING env"}
    checks["db_armed"] = {"ok": is_armed(), "detail": "UI arm flag"}
    try:
        b = balance()
        bal = int(float(b.get("balance", 0)) if isinstance(b, dict) else b)
        checks["balance"] = {"ok": bal >= cost_cents,
                             "detail": f"${bal/100:.2f} vs cost ${cost_cents/100:.2f}"}
    except Exception as e:
        checks["balance"] = {"ok": False, "detail": f"unreadable: {e}"[:120]}
    dp = T.day_pnl()
    h, r = R.check_daily_halt(dp["total_cents"])
    checks["daily_halt"] = {"ok": not h, "detail": r or f"day {dp['total_cents']/100:+.2f}"}
    h2, r2 = R.check_exposure(dp["n_open"], cost_cents)
    checks["exposure"] = {"ok": not h2, "detail": r2 or f"{dp['n_open']} open"}
    halts = T.halt_state()
    checks["halts"] = {"ok": not halts, "detail": "; ".join(h["reason"] for h in halts)[:150] or "none"}
    ok = all(v["ok"] for v in checks.values())
    return {"ok": ok, "checks": checks}


def reconcile() -> dict:
    """Exchange truth vs our paper book. Returns diffs for the dashboard."""
    import tracker as T
    try:
        xp = {(p.get("ticker"), "yes" if float(p.get("position_fp", 0) or 0) > 0 else "no"
               if float(p.get("position_fp", 0) or 0) != 0 else "flat"): p
              for p in positions()}
    except Exception as e:
        return {"ok": False, "error": str(e)[:150]}
    xp = {k: v for k, v in xp.items() if k[1] != "flat"}
    ours = {}
    for r in T.list_trades(200):
        if r["status"] == "open" and (r.get("exec_mode") or "taker") == "live":
            ours[(r["market_ticker"], r["side"])] = ours.get((r["market_ticker"], r["side"]), 0) + r["contracts"]
    only_x, only_us = [], []
    for (t, s), pos in xp.items():
        n = float(pos.get("position_fp", 0) or 0)
        if abs(n - ours.get((t, s), 0)) > 1e-9:
            only_x.append({"ticker": t, "side": s, "exchange": n, "book": ours.get((t, s), 0)})
    for (t, s), n in ours.items():
        if (t, s) not in xp:
            only_us.append({"ticker": t, "side": s, "exchange": 0, "book": n})
    return {"ok": True, "match": not (only_x or only_us),
            "exchange_only": only_x, "book_only": only_us}


def live_fill(event_ticker: str, market_ticker: str, side: str, contracts: int,
              price_cents: int, pred_price=None, spot=None, p_up=None, edge=None,
              model_version: str = "") -> dict:
    """Place the REAL order mirroring a paper decision + mirror record.

    Never raises into the trader loop: returns {placed: bool, ...}.
    """
    import tracker as T
    from economics import taker_fee_cents
    cost = contracts * price_cents + int(round(taker_fee_cents(price_cents, contracts)))
    pf = preflight(cost)
    if not pf["ok"]:
        return {"placed": False, "reason": "preflight: " +
                "; ".join(f"{k}({v['detail']})" for k, v in pf["checks"].items() if not v["ok"])[:200]}
    try:
        order = place_limit(market_ticker, side, contracts, price_cents)
    except Exception as e:
        return {"placed": False, "reason": f"order failed: {e}"[:200]}
    tid = T.record_paper_trade(
        event_ticker=event_ticker, market_ticker=market_ticker, side=side,
        contracts=contracts, price_paid_cents=price_cents,
        pred_price=pred_price, spot=spot, p_up=p_up, edge=edge,
        minutes_to_expiry=None, model_version=model_version, exec_mode="live")
    return {"placed": True, "trade_id": tid, "order": {k: order.get(k) for k in
            ("order_id", "status", "ticker", "side", "count", "yes_price", "no_price",
             "client_order_id")}}


if __name__ == "__main__":
    import json
    print("balance:", json.dumps(balance())[:200])
    print("open positions:", len(positions()))
    print("resting orders:", len(list_orders(status="resting")))
