"""SQLite store for paper trades (live) and backtest picks (settled events)."""
import json
import os
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent


def _resolve(p: Path) -> Path:
    return p if p.is_absolute() else ROOT / p


DB_PATH = _resolve(Path(os.getenv("DATA_DIR", "data_cache"))) / "paper_trades.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_trades(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  event_ticker TEXT NOT NULL,
  market_ticker TEXT NOT NULL,
  side TEXT NOT NULL,               -- 'yes' only for now
  contracts INTEGER NOT NULL,
  price_paid_cents INTEGER NOT NULL,-- ask paid per contract
  cost_cents INTEGER NOT NULL,      -- contracts * price
  pred_price REAL, spot REAL, p_up REAL, edge REAL, minutes_to_expiry REAL,
  model_version TEXT DEFAULT '',
  status TEXT NOT NULL DEFAULT 'open',  -- open | won | lost | void
  result TEXT, pnl_cents INTEGER, settled_at TEXT, expiry_value TEXT
);
CREATE TABLE IF NOT EXISTS backtest_runs(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  started_at TEXT NOT NULL, n_events INTEGER, signal_lead_min INTEGER, note TEXT
);
CREATE TABLE IF NOT EXISTS backtest_picks(
  run_id INTEGER NOT NULL, event_ticker TEXT NOT NULL,
  expiry_ts TEXT, signal_time TEXT, spot REAL, pred_price REAL,
  pred_bracket TEXT, winner_bracket TEXT, hit INTEGER, err15 REAL,
  expiry_spot REAL
);
"""


def conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    return c


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------ paper trades
def record_paper_trade(event_ticker: str, market_ticker: str, side: str, contracts: int,
                       price_paid_cents: int, pred_price: float | None = None,
                       spot: float | None = None, p_up: float | None = None,
                       edge: float | None = None, minutes_to_expiry: float | None = None,
                       model_version: str = "") -> int:
    with conn() as c:
        try:
            c.execute("ALTER TABLE paper_trades ADD COLUMN model_version TEXT DEFAULT ''")
        except Exception:
            pass
        cur = c.execute(
            "INSERT INTO paper_trades(created_at,event_ticker,market_ticker,side,contracts,"
            "price_paid_cents,cost_cents,pred_price,spot,p_up,edge,minutes_to_expiry,model_version)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (now_iso(), event_ticker, market_ticker, side, contracts, price_paid_cents,
             contracts * price_paid_cents, pred_price, spot, p_up, edge, minutes_to_expiry,
             model_version))
        return cur.lastrowid


def stamp(key: str) -> str:
    """Record a new trained version id, return it."""
    from signals import MODEL_DIR
    from datetime import datetime, timezone
    p = MODEL_DIR / "versions.json"
    v = json.loads(p.read_text()) if p.exists() else {}
    v[key] = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    p.write_text(json.dumps(v, indent=1))
    return v[key]


def model_version(key: str) -> str:
    """Stamped version id for a model family (e.g. 'dir_BTC', 'wx_NYC')."""
    try:
        from signals import MODEL_DIR
        return json.loads((MODEL_DIR / "versions.json").read_text()).get(key, "")
    except Exception:
        return 


def resolve_open_trades() -> list[dict]:
    """Poll Kalshi for open trades; settle won/lost/void (P&L net of taker fee)."""
    from kalshi_client import get_market
    from economics import taker_fee_cents
    updated = []
    with conn() as c:
        rows = c.execute("SELECT * FROM paper_trades WHERE status='open'").fetchall()
    for r in rows:
        try:
            m = get_market(r["market_ticker"])
        except Exception:
            continue
        status, result = (m.get("status") or ""), (m.get("result") or "")
        if status not in ("finalized", "settled") or result not in ("yes", "no"):
            continue
        side, px, n = r["side"], r["price_paid_cents"], r["contracts"]
        fee = taker_fee_cents(px, n)
        if result == "void":
            new_status, pnl = "void", 0
        elif (result == "yes" and side == "yes") or (result == "no" and side == "no"):
            new_status, pnl = "won", n * (100 - px) - fee
        else:
            new_status, pnl = "lost", -(n * px) - fee
        pnl = int(round(pnl))
        with conn() as c:
            c.execute("UPDATE paper_trades SET status=?, result=?, pnl_cents=?, settled_at=?,"
                      "expiry_value=? WHERE id=?",
                      (new_status, result, pnl, now_iso(), str(m.get("expiration_value")), r["id"]))
        updated.append({"id": r["id"], "market": r["market_ticker"], "status": new_status, "pnl_cents": pnl})
    return updated


def paper_stats() -> dict:
    resolve_open_trades()
    with conn() as c:
        rows = c.execute("SELECT status, COUNT(*) n, COALESCE(SUM(pnl_cents),0) p FROM paper_trades GROUP BY status").fetchall()
    s = {r["status"]: {"n": r["n"], "pnl_cents": r["p"]} for r in rows}
    settled = s.get("won", {"n": 0})["n"] + s.get("lost", {"n": 0})["n"]
    wins = s.get("won", {"n": 0})["n"]
    pnl = sum(v["pnl_cents"] for v in s.values())
    return {
        "total": sum(v["n"] for v in s.values()),
        "open": s.get("open", {"n": 0})["n"],
        "settled": settled,
        "wins": wins,
        "losses": s.get("lost", {"n": 0})["n"],
        "win_rate": round(wins / settled, 4) if settled else None,
        "pnl_cents": pnl,
        "pnl_dollars": round(pnl / 100.0, 2),
    }


def list_trades(limit: int = 100) -> list[dict]:
    with conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT * FROM paper_trades ORDER BY id DESC LIMIT ?", (limit,))]


# ------------------------------------------------------------ backtests
def save_backtest_run(n_events: int, lead_min: int, picks: list[dict], note: str = "") -> int:
    with conn() as c:
        cur = c.execute("INSERT INTO backtest_runs(started_at,n_events,signal_lead_min,note) VALUES(?,?,?,?)",
                        (now_iso(), n_events, lead_min, note))
        rid = cur.lastrowid
        c.executemany(
            "INSERT INTO backtest_picks(run_id,event_ticker,expiry_ts,signal_time,spot,pred_price,"
            "pred_bracket,winner_bracket,hit,err15,expiry_spot) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            [(rid, p["event_ticker"], p.get("expiry_ts"), p.get("signal_time"), p.get("spot"),
              p.get("pred_price"), p.get("pred_bracket"), p.get("winner_bracket"),
              p.get("hit"), p.get("err15"), p.get("expiry_spot")) for p in picks])
    return rid


def backtest_stats(run_id: int | None = None) -> dict:
    with conn() as c:
        if run_id is None:
            r = c.execute("SELECT id FROM backtest_runs ORDER BY id DESC LIMIT 1").fetchone()
            if not r:
                return {"runs": 0}
            run_id = r["id"]
        picks = [dict(x) for x in c.execute("SELECT * FROM backtest_picks WHERE run_id=?", (run_id,))]
        n_runs = c.execute("SELECT COUNT(*) n FROM backtest_runs").fetchone()["n"]
    scored = [p for p in picks if p["hit"] is not None]
    hits = sum(1 for p in scored if p["hit"])
    errs = [p["err15"] for p in scored if p["err15"] is not None]
    return {
        "runs": n_runs, "run_id": run_id,
        "n_events": len(picks), "n_scored": len(scored),
        "hits": hits,
        "hit_rate": round(hits / len(scored), 4) if scored else None,
        "mean_err15": round(float(sum(errs) / len(errs)), 2) if errs else None,
        "picks": picks[-50:][::-1],
    }
