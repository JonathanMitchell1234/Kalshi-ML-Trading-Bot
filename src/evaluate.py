"""Evaluation harness: attribution, uncertainty, risk. No vibes.

- Wilson CIs on every hit rate (n=90 backtests have ±10pt bands — say so).
- Per-model-version P&L (versions are stamped on trades).
- Paper equity curve + Sharpe + max drawdown (realized, daily).
- Reliability: stored p_up buckets vs realized win rate (calibration live).
"""
import math
import sqlite3
import pandas as pd
import tracker as T


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if not n:
        return None, None
    p = wins / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round((c - m) / d, 4), round((c + m) / d, 4)


def trades_df() -> pd.DataFrame:
    with T.conn() as c:
        return pd.read_sql("SELECT * FROM paper_trades ORDER BY id", c)


def version_pnl() -> list[dict]:
    df = trades_df()
    out = []
    for ver, g in df.groupby(df["model_version"].fillna("(unstamped)")):
        st = g[g["status"].isin(("won", "lost"))]
        w = int((st["status"] == "won").sum())
        lo, hi = wilson(w, len(st))
        out.append({"version": ver, "trades": len(g), "settled": len(st),
                    "wins": w, "win_rate": round(w / len(st), 4) if len(st) else None,
                    "ci95": [lo, hi], "pnl_cents": int(st["pnl_cents"].fillna(0).sum()),
                    "open": int((g["status"] == "open").sum())})
    return sorted(out, key=lambda r: r["version"])


def equity() -> dict:
    df = trades_df()
    st = df[df["status"].isin(("won", "lost"))].copy()
    if st.empty:
        return {"points": [], "sharpe": None, "max_dd_cents": 0, "total_cents": 0}
    st["day"] = pd.to_datetime(st["settled_at"]).dt.date.astype(str)
    daily = st.groupby("day")["pnl_cents"].sum().reset_index()
    daily["equity"] = daily["pnl_cents"].cumsum()
    pts = [{"day": r["day"], "pnl": int(r["pnl_cents"]), "equity": int(r["equity"])}
           for _, r in daily.iterrows()]
    rets = daily["pnl_cents"] / 100000.0  # vs $1000 paper bankroll
    sharpe = round(float(rets.mean() / rets.std() * (252 ** 0.5)), 2) if len(rets) > 2 and rets.std() else None
    peak = daily["equity"].cummax()
    dd = int((daily["equity"] - peak).min())
    open_risk = int(df[df["status"] == "open"]["cost_cents"].fillna(0).sum())
    return {"points": pts, "sharpe": sharpe, "max_dd_cents": dd,
            "total_cents": int(daily["pnl_cents"].sum()), "open_risk_cents": open_risk}


def reliability(n_bins: int = 5) -> list[dict]:
    df = trades_df()
    st = df[df["status"].isin(("won", "lost")) & df["p_up"].notna()].copy()
    if st.empty:
        return []
    st["win"] = (st["status"] == "won").astype(int)
    st["bin"] = pd.qcut(st["p_up"].clip(0.01, 0.99), q=min(n_bins, len(st)), duplicates="drop")
    out = []
    for b, g in st.groupby(st["bin"], observed=True):
        lo, hi = wilson(int(g["win"].sum()), len(g))
        out.append({"range": f"{b.left:.2f}-{b.right:.2f}", "n": len(g),
                    "mean_p": round(float(g["p_up"].mean()), 3),
                    "hit_rate": round(float(g["win"].mean()), 3), "ci95": [lo, hi]})
    return out


def summary() -> dict:
    return {"versions": version_pnl(), "equity": equity(), "reliability": reliability()}
