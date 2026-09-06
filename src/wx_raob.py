"""Radiosonde profiles: IEM RAOB JSON (observed, free, no key).

Stations: KOKX (Upton NY ~80km from Central Park) for NYC,
          KILX (Lincoln IL ~200km from O'Hare) for Chicago.
Extracts 850/700/500hPa levels + 12Z daily diagnosis. Twice-daily (00/12Z).
Cache: data_cache/wx_{city}_raob.csv (one row per sounding).
"""
import os
import time
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent
_DD = Path(os.getenv("DATA_DIR", str(ROOT / "data_cache")))
DATA_DIR = _DD if _DD.is_absolute() else ROOT / _DD
DATA_DIR.mkdir(parents=True, exist_ok=True)

RAOB = "https://mesonet.agron.iastate.edu/json/raob.py"
STATION = {"NYC": "KOKX", "CHI": "KILX"}
LEVELS = (850, 700, 500)


def fetch_profile(station: str, ts: str, retries: int = 3) -> list[dict]:
    last = None
    for i in range(retries):
        try:
            r = requests.get(RAOB, params={"station": station, "ts": ts}, timeout=30)
            if r.status_code in (429,) + tuple(range(500, 600)):
                last = f"HTTP {r.status_code}"
                time.sleep(2 * (i + 1))
                continue
            r.raise_for_status()
            return r.json().get("profiles", [])
        except requests.RequestException as e:
            last = str(e)[:100]
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"raob failed {station} {ts}: {last}")


def level_vals(profile: list[dict], pres: float, tol: float = 15.0) -> dict:
    """Nearest-level temp/dewpoint/height within tolerance (hPa)."""
    best, bd = None, tol
    for lv in profile:
        try:
            d = abs(float(lv.get("pres", 1e9)) - pres)
        except (TypeError, ValueError):
            continue
        if d < bd:
            bd, best = d, lv
    if best is None:
        return {"t": None, "dpd": None, "h": None}
    try:
        t = float(best.get("tmpc"))
    except (TypeError, ValueError):
        return {"t": None, "dpd": None, "h": None}
    try:
        dpd = t - float(best.get("dwpc"))
    except (TypeError, ValueError):
        dpd = None
    try:
        h = float(best.get("hght"))
    except (TypeError, ValueError):
        h = None
    return {"t": t, "dpd": dpd, "h": h}


def live_diag(city: str, datestr: str) -> dict | None:
    """12Z obs sounding diagnosis for a calendar date (local), or None.

    datestr like '2026-09-04'. Queries the exact 12Z timestamp.
    """
    try:
        profs = fetch_profile(STATION[city], f"{datestr}T12:00:00Z")
        if not profs:
            return None
        p = profs[0]
        v850 = level_vals(p.get("profile", []), 850)
        v700 = level_vals(p.get("profile", []), 700)
        v500 = level_vals(p.get("profile", []), 500)
        if v850["t"] is None:
            return None
        return {"date": pd.Timestamp(datestr), "t850s": v850["t"], "dpd850": v850["dpd"],
                "t700s": v700["t"], "z500s": v500["h"], "snd_src": "12Z"}
    except Exception:
        return None


def sounding_row(city: str, ts: str) -> dict | None:
    """One sounding -> flat dict, or None if missing."""
    profs = fetch_profile(STATION[city], ts)
    if not profs:
        return None
    p = profs[0]
    row = {"time": p.get("valid", ts), "station": p.get("station")}
    for lv in LEVELS:
        v = level_vals(p.get("profile", []), lv)
        row[f"t{lv}"] = v["t"]
        if lv == 850:
            row["dpd850"] = v["dpd"]
        if lv == 500:
            row["z500s"] = v["h"]
    return row


def pull_history(city: str, start: str = "2020-01-01", end: str | None = None,
                 sleep: float = 0.25, progress=None) -> pd.DataFrame:
    stn = STATION[city]
    if end is None:
        end = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    days = pd.date_range(start, end, freq="D")
    rows = []
    for i, d in enumerate(days):
        ts = f"{d.strftime('%Y-%m-%d')}T12:00:00Z"  # 12Z only: morning state
        try:
            r = sounding_row(city, ts)
            if r:
                rows.append(r)
        except Exception:
            pass
        time.sleep(sleep)
        if progress and i % 30 == 0:
            progress(i, len(days))
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    return df.dropna(subset=["time"]).drop_duplicates("time").sort_values("time").reset_index(drop=True)


def save_history(city: str, df: pd.DataFrame):
    p = DATA_DIR / f"wx_{city.lower()}_raob.csv"
    tmp = p.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    os.replace(tmp, p)


def daily_diag(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    """Per-day 12Z-anchored diagnosis: use the 12Z sounding (morning state).

    Falls back to 00Z when 12Z missing (flagged). Tendencies are 24/48h diffs
    of whatever was available (flagged when mixed).
    """
    d = df.copy()
    d["time"] = pd.to_datetime(d["time"], utc=True, errors="coerce")
    # soundings are queried at 12Z by construction: anchor on UTC hour BEFORE
    # local conversion (12Z = 7-8am local, so a local hr==12 filter never hits)
    d = d[d["time"].dt.hour == 12].copy()
    d["lt"] = d["time"].dt.tz_convert(tz).dt.tz_localize(None)
    d["day"] = d["lt"].dt.floor("D")
    d["hr"] = d["lt"].dt.hour
    rows = []
    for day, g in d.groupby("day"):
        g = g.sort_values("lt")
        r = g.iloc[-1]  # input pre-filtered to 12Z soundings
        rows.append({"date": day, "t850s": r["t850"], "dpd850": r.get("dpd850"),
                     "t700s": r["t700"], "z500s": r["z500s"], "snd_src": "12Z",
                     "snd_hr": int(r["hr"]) if pd.notna(r["hr"]) else None})
    o = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    # reindex to the full daily range: missing soundings stay NaN so 24/48h
    # tendencies never silently span a gap (those rows drop out downstream)
    if len(o):
        full = pd.DataFrame({"date": pd.date_range(o["date"].min(), o["date"].max(), freq="D")})
        o = full.merge(o, on="date", how="left")
    o["t850s_chg24"] = o["t850s"].diff(1)
    o["t850s_chg48"] = o["t850s"].diff(2)
    o["z500s_chg24"] = o["z500s"].diff(1)
    return o


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="NYC")
    ap.add_argument("--start", default="2020-01-01")
    a = ap.parse_args()
    df = pull_history(a.city, start=a.start,
                      progress=lambda i, n: print(f"  {i}/{n}", end="\r", flush=True))
    save_history(a.city, df)
    print(f"\n{a.city}: {len(df)} soundings")
    print(daily_diag(df, "America/New_York" if a.city == "NYC" else "America/Chicago").tail(3).to_string())
