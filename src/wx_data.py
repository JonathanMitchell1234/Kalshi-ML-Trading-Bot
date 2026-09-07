"""Weather data from Open-Meteo (free, no key): forecast + ensemble + archive.

Cities = Kalshi daily-high markets (settlement: The Weather Company):
  NYC = Central Park  (40.78, -73.97)  -> KXHIGHNY
  CHI = O'Hare        (41.98, -87.90)  -> KXHIGHCHI
Cache: data_cache/wx_{city}_{kind}.csv, kinds: obs (archive daily),
       fcst (GFS daily snapshots), ens (ensemble daily snapshots).
All temps Fahrenheit, dates America/New_York (NYC) / America/Chicago (CHI).
"""
import os
import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent
_DD = Path(os.getenv("DATA_DIR", str(ROOT / "data_cache")))
DATA_DIR = _DD if _DD.is_absolute() else ROOT / _DD
DATA_DIR.mkdir(parents=True, exist_ok=True)

CITIES = {
    "NYC": {"lat": 40.78, "lon": -73.97, "tz": "America/New_York", "series": "KXHIGHNY"},
    "CHI": {"lat": 41.98, "lon": -87.90, "tz": "America/Chicago", "series": "KXHIGHCHI"},
}
FCST = "https://api.open-meteo.com/v1/forecast"
ENS = "https://ensemble-api.open-meteo.com/v1/ensemble"
ARCH = "https://archive-api.open-meteo.com/v1/archive"


def _get(url: str, params: dict, retries: int = 3) -> dict:
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code in (429,) + tuple(range(500, 600)):
                last = f"HTTP {r.status_code}"
                time.sleep(2 * (i + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            last = str(e)[:100]
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"open-meteo failed: {last}")


NWS_UA = os.getenv("NWS_USER_AGENT", "kalshi-weather-bot")
NWS_STATION = {"NYC": "KNYC", "CHI": "KORD"}
IEM_SITE = {"NYC": ("NYC", "NY_ASOS"), "CHI": ("ORD", "IL_ASOS")}


def nws_headers() -> dict:
    return {"User-Agent": NWS_UA, "Accept": "application/geo+json"}


def _atomic_save(df: pd.DataFrame, path: Path):
    tmp = path.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def fetch_nws_recent(city: str) -> pd.DataFrame:
    """Recent station obs from api.weather.gov (serve-time fresh, ~1h lag)."""
    r = requests.get(f"https://api.weather.gov/stations/{NWS_STATION[city]}/observations",
                     headers=nws_headers(), timeout=30)
    r.raise_for_status()
    rows = []
    for f in r.json().get("features", []):
        p = f.get("properties", {})
        t = p.get("temperature", {}).get("value")
        rows.append({"time": pd.to_datetime(p.get("timestamp")),
                     "temp_c": t,
                     "mslp": (p.get("seaLevelPressure", {}) or {}).get("value")})
    df = pd.DataFrame(rows).dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    df["temp_f"] = df["temp_c"] * 9 / 5 + 32
    df["mslp"] = df["mslp"] / 100.0  # Pa -> hPa
    return df


def station_daily(city: str) -> pd.DataFrame:
    """Daily obs from the settlement-site station (IEM ASOS history).

    Columns match the ERA5 obs frame (date,tmax,tmin,precip,cloud,wind) with
    NaN where the station lacks a field; plus src_stn=1. Trainer fills gaps
    from ERA5 where available.
    """
    cp = DATA_DIR / f"wx_{city.lower()}_stn.csv"
    tz = CITIES[city]["tz"]
    h = pd.read_csv(cp, parse_dates=["time"])
    h["time"] = pd.to_datetime(h["time"]).dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT").dt.tz_localize(None)
    h["day"] = h["time"].dt.floor("D")
    g = h.groupby("day").agg(tmax=("tmpf", "max"), tmin=("tmpf", "min"), n=("tmpf", "count"))
    g = g[g["n"] >= 12].drop(columns="n").reset_index().rename(columns={"day": "date"})
    g["precip"] = np.nan
    g["cloud"] = np.nan
    g["wind"] = np.nan
    g["src_stn"] = 1
    # QC: physical range, spike (>40F day jump impossible), persistence (>6 identical)
    n0 = len(g)
    g = g[(g["tmax"] >= -60) & (g["tmax"] <= 130)]
    g = g[g["tmax"].diff().abs().fillna(0) <= 40]
    g = g[~(g["tmax"].rolling(7, min_periods=7).std().fillna(1) == 0)]
    if len(g) < n0:
        print(f"station QC dropped {n0 - len(g)} rows for {city}")
    return g


def iem_recent_daily(city: str, days: int = 14, ttl_h: int = 6) -> pd.DataFrame:
    """Fresh station daily rows (through yesterday) for live features.

    File-cached with TTL (IEM rate-limits aggressively).
    """
    import time as _t
    cp = DATA_DIR / f"wx_{city.lower()}_recent.csv"
    if cp.exists() and (_t.time() - cp.stat().st_mtime) < ttl_h * 3600:
        df = pd.read_csv(cp, parse_dates=["date"])
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        if len(df):
            return df
    site, _net = IEM_SITE[city]
    tz = CITIES[city]["tz"]
    end = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    start = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days + 2)).strftime("%Y-%m-%d")
    q = {"station": site, "data": "tmpf", "year1": 2000, "month1": 1, "day1": 1,
         "year2": 2000, "month2": 1, "day2": 1, "tz": tz, "format": "onlycomma"}
    y1, m1, d1 = start.split("-")
    y2, m2, d2 = end.split("-")
    q.update(year1=int(y1), month1=int(m1), day1=int(d1), year2=int(y2), month2=int(m2), day2=int(d2))
    frames = []
    for var in ("tmpf", "mslp"):
        q["data"] = var
        r = requests.get("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py",
                         params=q, timeout=120)
        r.raise_for_status()
        lines = [ln for ln in r.text.splitlines() if ln and not ln.startswith("station")]
        f = pd.DataFrame([ln.split(",") for ln in lines if ln.count(",") >= 2],
                         columns=["station", "valid", var])
        f["time"] = pd.to_datetime(f["valid"], errors="coerce").dt.floor("h")
        f[var] = pd.to_numeric(f[var], errors="coerce")
        frames.append(f[["time", var]])
    m = frames[0].merge(frames[1], on="time", how="outer").dropna(subset=["time"])
    m["time"] = pd.to_datetime(m["time"]).dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT").dt.tz_localize(None)
    m["day"] = m["time"].dt.floor("D")
    g = m.groupby("day").agg(tmax=("tmpf", "max"), tmin=("tmpf", "min"),
                             mslp_noon=("mslp", lambda s: s[m.loc[s.index, "time"].dt.hour == 12].mean()
                                        if (m.loc[s.index, "time"].dt.hour == 12).any() else np.nan))
    g = g.reset_index().rename(columns={"day": "date"})
    g["precip"] = np.nan
    g["cloud"] = np.nan
    g["wind"] = np.nan
    g["src_stn"] = 1
    g.to_csv(cp, index=False)
    return g


def assemble_obs(city: str) -> pd.DataFrame:
    """Single obs source for train/backtest/live: settlement-site station
    daily rows first (IEM history + fresh recent), ERA5 fills precip/cloud/
    wind gaps. src_stn=1 where tmax came from the station."""
    era = load_obs(city, start="2020-01-01")
    era["date"] = pd.to_datetime(era["date"]).dt.tz_localize(None)
    try:
        stn = station_daily(city)
        try:
            rec = iem_recent_daily(city)
            stn = pd.concat([stn, rec], ignore_index=True).drop_duplicates("date").sort_values("date")
        except Exception as e:
            print(f"recent station fetch failed ({e})")
        stn["date"] = pd.to_datetime(stn["date"]).dt.tz_localize(None)
        obs = stn.merge(era[["date", "precip", "cloud", "wind", "hum", "gust"]], on="date", how="left",
                        suffixes=("", "_era"))
        for col in ("precip", "cloud", "wind", "hum", "gust"):
            ecol = f"{col}_era"
            if col in obs.columns and ecol in obs.columns:
                obs[col] = obs[col].fillna(obs[ecol])
            elif ecol in obs.columns:
                obs[col] = obs[ecol]
        obs = obs.drop(columns=[c for c in obs.columns if c.endswith("_era")])
    except Exception as e:
        print(f"station obs unavailable ({e}), ERA5 only")
        obs = era.copy()
        obs["src_stn"] = 0
    return obs.sort_values("date").reset_index(drop=True)


def fetch_ndfd_maxt(city: str, days: int = 4) -> dict:
    """NWS official point forecast daily highs (DWML time-series).

    Human-augmented NWS forecast = the strongest public reference for the
    NWP-disagreement rail and future blending. Returns {date: fahrenheit}.
    """
    import xml.etree.ElementTree as ET
    from datetime import datetime, timedelta, timezone
    c = CITIES[city]
    now = datetime.now(timezone.utc)
    r = requests.get("https://graphical.weather.gov/xml/sample_products/browser_interface/ndfdXMLclient.php",
                     params={"lat": c["lat"], "lon": c["lon"], "product": "time-series",
                             "maxt": "maxt",
                             "begin": (now - timedelta(days=1)).strftime("%Y-%m-%dT00:00:00"),
                             "end": (now + timedelta(days=days)).strftime("%Y-%m-%dT00:00:00")},
                     timeout=40)
    r.raise_for_status()
    root = ET.fromstring(r.content)
    layouts = {}
    for layout in root.iter("time-layout"):
        key = layout.findtext("layout-key")
        times = [pd.Timestamp(t.text).tz_localize(None) for t in layout.findall("start-valid-time")]
        layouts[key] = times
    out = {}
    for temp in root.iter("temperature"):
        if temp.attrib.get("type") != "maximum":
            continue
        times = layouts.get(temp.attrib.get("time-layout", ""), [])
        for t, v in zip(times, temp.findall("value")):
            try:
                out.setdefault(t.strftime("%Y-%m-%d"), float(v.text))
            except (TypeError, ValueError):
                pass
    return out


def fetch_iem_history(city: str, start: str = "2020-01-01", end: str | None = None) -> pd.DataFrame:
    """Hourly station history (tmpf + mslp) from Iowa State IEM ASOS archive."""
    site, _net = IEM_SITE[city]


def fetch_obs(city: str, start: str, end: str) -> pd.DataFrame:
    """Archive daily obs: tmax, tmin, precip, cloud, wind. ERA5-based."""
    c = CITIES[city]
    j = _get(ARCH, {"latitude": c["lat"], "longitude": c["lon"],
                    "start_date": start, "end_date": end,
                    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,"
                             "cloud_cover_mean,wind_speed_10m_max,relative_humidity_2m_mean,"
                             "wind_gusts_10m_max",
                    "timezone": c["tz"], "temperature_unit": "fahrenheit",
                    "wind_speed_unit": "mph", "precipitation_unit": "inch"})
    d = j.get("daily", {})
    return pd.DataFrame({
        "date": pd.to_datetime(d.get("time", [])),
        "tmax": pd.to_numeric(pd.Series(d.get("temperature_2m_max", [])), errors="coerce"),
        "tmin": pd.to_numeric(pd.Series(d.get("temperature_2m_min", [])), errors="coerce"),
        "precip": pd.to_numeric(pd.Series(d.get("precipitation_sum", [])), errors="coerce"),
        "cloud": pd.to_numeric(pd.Series(d.get("cloud_cover_mean", [])), errors="coerce"),
        "wind": pd.to_numeric(pd.Series(d.get("wind_speed_10m_max", [])), errors="coerce"),
        "hum": pd.to_numeric(pd.Series(d.get("relative_humidity_2m_mean", [])), errors="coerce"),
        "gust": pd.to_numeric(pd.Series(d.get("wind_gusts_10m_max", [])), errors="coerce"),
    })


def fetch_forecast_snapshot(city: str) -> dict:
    """Current GFS snapshot: daily max today + next 2 days, hourly today, model time."""
    c = CITIES[city]
    j = _get(FCST, {"latitude": c["lat"], "longitude": c["lon"],
                    "hourly": "temperature_2m,precipitation_probability,cloud_cover,pressure_msl,relative_humidity_2m,wind_gusts_10m,shortwave_radiation,wind_speed_10m,wind_direction_10m,dew_point_2m,"
                              "temperature_850hPa,temperature_700hPa,geopotential_height_500hPa,"
                              "dew_point_2m",
                    "daily": "temperature_2m_max,precipitation_sum",
                    "timezone": c["tz"], "temperature_unit": "fahrenheit",
                    "wind_speed_unit": "mph",
                    "past_days": 3, "forecast_days": 3})
    return {"fetched_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
            "utc_offset": j.get("utc_offset_seconds", 0),
            "hourly": j.get("hourly", {}), "daily": j.get("daily", {})}


def fetch_ensemble_snapshot(city: str) -> dict:
    """31-member GFS ensemble hourly temps, next 48h."""
    c = CITIES[city]
    j = _get(ENS, {"latitude": c["lat"], "longitude": c["lon"],
                   "hourly": "temperature_2m", "timezone": c["tz"],
                   "forecast_days": 2, "temperature_unit": "fahrenheit",
                   "models": "gfs_seamless"})
    return {"fetched_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
            "hourly": j.get("hourly", {})}


def obs_path(city: str) -> Path:
    return DATA_DIR / f"wx_{city.lower()}_obs.csv"


def load_obs(city: str, start: str = "2020-01-01", end: str | None = None) -> pd.DataFrame:
    """Full obs cache on disk; fetches missing tail. Returns requested range."""
    cp = obs_path(city)
    if end is None:
        end = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=6)).strftime("%Y-%m-%d")
    have = None
    if cp.exists():
        have = pd.read_csv(cp, parse_dates=["date"])
        have["date"] = pd.to_datetime(have["date"]).dt.tz_localize(None)
        mx = have["date"].max()
        need_from = (mx + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if need_from <= end:
            try:
                tail = fetch_obs(city, need_from, end)
                have = pd.concat([have, tail], ignore_index=True)
            except Exception as e:
                print(f"obs tail failed ({e}), using cache")
        have = have.drop_duplicates("date").sort_values("date").reset_index(drop=True)
        _atomic_save(have, cp)
    else:
        print(f"Fetching {city} obs {start}..{end} (one paginated pull)...")
        frames, cur = [], pd.Timestamp(start)
        stop = pd.Timestamp(end)
        while cur <= stop:
            nxt = min(cur + pd.Timedelta(days=730), stop)
            frames.append(fetch_obs(city, cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")))
            cur = nxt + pd.Timedelta(days=1)
            time.sleep(0.5)
        have = pd.concat(frames, ignore_index=True).drop_duplicates("date").sort_values("date").reset_index(drop=True)
        _atomic_save(have, cp)
    out = have.copy()
    out["date"] = pd.to_datetime(out["date"])
    return out[(out["date"] >= pd.Timestamp(start)) & (out["date"] <= pd.Timestamp(end))].reset_index(drop=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="NYC")
    ap.add_argument("--start", default="2020-01-01")
    a = ap.parse_args()
    df = load_obs(a.city, start=a.start)
    print(len(df), "rows", df["date"].min(), "->", df["date"].max())
    print(df.tail(2).to_string())
    print("snapshot keys:", list(fetch_forecast_snapshot(a.city).keys()))
