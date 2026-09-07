"""Daily-state features for T_max prediction. Strictly causal (prior days only).

Same builder for training (ERA5 archive obs) and live (archive + fresh
forecast-API daily rows). No same-day info -> every GFS cycle sees the same
prediction for a target date; cycles differ by market prices.
"""
import numpy as np
import pandas as pd

FEATS: list[str] = []  # filled by build_frame column order


def _clim_expanding(doy: pd.Series, y: pd.Series, min_n: int = 2) -> tuple[pd.Series, pd.Series]:
    """Causal day-of-year mean/std via per-doy accumulators (past only)."""
    mean = pd.Series(index=y.index, dtype=float)
    std = pd.Series(index=y.index, dtype=float)
    n: dict[int, int] = {}
    s: dict[int, float] = {}
    s2: dict[int, float] = {}
    for i, (d, v) in enumerate(zip(doy.values, y.values)):
        d = int(d)
        if n.get(d, 0) >= min_n:
            m = s[d] / n[d]
            var = max(s2[d] / n[d] - m * m, 0.0)
            mean.iloc[i] = m
            std.iloc[i] = float(np.sqrt(var)) if var > 0 else np.nan
        else:
            mean.iloc[i] = np.nan
            std.iloc[i] = np.nan
        if not np.isnan(v):
            n[d] = n.get(d, 0) + 1
            s[d] = s.get(d, 0.0) + v
            s2[d] = s2.get(d, 0.0) + v * v
    glob_m, glob_s = float(np.nanmean(y)), float(np.nanstd(y))
    return mean.fillna(glob_m), std.fillna(glob_s if glob_s > 0 else 3.0)


def daily_upper(hourly: pd.DataFrame) -> pd.DataFrame:
    """Local-noon snapshot diagnosis: 850hPa temp + 500hPa height at 12:00.

    Fixed 12:00-local values (not daily max) so partial days never skew:
    identical construction for ERA5 history (training) and GFS past-days
    hours (serve). Tendencies are the front detectors NWP models see.
    """
    h = hourly.copy()
    h["time"] = pd.to_datetime(h["time"]).dt.tz_localize(None)
    d = h[h["time"].dt.hour == 12][["time", "mslp"]].copy()
    d["date"] = d["time"].dt.floor("D")
    d = d.rename(columns={"mslp": "mslp_noon"})
    d["mslp_chg24"] = d["mslp_noon"].diff(1)
    d["mslp_chg48"] = d["mslp_noon"].diff(2)
    return d[["date", "mslp_noon", "mslp_chg24", "mslp_chg48"]]


def daily_sw(hourly: pd.DataFrame) -> pd.DataFrame:
    """Hourly shortwave -> daily max/mean (local days). ERA5 history or GFS hours."""
    h = hourly.copy()
    h["time"] = pd.to_datetime(h["time"]).dt.tz_localize(None)
    h["day"] = h["time"].dt.floor("D")
    g = h.groupby("day")["sw"].agg(["max", "mean"]).reset_index().rename(
        columns={"day": "date", "max": "sw_max", "mean": "sw_mean"})
    return g


def daily_wind(hourly: pd.DataFrame) -> pd.DataFrame:
    """Hourly speed/dir -> local-noon u/v + onshore (easterly) component.

    u = -spd*sin(dir), v = -spd*cos(dir) (met convention, FROM which wind blows).
    onshore = max(0, -u): easterly lake/sea breeze that collapses afternoon highs.
    """
    import math
    h = hourly.copy()
    h["time"] = pd.to_datetime(h["time"]).dt.tz_localize(None)
    h["day"] = h["time"].dt.floor("D")
    rows = []
    for day, g in h.groupby("day"):
        noon = g[(g["time"].dt.hour >= 11) & (g["time"].dt.hour <= 13)]
        if noon.empty:
            continue
        r = noon.iloc[len(noon) // 2]
        try:
            u = -float(r["wspd"]) * math.sin(math.radians(float(r["wdir"])))
            v = -float(r["wspd"]) * math.cos(math.radians(float(r["wdir"])))
        except (TypeError, ValueError):
            continue
        rows.append({"date": day, "wind_u": u, "wind_v": v, "onshore": max(0.0, -u)})
    return pd.DataFrame(rows)


def station_dpd_daily(city: str) -> pd.DataFrame:
    """2m dewpoint depression from settlement-site obs (station tmpf + dewpoint).

    Same site, same hours: no basis games. Returns date, dpd_min, dpd_mean.
    """
    import pandas as pd
    from wx_data import DATA_DIR
    stn = pd.read_csv(DATA_DIR / f"wx_{city.lower()}_stn.csv", parse_dates=["time"])
    dpd = pd.read_csv(DATA_DIR / f"wx_{city.lower()}_dpd.csv", parse_dates=["time"])
    stn["time"] = pd.to_datetime(stn["time"]).dt.floor("h")
    dpd["time"] = pd.to_datetime(dpd["time"]).dt.floor("h")
    m = stn[["time", "tmpf"]].merge(dpd[["time", "dw"]], on="time", how="inner")
    m["t2m"] = (m["tmpf"] - 32) * 5 / 9
    m["day"] = m["time"].dt.floor("D")
    m["dpd"] = m["t2m"] - m["dw"]
    g = m.groupby("day")["dpd"].agg(["min", "mean"]).reset_index().rename(
        columns={"day": "date", "min": "dpd_min", "mean": "dpd_mean"})
    return g


def daily_dpd(hourly: pd.DataFrame) -> pd.DataFrame:
    """Hourly 2m dewpoint -> daily min/mean depression vs 2m temp.

    Requires columns time, t2m, dw (same units). Dry afternoons (large
    depression) precede strong heating; saturated air caps it.
    """
    h = hourly.copy()
    h["time"] = pd.to_datetime(h["time"]).dt.tz_localize(None)
    h["dpd"] = h["t2m"] - h["dw"]
    h["day"] = h["time"].dt.floor("D")
    g = h.groupby("day")["dpd"].agg(["min", "mean"]).reset_index().rename(
        columns={"day": "date", "min": "dpd_min", "mean": "dpd_mean"})
    return g


def solar_H0(doy: pd.Series, lat_deg: float) -> pd.Series:
    """Daily extraterrestrial insolation (MJ/m2/day): deterministic ceiling."""
    import math
    lat, out = math.radians(lat_deg), []
    for d in doy.values:
        dec = math.radians(23.45 * math.sin(math.radians(360 / 365 * (d - 81))))
        c = -math.tan(lat) * math.tan(dec)
        hs = math.acos(min(max(c, -1.0), 1.0))
        out.append(24 / math.pi * 0.0820 * (1 + 0.033 * math.cos(math.radians(360 * d / 365))) *
                   (math.cos(lat) * math.cos(dec) * math.sin(hs) + hs * math.sin(lat) * math.sin(dec)))
    return pd.Series(out, index=doy.index)


def build_frame(obs: pd.DataFrame, upper_daily: pd.DataFrame | None = None,
                keep_all: bool = False, snd_daily: pd.DataFrame | None = None,
                lat_deg: float | None = None, sw_daily: pd.DataFrame | None = None,
                wind_daily: pd.DataFrame | None = None,
                dpd_daily: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[str]]:
    _ = sw_daily  # intentionally unused (see NOTE above)
    df = obs.copy().sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    df["doy"] = df["date"].dt.dayofyear
    df["year"] = df["date"].dt.year
    df["sin1"] = np.sin(2 * np.pi * df["doy"] / 365.25)
    df["cos1"] = np.cos(2 * np.pi * df["doy"] / 365.25)
    df["sin2"] = np.sin(4 * np.pi * df["doy"] / 365.25)
    df["cos2"] = np.cos(4 * np.pi * df["doy"] / 365.25)
    c = df["tmax"]
    df["lag1"] = c.shift(1)
    df["lag2"] = c.shift(2)
    df["lag3"] = c.shift(3)
    df["tmin1"] = df["tmin"].shift(1)
    df["range1"] = df["lag1"] - df["tmin1"]
    df["mean3"] = c.shift(1).rolling(3).mean()
    df["mean7"] = c.shift(1).rolling(7).mean()
    df["slope3"] = c.shift(1).rolling(3).apply(lambda w: np.polyfit([0, 1, 2], w, 1)[0], raw=False)
    cm, cs = _clim_expanding(df["doy"], c)
    df["clim"] = cm.shift(1)
    df["climstd"] = cs.shift(1)
    df["anom1"] = df["lag1"] - df["clim"]
    df["anom3"] = df["mean3"] - df["clim"]
    df["precip1"] = df["precip"].shift(1).fillna(0)
    df["cloud1"] = df["cloud"].shift(1)
    df["wind1"] = df["wind"].shift(1)
    if "hum" in df.columns:
        df["hum1"] = df["hum"].shift(1)
    if "gust" in df.columns:
        df["gust1"] = df["gust"].shift(1)
    feats = (["doy", "year", "sin1", "cos1", "sin2", "cos2", "lag1", "lag2", "lag3",
              "tmin1", "range1", "mean3", "mean7", "slope3", "clim", "climstd",
              "anom1", "anom3", "precip1", "cloud1", "wind1"] +
             [c for c in ("hum1", "gust1") if c in df.columns])
    if "src_stn" in df.columns:
        df["src_stn"] = df["src_stn"].fillna(0)
        feats = feats + ["src_stn"]
    # diurnal-range tendency + precip deficits (Bowen drought proxy)
    df["dtr_chg"] = df["range1"] - (df["lag1"].shift(1) - df["tmin1"].shift(1))
    df["precip14"] = df["precip"].shift(1).rolling(14, min_periods=7).sum()
    df["precip30"] = df["precip"].shift(1).rolling(30, min_periods=15).sum()
    feats = feats + ["dtr_chg", "precip14", "precip30"]
    if lat_deg is not None:
        df["H0"] = solar_H0(df["doy"], lat_deg).values
        feats = feats + ["H0"]
    # NOTE: shortwave history evaluated 2026-09: no gain over H0 harmonics;
    # sw_daily accepted but ignored to keep the production surface small.
    if upper_daily is not None:
        u = upper_daily.copy()
        u["date"] = pd.to_datetime(u["date"]).dt.tz_localize(None)
        # decision-time causal: target D uses D-1 12Z diagnosis (observed by
        # every cycle; 00Z cycle sees today's, never tomorrow's)
        for col in ["mslp_noon", "mslp_chg24", "mslp_chg48"]:
            df[col] = df["date"].map(dict(zip(u["date"], u[col]))).shift(1)
        feats += ["mslp_noon", "mslp_chg24", "mslp_chg48"]
    if wind_daily is not None:
        u = wind_daily.copy()
        u["date"] = pd.to_datetime(u["date"]).dt.tz_localize(None)
        for col in ["wind_u", "wind_v", "onshore"]:
            df[col] = df["date"].map(dict(zip(u["date"], u[col]))).shift(1)
        feats += ["wind_u", "wind_v", "onshore"]
    if dpd_daily is not None:
        u = dpd_daily.copy()
        u["date"] = pd.to_datetime(u["date"]).dt.tz_localize(None)
        for col in ["dpd_min", "dpd_mean"]:
            df[col] = df["date"].map(dict(zip(u["date"], u[col]))).shift(1)
        feats += ["dpd_min", "dpd_mean"]
    if snd_daily is not None:
        u = snd_daily.copy()
        u["date"] = pd.to_datetime(u["date"]).dt.tz_localize(None)
        for col in ["t850s", "t850s_chg24", "t850s_chg48", "dpd850", "t700s",
                    "z500s", "z500s_chg24"]:
            if col in u.columns:
                df[col] = df["date"].map(dict(zip(u["date"], u[col]))).shift(1)
        df["snd_src_obs"] = df["date"].map(
            dict(zip(u["date"], (u.get("snd_src", "12Z") == "12Z").astype(int)))).shift(1).fillna(0)
        feats += ["t850s", "t850s_chg24", "t850s_chg48", "dpd850", "t700s",
                  "z500s", "z500s_chg24", "snd_src_obs"]
    if keep_all:  # live single-row inference: NaNs ok, caller needs target row only
        return df.reset_index(drop=True), feats
    return df.dropna(subset=feats + ["tmax"]).reset_index(drop=True), feats
