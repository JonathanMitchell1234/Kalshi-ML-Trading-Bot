"""Live weather signals: P(target_date T_max) + bracket probabilities.

Prediction for date D uses complete obs days strictly before D, plus stubs:
  - D == today: lags need yesterday (fresh via forecast-API past days) ✓
  - D == tomorrow: today's tmax stubbed from the GFS daily forecast
    (logged; backtest uses actuals — small documented skew).
Other stub fields persist yesterday's values. The Bayesian bias layer absorbs
systematic offsets (incl. ERA5-vs-TWC basis).
GFS snapshots are logged every call (wx_gfs_log.csv) for future blending.
"""
import json
import numpy as np
import pandas as pd
import joblib
import requests

from wx_data import load_obs, fetch_forecast_snapshot, fetch_ensemble_snapshot, CITIES, DATA_DIR
from wx_features import build_frame
from wx_calibrate import BayesT
from signals import MODEL_DIR

_wx = {}


def load_wx(city: str):
    if city not in _wx:
        c = city.lower()
        cal = json.loads((MODEL_DIR / f"wx_{c}_cal.json").read_text())
        bt = BayesT()
        # restore concentrated posterior (n=181: predictive ~ Student-t)
        bt.mn = cal.get("bias", 0.0)
        bt.an = cal.get("nu", 180) / 2 if "nu" in cal else 1e6
        bt.bn = cal.get("sigma", 7.0) ** 2 * bt.an
        bt.kn = 1e6
        bt.n = cal.get("n", 0)
        _wx[city] = {"gbm": joblib.load(MODEL_DIR / f"wx_{c}_gbm.pkl"),
                     "feats": joblib.load(MODEL_DIR / f"wx_{c}_feats.pkl"),
                     "target_mode": cal.get("target_mode", "absolute"),
                     "cal": bt, "kind": cal.get("kind", "bayes-t")}
    return _wx[city]


def live_daily_rows(city: str, days: int = 12) -> pd.DataFrame:
    """Complete recent daily rows from the forecast API (fresh, ~1d lag)."""
    snap = fetch_forecast_snapshot(city)
    tz = CITIES[city]["tz"]
    h = pd.DataFrame({"time": pd.to_datetime(snap["hourly"].get("time", [])),
                      "t": pd.to_numeric(pd.Series(snap["hourly"].get("temperature_2m", [])), errors="coerce"),
                      "cl": pd.to_numeric(pd.Series(snap["hourly"].get("cloud_cover", [])), errors="coerce"),
                      "hu": pd.to_numeric(pd.Series(snap["hourly"].get("relative_humidity_2m", [])), errors="coerce"),
                      "gu": pd.to_numeric(pd.Series(snap["hourly"].get("wind_gusts_10m", [])), errors="coerce")})
    h = h.dropna(subset=["time"])
    h["day"] = h["time"].dt.tz_localize(None).dt.date
    today_local = pd.Timestamp.now(tz=tz).date()
    g = h.groupby("day")["t"].agg(["max", "min", "count"])
    g = g.loc[[c for c in g.index if pd.Timestamp(c).date() < today_local]].tail(days)
    rows = pd.DataFrame({
        "date": pd.to_datetime(g.index),
        "tmax": g["max"].values, "tmin": g["min"].values,
        "precip": np.nan, "cloud": np.nan, "wind": np.nan,
        "hum": np.nan, "gust": np.nan,
    })
    cl = h.groupby("day")["cl"].mean()
    rows["cloud"] = rows["date"].dt.date.map(cl.to_dict()).values
    rows["hum"] = rows["date"].dt.date.map(h.groupby("day")["hu"].mean().to_dict()).values
    rows["gust"] = rows["date"].dt.date.map(h.groupby("day")["gu"].max().to_dict()).values
    # precip from daily array (past days included)
    dd = snap.get("daily", {})
    pmap = dict(zip(pd.to_datetime(dd.get("time", [])).strftime("%Y-%m-%d"),
                    pd.to_numeric(pd.Series(dd.get("precipitation_sum", [])), errors="coerce")))
    rows["precip"] = rows["date"].dt.strftime("%Y-%m-%d").map(pmap).fillna(0).values
    return rows.reset_index(drop=True)


def gfs_for_target(city: str, target, do_log: bool = True) -> tuple[float | None, dict]:
    """GFS daily-max forecast for target date + ensemble spread. Logs the call."""
    snap = fetch_forecast_snapshot(city)
    dd = snap.get("daily", {})
    tstr = pd.Timestamp(target).strftime("%Y-%m-%d")
    gfs = None
    try:
        idx = list(pd.to_datetime(dd.get("time", [])).strftime("%Y-%m-%d")).index(tstr)
        gfs = float(dd["temperature_2m_max"][idx])
    except (ValueError, TypeError, IndexError):
        pass
    ens_mean, ens_std = None, None
    try:
        e = fetch_ensemble_snapshot(city)
        eh = e.get("hourly", {})
        keys = [k for k in eh if k != "time"]
        if keys:
            member_max = []
            for k in keys:
                s = pd.Series(pd.to_numeric(pd.Series(eh[k]), errors="coerce"))
                tm = pd.to_datetime(eh["time"]).tz_localize(None)
                m = tm.strftime("%Y-%m-%d") == tstr
                if m.any():
                    member_max.append(float(np.nanmax(s[m].values)))
            if member_max:
                ens_mean, ens_std = float(np.mean(member_max)), float(np.std(member_max))
    except Exception:
        pass
    entry = {"ts": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "city": city,
             "target": tstr, "gfs_max": gfs, "ens_mean": ens_mean, "ens_std": ens_std}
    try:  # second opinion: ECMWF IFS daily max (multi-model future blending)
        from wx_data import CITIES as _CCe
        _c = _CCe[city]
        _e = requests.get("https://api.open-meteo.com/v1/forecast",
                          params={"latitude": _c["lat"], "longitude": _c["lon"],
                                  "daily": "temperature_2m_max", "timezone": _c["tz"],
                                  "temperature_unit": "fahrenheit", "forecast_days": 3,
                                  "models": "ecmwf_ifs"}, timeout=20).json()
        _dd = _e.get("daily", {})
        _idx = list(pd.to_datetime(_dd.get("time", [])).strftime("%Y-%m-%d")).index(tstr)
        entry["ecmwf_max"] = float(_dd["temperature_2m_max"][_idx])
    except Exception:
        pass
    if do_log:
        lp = DATA_DIR / "wx_gfs_log.csv"
        pd.DataFrame([entry]).to_csv(lp, mode="a", header=not lp.exists(), index=False)
    return gfs, {"ens_mean": ens_mean, "ens_std": ens_std}


def _stub_row(prev: pd.Series, tmax_est: float | None) -> dict:
    d = prev.to_dict()
    d["date"] = prev["date"] + pd.Timedelta(days=1)
    d["tmax"] = tmax_est if tmax_est is not None else prev["tmax"]
    return d


def _dpd_hist_full(city: str):
    try:
        from wx_features import station_dpd_daily
        return station_dpd_daily(city)
    except Exception:
        return None


def _wind_hist_full(city: str):
    try:
        from wx_features import daily_wind
        return daily_wind(pd.read_csv(DATA_DIR / f"wx_{city.lower()}_wind.csv", parse_dates=["time"]))
    except Exception:
        return None


def live_dpd_rows(city: str) -> pd.DataFrame:
    """Recent daily dewpoint-depression rows from the GFS snapshot hours."""
    from wx_features import daily_dpd
    snap = fetch_forecast_snapshot(city)
    h = snap.get("hourly", {})
    df = pd.DataFrame({"time": pd.to_datetime(h.get("time", [])),
                       "t2m": pd.to_numeric(pd.Series(h.get("temperature_2m", [])), errors="coerce"),
                       "dw": pd.to_numeric(pd.Series(h.get("dew_point_2m", [])), errors="coerce")})
    df["t2m"] = (df["t2m"] - 32) * 5 / 9 if df["t2m"].mean() > 45 else df["t2m"]
    df["dw"] = (df["dw"] - 32) * 5 / 9 if df["dw"].mean() > 45 else df["dw"]
    return daily_dpd(df.dropna(subset=["time"]))


def live_upper_rows(city: str) -> pd.DataFrame:
    """Noon-diagnosis upper-air rows for recent days, from GFS past-days hours."""
    from wx_features import daily_upper
    snap = fetch_forecast_snapshot(city)
    h = snap.get("hourly", {})
    df = pd.DataFrame({"time": pd.to_datetime(h.get("time", [])),
                       "mslp": pd.to_numeric(pd.Series(h.get("pressure_msl", [])), errors="coerce")})
    return daily_upper(df.dropna(subset=["time"]))


def _live_snd_frame(city: str, target) -> pd.DataFrame | None:
    """Sounding diagnosis rows for D-3..D-1 (target D): obs 12Z soundings,
    GFS forecast profile fallback per date. Tendencies computed across the
    (possibly mixed-source) series; snd_src_obs flags pure-obs rows."""
    try:
        from wx_raob import live_diag
        target = pd.Timestamp(target).tz_localize(None)
        rows = []
        for back in (1, 2, 3):
            d = (target - pd.Timedelta(days=back)).strftime("%Y-%m-%d")
            r = live_diag(city, d)
            if r is None:
                r = _gfs_diag(city, d)
            if r is not None:
                rows.append(r)
        if not rows:
            return None
        o = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        o["t850s_chg24"] = o["t850s"].diff(1)
        o["t850s_chg48"] = o["t850s"].diff(2)
        o["z500s_chg24"] = o["z500s"].diff(1)
        return o
    except Exception:
        return None


def _gfs_diag(city: str, datestr: str) -> dict | None:
    """Fallback diagnosis from the GFS forecast profile (flagged src=gfs)."""
    try:
        snap = fetch_forecast_snapshot(city)
        h = snap.get("hourly", {})
        t = pd.to_datetime(pd.Series(h.get("time", [])))
        df = pd.DataFrame({
            "time": t,
            "t850": pd.to_numeric(pd.Series(h.get("temperature_850hPa", [])), errors="coerce"),
            "t700": pd.to_numeric(pd.Series(h.get("temperature_700hPa", [])), errors="coerce"),
            "z500": pd.to_numeric(pd.Series(h.get("geopotential_height_500hPa", [])), errors="coerce"),
        }).dropna(subset=["time"])
        day = df[t.dt.strftime("%Y-%m-%d") == datestr]
        noon = day[(t.dt.hour >= 11) & (t.dt.hour <= 13)]
        if noon.empty:
            return None
        r = noon.iloc[len(noon) // 2]
        # GFS upper temps honor temperature_unit (F); soundings are C. Train in C.
        def _c(f):
            try:
                return (float(f) - 32) * 5 / 9
            except (TypeError, ValueError):
                return None
        return {"date": pd.Timestamp(datestr), "t850s": _c(r["t850"]),
                "dpd850": None, "t700s": _c(r["t700"]), "z500s": float(r["z500"]),
                "snd_src": "gfs"}
    except Exception:
        return None


def day_high_so_far(city: str) -> float | None:
    """Max observed hourly temp today (local), strictly before now.

    Prefers NWS station obs (truly observed); falls back to the forecast-API
    hourly proxy when NWS is unreachable.
    """
    try:
        from wx_data import fetch_nws_recent
        tz = CITIES[city]["tz"]
        now_local = pd.Timestamp.now(tz=tz).tz_localize(None)
        df = fetch_nws_recent(city)
        df["lt"] = pd.to_datetime(df["time"]).dt.tz_convert(tz).dt.tz_localize(None)
        past = df[(df["lt"].dt.date == now_local.date()) & (df["lt"] < now_local)]
        v = pd.to_numeric(past["temp_f"], errors="coerce").dropna()
        if len(v):
            return float(v.max())
    except Exception:
        pass
    try:
        snap = fetch_forecast_snapshot(city)
        tz = CITIES[city]["tz"]
        now_local = pd.Timestamp.now(tz=tz).tz_localize(None)
        h = pd.DataFrame({"time": pd.to_datetime(snap["hourly"].get("time", [])),
                          "t": pd.to_numeric(pd.Series(snap["hourly"].get("temperature_2m", [])), errors="coerce")})
        h = h.dropna()
        past = h[(h["time"].dt.date == now_local.date()) & (h["time"] < now_local)]
        return float(past["t"].max()) if len(past) else None
    except Exception:
        return None


def city_probs(city: str, target, brackets: list[dict], do_log: bool = True) -> tuple[dict, dict]:
    """Prediction + raw bracket probabilities for a target date."""
    from wx_calibrate import trailing_cal_regime
    from wx_data import assemble_obs
    M = load_wx(city)
    target = pd.Timestamp(target).tz_localize(None)
    obs = assemble_obs(city)
    try:  # fresh precip/cloud for recent days (station lacks them)
        live = live_daily_rows(city)[["date", "precip", "cloud"]]
        live["date"] = pd.to_datetime(live["date"]).dt.tz_localize(None)
        obs = obs.merge(live, on="date", how="left", suffixes=("", "_live"))
        for col in ("precip", "cloud"):
            obs[col] = obs[f"{col}_live"].combine_first(obs[col])
        obs = obs.drop(columns=["precip_live", "cloud_live"])
    except Exception:
        pass
    obs["date"] = pd.to_datetime(obs["date"]).dt.tz_localize(None)
    complete = obs[obs["date"] < target].copy()
    gfs, ens = gfs_for_target(city, target, do_log=do_log)
    ndfd = None
    try:
        from wx_data import fetch_ndfd_maxt
        ndfd = fetch_ndfd_maxt(city).get(target.strftime("%Y-%m-%d"))
        lp = DATA_DIR / "wx_ndfd_log.csv"
        pd.DataFrame([{"ts": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
                       "city": city, "target": target.strftime("%Y-%m-%d"),
                       "ndfd_max": ndfd}]).to_csv(lp, mode="a", header=not lp.exists(), index=False)
    except Exception:
        pass
    cur = complete["date"].max()
    stubs, guard = [], 0
    while cur < target and guard < 5:
        prev = complete.iloc[-1] if not stubs else pd.Series(stubs[-1])
        stubs.append(_stub_row(prev, gfs if cur + pd.Timedelta(days=1) == target else None))
        cur += pd.Timedelta(days=1)
        guard += 1
    want_snd = any(c.startswith(("t850s", "z500s", "dpd850", "t700s", "snd_")) for c in M["feats"])
    feat_df = pd.concat([complete, pd.DataFrame(stubs)], ignore_index=True) if stubs else complete
    try:
        up_live = live_upper_rows(city)
    except Exception:
        up_live = None
    _snd_live = _live_snd_frame(city, target) if want_snd else None
    from wx_features import daily_sw as _dsw
    from wx_data import CITIES as _CC3
    _sw_live = None
    try:
        _snap = fetch_forecast_snapshot(city)
        _h = pd.DataFrame({"time": pd.to_datetime(_snap["hourly"].get("time", [])),
                           "sw": pd.to_numeric(pd.Series(_snap["hourly"].get("shortwave_radiation", [])), errors="coerce")})
        _sw_live = _dsw(_h.dropna(subset=["time"]))
    except Exception:
        pass
    from wx_features import daily_wind as _dw2
    _wind_live = None
    try:
        _snap2 = fetch_forecast_snapshot(city)
        _h2 = pd.DataFrame({"time": pd.to_datetime(_snap2["hourly"].get("time", [])),
                            "wspd": pd.to_numeric(pd.Series(_snap2["hourly"].get("wind_speed_10m", [])), errors="coerce"),
                            "wdir": pd.to_numeric(pd.Series(_snap2["hourly"].get("wind_direction_10m", [])), errors="coerce")})
        _wind_live = _dw2(_h2.dropna(subset=["time"]))
    except Exception:
        pass
    try:
        _dpd_live = live_dpd_rows(city)
    except Exception:
        _dpd_live = None
    df, _ = build_frame(feat_df, up_live, keep_all=True, snd_daily=_snd_live,
                        lat_deg=_CC3[city]["lat"], sw_daily=_sw_live, wind_daily=_wind_live,
                        dpd_daily=_dpd_live)
    row = df[df["date"] == target]
    if row.empty:
        row = df.iloc[[-1]]
    for c in M["feats"]:  # never KeyError live: missing (rate-limited upstream) -> NaN, GBM handles
        if c not in row.columns:
            row[c] = np.nan
    t850_disp, t850_chg_disp = None, None
    if _snd_live is not None:
        try:
            _r = _snd_live[_snd_live["date"] == target - pd.Timedelta(days=1)]
            if len(_r):
                t850_disp = round(float(_r["t850s"].iloc[0]), 1)
                t850_chg_disp = round(float(_r["t850s_chg24"].iloc[0]), 1)
        except Exception:
            pass
    _mode = "anomaly" if str(M.get("target_mode", "absolute")) == "anomaly" else "absolute"
    point_raw = float(M["gbm"].predict(row[M["feats"]].values)[0])
    _clim_t = float(row["clim"].iloc[0]) if "clim" in row.columns else 0.0
    point = point_raw + (_clim_t if _mode == "anomaly" else 0.0)
    # physical validation guard: blend runaway leaves back toward NWP
    phys_blend = False
    _ref = ndfd if ndfd is not None else gfs
    try:
        up_hist = pd.read_csv(DATA_DIR / f"wx_{city.lower()}_mslp.csv", parse_dates=["time"])
        from wx_features import daily_upper as _du
        up_hist = _du(up_hist)
    except Exception:
        up_hist = None
    snd_hist = None
    if want_snd:
        try:
            import os as _os
            _sp = DATA_DIR / f"wx_{city.lower()}_raob.csv"
            if _os.path.exists(_sp):
                from wx_raob import daily_diag as _dd
                from wx_data import CITIES as _CC
                snd_hist = _dd(pd.read_csv(_sp, parse_dates=["time"]), _CC[city]["tz"])
        except Exception:
            pass
    df_full, _ = build_frame(complete, up_hist, snd_daily=snd_hist,
                             lat_deg=_CC3[city]["lat"], sw_daily=_sw_live,
                             wind_daily=_wind_hist_full(city), dpd_daily=_dpd_hist_full(city))
    for _c in M["feats"]:
        if _c not in df_full.columns:
            df_full[_c] = np.nan
    _rrow = row.iloc[0].to_dict() if hasattr(row, "iloc") else row
    cal, regime = trailing_cal_regime(M["gbm"], M["feats"], df_full, target, _rrow, mode=_mode)
    cp0 = cal.params
    if _ref is not None and abs(point - _ref) > 3 * cp0["sigma"]:
        point = (point + _ref) / 2  # runaway leaf: fall back halfway to physics
        phys_blend = True
    lo, hi = cal.interval(point)
    cp = cal.params
    pred = {"city": city, "target": target.strftime("%Y-%m-%d"), "pred": round(point, 2),
            "lo80": round(lo, 2), "hi80": round(hi, 2),
            "sigma": round(cp["sigma"], 2), "bias": round(cp["bias"], 2),
            "regime": regime,
            "t850s": t850_disp, "t850s_chg24": t850_chg_disp,
            "gfs_max": gfs, "ndfd_max": ndfd,
            "ens_mean": ens["ens_mean"], "ens_std": ens["ens_std"],
            "phys_blend": phys_blend,
            "cal_kind": "bayes-t-trailing"}
    # intraday conditioning: final max >= observed max-so-far (today only)
    obs_max = None
    try:
        tz = CITIES[city]["tz"]
        if target.strftime("%Y-%m-%d") == pd.Timestamp.now(tz=tz).strftime("%Y-%m-%d"):
            obs_max = day_high_so_far(city)
    except Exception:
        pass
    pred["obs_max_so_far"] = obs_max
    fairs = bracket_probs(city, point, brackets, cal, lo_bound=obs_max)
    return pred, fairs


def predict_tmax(city: str, target, do_log: bool = True) -> dict:
    """Point + interval forecast for target date's high."""
    pred, _ = city_probs(city, target, [], do_log=do_log)
    return pred


def bracket_probs(city: str, pred: float, brackets: list[dict], cal=None,
                  lo_bound: float | None = None) -> dict[str, float]:
    """ticker -> P(settles in bracket) under the Bayesian predictive t.

    lo_bound: condition on final max >= observed max-so-far (intraday today).
    Mass below the bound is impossible; tradable fairs keep raw (gap-style)
    convention — no renormalization.
    """
    if cal is None:
        cal = load_wx(city)["cal"]
    if lo_bound is not None:
        denom = max(1 - cal.cdf(lo_bound, pred), 1e-9)
    else:
        denom = 1.0
    out = {}
    for b in brackets:
        st = b["strike_type"]
        if st == "less":
            cap = float(b["cap"] or b["floor"] or 0)
            p = cal.cdf(cap, pred)
            if lo_bound is not None:
                p = max(p - cal.cdf(lo_bound, pred), 0.0) / denom
            out[b["ticker"]] = p
        elif st == "greater":
            fl = float(b["floor"] or 0)
            p = 1 - cal.cdf(fl, pred)
            if lo_bound is not None:
                p = p / denom if fl >= lo_bound else 1.0
            out[b["ticker"]] = p
        else:
            fl = float(b["floor"] or 0)
            cap = float(b.get("cap") or fl + 1)
            p = cal.prob_between(fl, cap, pred)
            if lo_bound is not None:
                lo2 = max(fl, lo_bound)
                p = (cal.prob_between(lo2, cap, pred) / denom) if cap > lo2 else 0.0
            out[b["ticker"]] = p
    return out
