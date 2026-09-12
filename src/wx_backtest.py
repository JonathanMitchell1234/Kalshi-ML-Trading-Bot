"""Backtest weather density on settled HIGHNY/HIGHCHI daily events.

Per settled event (date D): GBM prediction from the causal archive frame row
for D (prior days only — no leakage) -> Bayesian bracket probabilities over
that event's actual brackets. Winner + expiration_value = ground truth.
Reports: argmax hit-rate, mean winner-mass, Brier vs climatology, and
ERA5-vs-TWC basis where obs exists. No historical Kalshi prices exist, so
this measures density quality, not P&L.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from wx_data import load_obs, CITIES
from wx_features import build_frame
from wx_signals import load_wx, bracket_probs
from wx_calibrate import trailing_cal_regime
from tracker import save_backtest_run
import kalshi_client as K


def _parse_bracket(m: dict) -> dict:
    def _c(x):
        try:
            return int(round(float(x) * 100))
        except (TypeError, ValueError):
            return None
    return {"ticker": m.get("ticker"), "strike_type": m.get("strike_type"),
            "floor": m.get("floor_strike"), "cap": m.get("cap_strike"),
            "status": m.get("status"), "result": m.get("result") or None,
            "yes_ask": _c(m.get("yes_ask_dollars")),
            "expiration_value": m.get("expiration_value")}


def _pooled_frame():
    """Both cities stacked with city one-hot (mirrors train_pooled)."""
    import joblib as _jl
    from signals import MODEL_DIR as _MD
    from wx_data import assemble_obs
    pfeats = _jl.load(_MD / "wx_pooled_feats.pkl")
    frames = {}
    for _c in ("NYC", "CHI"):
        _obs = assemble_obs(_c)
        from wx_data import DATA_DIR as _DD
        import os as _os
        _up = pd.read_csv(_DD / f"wx_{_c.lower()}_mslp.csv", parse_dates=["time"])
        from wx_features import daily_upper as _du, build_frame as _bf
        _df, _ = _bf(_obs, _du(_up))
        _df["city"] = _c
        frames[_c] = _df
    full = pd.concat(frames, ignore_index=True)
    full = pd.get_dummies(full, columns=["city"], prefix="city", dtype=float)
    return full.sort_values("date").reset_index(drop=True), pfeats


def _crps_t(cal, loc_pred: float, outcome: float, span: float = 40.0, steps: int = 161) -> float:
    """Mean CRPS of the predictive CDF vs outcome (numeric integration)."""
    import numpy as _np
    xs = _np.linspace(outcome - span, outcome + span, steps)
    F = _np.array([cal.cdf(float(x), loc_pred) for x in xs])
    H = (xs >= outcome).astype(float)
    return float(_np.trapezoid((F - H) ** 2, xs))


def run_wx_backtest(city: str = "NYC", n_days: int = 120, progress=None, pooled: bool = False) -> dict:
    M = load_wx(city)
    if pooled:
        import joblib as _jl
        from signals import MODEL_DIR as _MD
        _pmodel = _jl.load(_MD / "wx_pooled_gbm.pkl")
        _pfull, _pfeats = _pooled_frame()
        _ppred = pd.Series(_pmodel.predict(_pfull[_pfeats].values), index=_pfull.index)
    series = CITIES[city]["series"]
    from wx_data import assemble_obs
    obs = assemble_obs(city)
    from wx_data import DATA_DIR
    from wx_features import daily_upper
    up = pd.read_csv(DATA_DIR / f"wx_{city.lower()}_mslp.csv", parse_dates=["time"])
    from wx_features import daily_wind as _dw
    import os as _osw
    _wp = DATA_DIR / f"wx_{city.lower()}_wind.csv"
    _wind = _dw(pd.read_csv(_wp, parse_dates=["time"])) if _osw.path.exists(_wp) else None
    from wx_features import daily_sw as _dsw
    import os as _os2
    _swp = DATA_DIR / f"wx_{city.lower()}_sw.csv"
    _sw = _dsw(pd.read_csv(_swp, parse_dates=["time"])) if _os2.path.exists(_swp) else None
    import os as _os
    sp = DATA_DIR / f"wx_{city.lower()}_raob.csv"
    import joblib as _jl
    from signals import MODEL_DIR as _MD
    want_snd = any(c.startswith(("t850s", "z500s", "dpd850", "t700s", "snd_"))
                   for c in _jl.load(_MD / f"wx_{city.lower()}_feats.pkl"))
    snd = None
    if want_snd and _os.path.exists(sp):
        from wx_raob import daily_diag
        from wx_data import CITIES as _C
        snd = daily_diag(pd.read_csv(sp, parse_dates=["time"]), _C[city]["tz"])
    from wx_data import CITIES as _CC2
    from wx_features import station_dpd_daily as _sdd
    try:
        _dpd = _sdd(city)
    except Exception:
        _dpd = None
    df, _ = build_frame(obs, daily_upper(up), snd_daily=snd,
                        lat_deg=_CC2[city]["lat"], sw_daily=_sw, wind_daily=_wind,
                        dpd_daily=_dpd)
    prow = dict(zip(df["date"], M["gbm"].predict(df[M["feats"]].values)))
    _anom_bt = str(M.get("target_mode", "absolute")) == "anomaly"
    _climmap = dict(zip(df["date"], df["clim"])) if _anom_bt else {}
    if pooled:
        # date-aligned pooled predictions on THIS city's frame (index-safe)
        _dmap = dict(zip(_pfull["date"].astype(str) + (_pfull["city_NYC"] > 0.5).map(
            {True: "NYC", False: "CHI"}),
            _ppred.values))
        df["_pp"] = [ _dmap.get(str(d) + city) for d in df["date"].astype(str) ]

    picks, scanned = [], 0
    today = datetime.now().date()
    d = today - timedelta(days=2)  # most recent plausibly-settled date
    while len(picks) < n_days and scanned < n_days + 40:
        scanned += 1
        t = f"{series}-{d.strftime('%y').upper()}{d.strftime('%b').upper()}{d.day:02d}"
        d -= timedelta(days=1)
        try:
            ev = K.get_event(t)
        except Exception:
            continue
        mk = ev.get("markets", [])
        if not mk or any((m.get("status") not in ("finalized", "settled")) for m in mk):
            continue
        winners = [m["ticker"] for m in mk if (m.get("result") or "") == "yes"]
        if len(winners) != 1:
            continue
        brackets = [_parse_bracket(m) for m in mk]
        key = pd.Timestamp(ev.get("event", {}).get("strike_date", "")[:10] or "NaT")
        # event date from ticker suffix instead (robust)
        try:
            evdate = pd.Timestamp(f"20{t.split('-')[1][:2]}-{t.split('-')[1][2:5]}-{t.split('-')[1][5:7]}")
        except Exception:
            continue
        if evdate not in prow:
            continue
        pred = float(prow[evdate])
        if _anom_bt:  # model speaks anomaly: translate to absolute degrees
            pred += float(_climmap.get(evdate, 0.0))
        _brow = df[df["date"] == evdate]
        _brow = _brow.iloc[0].to_dict() if len(_brow) else {}
        if pooled:
            _prow = _pfull[(_pfull["date"] == evdate) &
                           (_pfull["city_NYC"] == (1.0 if city == "NYC" else 0.0))]
            if _prow.empty:
                continue
            pred = float(_ppred.loc[_prow.index].iloc[0])
            _cal = trailing_cal_regime(None, [], df, evdate, _brow,
                                       pred_override=df["_pp"])[0]
        else:
            _cal = trailing_cal_regime(M["gbm"], M["feats"], df, evdate, _brow,
                                       mode=M.get("target_mode", "absolute"))[0]
        fairs = bracket_probs(city, pred, brackets, _cal)  # RAW density mass: gaps
        w = winners[0]                               # between listed brackets are dead outcomes
        try:
            expiry = float(mk[0].get("expiration_value"))
        except (TypeError, ValueError):
            expiry = None
        picks.append({
            "event_ticker": t,
            "expiry_ts": evdate.isoformat(),
            "signal_time": evdate.isoformat(),
            "spot": pred, "pred_price": round(pred, 2),
            "pred_bracket": max(fairs, key=fairs.get), "winner_bracket": w,
            "hit": int(max(fairs, key=fairs.get) == w),
            "err15": round(1 - fairs.get(w, 0.0), 4),
            "expiry_spot": expiry,
            "_pit": _cal.cdf(expiry, pred) if expiry is not None else None,
            "_crps": _crps_t(_cal, pred, expiry) if expiry is not None else None,
        })
        if progress:
            progress(len(picks), n_days)
    import numpy as _np
    from scipy import stats as _st
    _pits = [p["_pit"] for p in picks if p["_pit"] is not None]
    _crs = [p["_crps"] for p in picks if p["_crps"] is not None]
    for _p in picks:
        _p.pop("_pit", None)
        _p.pop("_crps", None)
    _ks = round(float(_st.kstest(_pits, "uniform").pvalue), 4) if len(_pits) > 10 else None
    _crps_mean = round(float(_np.mean(_crs)), 3) if _crs else None
    hits = sum(p["hit"] for p in picks)
    rid = save_backtest_run(n_events=len(picks), lead_min=0, picks=picks,
                            note=f"WX {city} density backtest, scanned={scanned}")
    res = {"run_id": rid, "n": len(picks),
           "hit_rate": round(hits / len(picks), 4) if picks else None,
           "mean_winner_mass": round(float(np.mean([1 - p["err15"] for p in picks])), 4) if picks else None,
           "crps_mean": _crps_mean, "pit_ks_pvalue": _ks, "n_pit": len(_pits)}
    try:
        from mlflow_log import log_run
        if res["hit_rate"] is not None:
            log_run("bitbot-weather", f"wxbacktest_{city}_{rid}",
                    params={"n_events": len(picks), "pooled": pooled},
                    metrics={"hit_rate": res["hit_rate"], "mean_winner_mass": res["mean_winner_mass"],
                             **({"crps_mean": _crps_mean} if _crps_mean else {}),
                             **({"pit_ks": _ks} if _ks else {})},
                    tags={"kind": "backtest", "city": city, "run_id": rid})
    except Exception:
        pass
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="NYC")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--pooled", action="store_true")
    a = ap.parse_args()
    print(run_wx_backtest(a.city, a.n, progress=lambda x, y: print(f"  {x}/{y}", end="\r"),
                           pooled=a.pooled))
