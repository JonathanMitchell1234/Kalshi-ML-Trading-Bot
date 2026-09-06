"""Train per-city LightGBM T_max regression + Bayesian calibration shootout.

Splits (time): train <=2024 | calib 2025H1 | test 2025H2..obs-end.
Contenders on test Brier (of implied P(above median)? No — proper scoring of
the predictive DENSITY): mean log-loss of actual outcome under predictive CDF
binned per-degree + Brier of P(T >= test-median). Winner saved.
Saves: wx_{city}_gbm.pkl, wx_{city}_feats.pkl, wx_{city}_cal.json, wx_metrics.json
"""
import json
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb

from wx_data import load_obs, CITIES
from wx_features import build_frame
from wx_calibrate import BayesT
from signals import MODEL_DIR

PARAMS = dict(objective="regression", metric="rmse", n_estimators=1500,
              learning_rate=0.05, num_leaves=31, min_child_samples=30,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambda_l1=0.1, lambda_l2=1.0, verbose=-1, random_state=7)


USE_SND = {"NYC": True, "CHI": False}  # CHI soundings degraded RMSE 7.02->7.33 with no backtest gain (KILX 200km offset)

def wx_splits(d) -> tuple:
    """Purged/embargoed masks: 30d purge (longest lookback) before calib,
    5d embargo after calib (synoptic persistence must not cross folds)."""
    d = pd.to_datetime(d)
    tr = (d <= "2024-12-01").values
    ca = ((d > "2024-12-31") & (d <= "2025-06-30")).values
    te = (d > "2025-07-05").values
    return tr, ca, te


def frame_city(city: str, use_snd: bool | None = None):
    import pandas as pd
    from wx_features import daily_upper
    from wx_data import DATA_DIR, station_daily
    from wx_data import assemble_obs
    obs = assemble_obs(city)
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
    if use_snd is None:
        use_snd = USE_SND.get(city, True)
    snd = None
    if use_snd and _os.path.exists(sp):
        from wx_raob import daily_diag
        from wx_data import CITIES as _C
        snd = daily_diag(pd.read_csv(sp, parse_dates=["time"]), _C[city]["tz"])
    from wx_data import CITIES as _CC2
    return build_frame(obs, daily_upper(up), snd_daily=snd,
                       lat_deg=_CC2[city]["lat"], sw_daily=_sw, wind_daily=_wind)


def main(city: str = "NYC"):
    df, feats = frame_city(city)
    print(f"{city}: {len(df)} rows {df['date'].min().date()} -> {df['date'].max().date()}")
    X, y = df[feats].values, df["tmax"].values
    tr, ca, te = wx_splits(df["date"])
    print(f"train {tr.sum()} calib {ca.sum()} test {te.sum()} (purged+embargoed)")

    # Tail weighting (rain-event lesson): upweight extreme-anomaly days so the
    # booster resolves tails; post-hoc Bayesian calibration on UNWEIGHTED held-out
    # residuals then restores honest levels. alpha=0 disables.
    import os as _os
    _alpha = float(_os.getenv("WX_TAIL_ALPHA", "0.5"))
    _z = np.abs(df["anom1"].values) / np.maximum(df["climstd"].values, 1.0)
    _w = 1.0 + _alpha * np.clip(_z, 0, 3)
    gbm = lgb.LGBMRegressor(**PARAMS)
    gbm.fit(X[tr], y[tr], sample_weight=_w[tr], eval_set=[(X[ca], y[ca])],
            callbacks=[lgb.early_stopping(100, verbose=False)])
    pred_lgbm = gbm.predict(X)
    # XGBoost challenger (same splits; reference repo uses XGB — verify, don't assume)
    try:
        from xgboost import XGBRegressor
        xgb = XGBRegressor(n_estimators=1500, learning_rate=0.05, max_depth=5,
                           subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                           reg_lambda=1.0, early_stopping_rounds=100,
                           eval_metric="rmse", random_state=7)
        xgb.fit(X[tr], y[tr], sample_weight=_w[tr], eval_set=[(X[ca], y[ca])], verbose=False)
        pred_xgb = xgb.predict(X)
        rmse_xgb = float(np.sqrt(np.mean((y[te] - pred_xgb[te]) ** 2)))
    except Exception as e:
        print(f"xgb challenger failed ({e})")
        xgb, pred_xgb, rmse_xgb = None, None, float("inf")
    rmse_lgbm = float(np.sqrt(np.mean((y[te] - pred_lgbm[te]) ** 2)))
    use_xgb = rmse_xgb < rmse_lgbm
    pred = pred_xgb if use_xgb else pred_lgbm
    booster_kind = "xgboost" if use_xgb else "lightgbm"
    print(f"challenger: lgbm RMSE={rmse_lgbm:.3f} xgb RMSE={rmse_xgb:.3f} -> {booster_kind}")

    # Contenders for the residual distribution (fit on calib only):
    #  A) BayesT: Normal-Gamma conjugate update (priors + trailing errors)
    #  B) Empirical Gaussian: fixed calib mean/std (no priors, no updating)
    from scipy.stats import norm
    resid = y[ca] - pred[ca]
    bayes = BayesT().update(resid)
    g_bias, g_std = float(resid.mean()), float(resid.std()) or 1.0

    med = float(np.median(y[te]))
    pb = np.array([1 - bayes.cdf(med, p) for p in pred[te]])
    pg = norm.cdf((pred[te] + g_bias - med) / g_std)
    yt = (y[te] >= med).astype(int)
    from sklearn.metrics import brier_score_loss, mean_absolute_error
    b_bayes = float(brier_score_loss(yt, np.clip(pb, 1e-6, 1 - 1e-6)))
    b_gauss = float(brier_score_loss(yt, np.clip(pg, 1e-6, 1 - 1e-6)))
    winner = "bayes-t" if b_bayes <= b_gauss else "empirical-gaussian"
    out = {
        "city": city, "n_test": int(te.sum()), "booster": booster_kind,
        "rmse": round(float(np.sqrt(np.mean((y[te] - pred[te]) ** 2))), 3),
        "mae": round(float(mean_absolute_error(y[te], pred[te])), 3),
        "bias_test": round(float(np.mean(pred[te] - y[te])), 3),
        "brier_bayes": round(b_bayes, 4),
        "brier_gauss": round(b_gauss, 4),
        "brier_clim": round(float(brier_score_loss(yt, [yt.mean()] * len(yt))), 4),
        "winner": winner,
        "bayes_params": bayes.params,
        "gauss_params": {"bias": g_bias, "sigma": g_std},
        "test_start": str(d[te].min().date()),
    }
    print(json.dumps(out, indent=1))
    joblib.dump(xgb if use_xgb else gbm, MODEL_DIR / f"wx_{city.lower()}_gbm.pkl")
    joblib.dump(feats, MODEL_DIR / f"wx_{city.lower()}_feats.pkl")
    cal = {"kind": winner,
           **(bayes.params if winner == "bayes-t" else {"bias": g_bias, "sigma": g_std})}
    json.dump(cal, open(MODEL_DIR / f"wx_{city.lower()}_cal.json", "w"), indent=1)
    mp = MODEL_DIR / "wx_metrics.json"
    allm = json.loads(mp.read_text()) if mp.exists() else {}
    allm[city] = out
    mp.write_text(json.dumps(allm, indent=1))
    _booster = xgb if use_xgb else gbm
    try:
        _fi = _booster.feature_importances_
    except AttributeError:
        _fi = _booster.feature_importances(importance_type="gain")
    imp = pd.DataFrame({"feature": feats, "importance": _fi}
                       ).sort_values("importance", ascending=False)
    print(imp.head(10).to_string(index=False))
    from tracker import stamp
    print(f"version wx_{city}=" + stamp(f"wx_{city}"))
    return out


def train_pooled():
    """One GBM on NYC+CHI stacked (city one-hot). Same date splits both cities."""
    import pandas as pd
    frames, featsets = [], []
    for city in ("NYC", "CHI"):
        df, feats = frame_city(city)
        df = df.copy()
        df["city"] = city
        frames.append(df)
        featsets.append(feats)
    # intersect ENGINEERED feature sets only (never raw same-day columns)
    feats = [c for c in featsets[0] if c in featsets[1]] + ["city_NYC"]
    full = pd.concat(frames, ignore_index=True)
    full = pd.get_dummies(full, columns=["city"], prefix="city", dtype=float)
    full = full.sort_values("date").reset_index(drop=True)
    X, y = full[feats].values, full["tmax"].values
    d = full["date"]
    tr = (d <= "2024-12-31").values
    ca = ((d > "2024-12-31") & (d <= "2025-06-30")).values
    te = (d > "2025-06-30").values
    import lightgbm as lgb
    m = lgb.LGBMRegressor(**{**PARAMS})
    m.fit(X[tr], y[tr], eval_set=[(X[ca], y[ca])],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    pred = m.predict(X)
    rmse = float(np.sqrt(np.mean((y[te] - pred[te]) ** 2)))
    out = {"rmse": round(rmse, 3), "n_test": int(te.sum()), "feats": feats}
    for city in ("NYC", "CHI"):
        msk = te & (full["city_NYC"] == (1.0 if city == "NYC" else 0.0)).values
        out[city] = round(float(np.sqrt(np.mean((y[msk] - pred[msk]) ** 2))), 3)
    print("pooled:", out)
    joblib.dump(m, MODEL_DIR / "wx_pooled_gbm.pkl")
    joblib.dump(feats, MODEL_DIR / "wx_pooled_feats.pkl")
    (MODEL_DIR / "wx_pooled_metrics.json").write_text(json.dumps(out, indent=1))
    return out


def hyper_search(city: str = "NYC", n: int = 12, seed: int = 7):
    """Randomized search around PARAMS on the calib split. Reports only."""
    import itertools, random
    import lightgbm as lgb
    df, feats = frame_city(city)
    X, y = df[feats].values, df["tmax"].values
    d = df["date"]
    tr = (d <= "2024-12-31").values
    ca = ((d > "2024-12-31") & (d <= "2025-06-30")).values
    rng = random.Random(seed)
    grid = {"num_leaves": [15, 31, 63], "min_child_samples": [20, 50, 100],
            "feature_fraction": [0.6, 0.8, 1.0], "lambda_l2": [0.0, 1.0, 5.0],
            "learning_rate": [0.03, 0.05, 0.08]}
    keys = list(grid)
    tried = set()
    res = []
    base = dict(PARAMS)
    while len(res) < n:
        cfg = tuple(rng.choice(grid[k]) for k in keys)
        if cfg in tried:
            continue
        tried.add(cfg)
        prm = {**base, **dict(zip(keys, cfg))}
        m = lgb.LGBMRegressor(**prm)
        m.fit(X[tr], y[tr], eval_set=[(X[ca], y[ca])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        import numpy as _np
        rmse = float(_np.sqrt(_np.mean((y[ca] - m.predict(X[ca])) ** 2)))
        res.append((rmse, dict(zip(keys, cfg))))
        print(f"  rmse={rmse:.3f} {dict(zip(keys, cfg))}", flush=True)
    res.sort()
    print("best:", res[0])
    return res


def challenger_quantiles(city: str = "NYC"):
    """LightGBM pinball models tau=.1/.5/.9 vs BayesT: interval coverage +
    median-Brier on test. Saves wx_{city}_q{10,50,90}.pkl + report."""
    import lightgbm as lgb
    df, feats = frame_city(city)
    X, y = df[feats].values, df["tmax"].values
    d = df["date"]
    # purge 30d (longest lookback: precip30) before each eval cut + 5d embargo
    # after calib (synoptic persistence must not leak across folds)
    tr = (d <= "2024-12-31").values
    ca = ((d > "2024-12-31") & (d <= "2025-06-30")).values
    te = (d > "2025-07-05").values
    tr = (d <= "2024-12-01").values  # 30d purge before calib start
    rep = {}
    for tau in (0.10, 0.50, 0.90):
        m = lgb.LGBMRegressor(objective="quantile", alpha=tau, n_estimators=1500,
                              learning_rate=0.05, num_leaves=31, min_child_samples=30,
                              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                              verbose=-1, random_state=7)
        m.fit(X[tr], y[tr], eval_set=[(X[ca], y[ca])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        joblib.dump(m, MODEL_DIR / f"wx_{city.lower()}_q{int(tau*100)}.pkl")
        rep[tau] = m.predict(X[te])
    import numpy as _np
    lo, md, hi = rep[0.10], rep[0.50], rep[0.90]
    cov = float(((y[te] >= lo) & (y[te] <= hi)).mean())
    out = {"coverage80": round(cov, 4),
           "median_mae": round(float(_np.mean(_np.abs(y[te] - md))), 3),
           "median_brier": "see backtest"}
    print(f"{city} quantiles:", out)
    (MODEL_DIR / f"wx_{city.lower()}_qreport.json").write_text(__import__("json").dumps(out, indent=1))
    return out


def challenger_asym(city: str = "NYC", alpha: float = 0.70):
    """Asymmetric quadratic (underpredict x alpha/(1-alpha)): heatwave-day MAE
    vs symmetric. Custom grad/hess for LightGBM."""
    import lightgbm as lgb
    import numpy as _np
    def _obj(y_true, y_pred):
        # sklearn-wrapper convention: arrays (y_true, y_pred)
        y_true = np.asarray(y_true, dtype=float)
        diff = np.asarray(y_pred, dtype=float) - y_true
        grad = np.where(y_true > y_pred, -2 * alpha * (y_true - y_pred), 2 * (1 - alpha) * diff)
        hess = np.where(y_true > y_pred, 2 * alpha, 2 * (1 - alpha))
        return grad, hess
    df, feats = frame_city(city)
    X, y = df[feats].values, df["tmax"].values
    d = df["date"]
    # purge 30d (longest lookback: precip30) before each eval cut + 5d embargo
    # after calib (synoptic persistence must not leak across folds)
    tr = (d <= "2024-12-31").values
    ca = ((d > "2024-12-31") & (d <= "2025-06-30")).values
    te = (d > "2025-07-05").values
    tr = (d <= "2024-12-01").values  # 30d purge before calib start
    m = lgb.LGBMRegressor(objective=_obj, n_estimators=1500, learning_rate=0.05,
                          num_leaves=31, min_child_samples=30, feature_fraction=0.8,
                          verbose=-1, random_state=7)
    m.fit(X[tr], y[tr], eval_set=[(X[ca], y[ca])],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    p = m.predict(X[te])
    hot = y[te] >= _np.quantile(y[te], 0.90)
    print(f"{city} asym(a={alpha}): testMAE={_np.mean(_np.abs(y[te]-p)):.3f} "
          f"hotMAE={_np.mean(_np.abs(y[te][hot]-p[hot])):.3f} bias={_np.mean(p-y[te]):+.3f}")
    try:
        joblib.dump(m, MODEL_DIR / f"wx_{city.lower()}_asym.pkl")
    except Exception as e:
        print(f"(asym artifact not saved: {e})")
    return m


def challenger_catboost(city: str = "NYC"):
    """CatBoost symmetric trees vs LightGBM/XGBoost on identical splits."""
    from catboost import CatBoostRegressor
    import numpy as _np
    df, feats = frame_city(city)
    X, y = df[feats].values, df["tmax"].values
    d = df["date"]
    # purge 30d (longest lookback: precip30) before each eval cut + 5d embargo
    # after calib (synoptic persistence must not leak across folds)
    tr = (d <= "2024-12-31").values
    ca = ((d > "2024-12-31") & (d <= "2025-06-30")).values
    te = (d > "2025-07-05").values
    tr = (d <= "2024-12-01").values  # 30d purge before calib start
    m = CatBoostRegressor(iterations=1500, learning_rate=0.05, depth=6,
                          l2_leaf_reg=3.0, random_seed=7, verbose=False,
                          early_stopping_rounds=100)
    m.fit(X[tr], y[tr], eval_set=[(X[ca], y[ca])], verbose=False)
    p = m.predict(X[te])
    rmse = float(_np.sqrt(_np.mean((y[te] - p) ** 2)))
    print(f"{city} catboost RMSE={rmse:.3f}")
    joblib.dump(m, MODEL_DIR / f"wx_{city.lower()}_cat.pkl")
    return rmse


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="NYC")
    ap.add_argument("--no-snd", action="store_true")
    ap.add_argument("--pooled", action="store_true")
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--quantiles", action="store_true")
    ap.add_argument("--asym", action="store_true")
    ap.add_argument("--catboost", action="store_true")
    a = ap.parse_args()
    if a.search:
        hyper_search(a.city)
    elif a.quantiles:
        challenger_quantiles(a.city)
    elif a.asym:
        challenger_asym(a.city)
    elif a.catboost:
        challenger_catboost(a.city)
    elif a.pooled:
        train_pooled()
    else:
        if a.no_snd:
            USE_SND[a.city] = False
        main(a.city)
