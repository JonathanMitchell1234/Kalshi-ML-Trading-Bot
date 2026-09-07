"""Bayesian calibration for weather T_max: Normal-Gamma conjugate residuals.

Model: error e = actual - GBM_pred. Level/shift absorbs TWC-vs-ERA5 basis +
model bias; variance absorbs regime noise. Conjugate update on trailing
residuals -> posterior predictive is Student-t(nu, loc, scale).
Bracket P(a<=T<b) from the t-CDF.credible intervals fall out free.
Also fits isotonic-on-PIT challenger; see train_wx for the Brier shootout.
Pure stdlib math (scipy for t CDF).
"""
import math
import numpy as np
import pandas as pd
from scipy import stats


class BayesT:
    def __init__(self, m0: float = 0.0, k0: float = 1.0, a0: float = 2.0, b0: float = 8.0):
        self.m0, self.k0, self.a0, self.b0 = m0, k0, a0, b0
        self.n = 0
        self.mn, self.kn, self.an, self.bn = m0, k0, a0, b0

    def update(self, errors: np.ndarray):
        e = np.asarray(errors, dtype=float)
        e = e[~np.isnan(e)]
        n = len(e)
        if n == 0:
            return self
        xbar = float(e.mean())
        ssd = float(((e - xbar) ** 2).sum())
        kn = self.k0 + n
        mn = (self.k0 * self.m0 + n * xbar) / kn
        an = self.a0 + n / 2
        bn = self.b0 + 0.5 * ssd + 0.5 * self.k0 * n * (xbar - self.m0) ** 2 / kn
        self.n, self.mn, self.kn, self.an, self.bn = n, mn, kn, an, bn
        return self

    @property
    def params(self) -> dict:
        nu = 2 * self.an
        scale = math.sqrt(self.bn * (self.kn + 1) / (self.an * self.kn))
        return {"bias": self.mn, "sigma": math.sqrt(self.bn / self.an),
                "nu": nu, "scale": scale, "n": self.n}

    def cdf(self, x: float, loc_pred: float) -> float:
        p = self.params
        return float(stats.t.cdf(x - (loc_pred + p["bias"]), df=p["nu"], scale=p["scale"]))

    def prob_between(self, lo: float, hi: float, loc_pred: float) -> float:
        return max(self.cdf(hi, loc_pred) - self.cdf(lo, loc_pred), 0.0)

    def interval(self, loc_pred: float, level: float = 0.80) -> tuple[float, float]:
        p = self.params
        q = stats.t.ppf([(1 - level) / 2, 1 - (1 - level) / 2], df=p["nu"], scale=p["scale"])
        return loc_pred + p["bias"] + float(q[0]), loc_pred + p["bias"] + float(q[1])


def trailing_cal(gbm, feats: list[str], frame: pd.DataFrame, asof,
                 window_days: int = 90, floor: str = "2025-01-01") -> "BayesT":
    """Adaptive recalibration: fresh Normal-Gamma posterior from trailing
    OUT-OF-SAMPLE residuals (rows after `floor` = post-train era, strictly
    before `asof`). Summer calm -> tight sigma; spring chaos -> wide."""
    asof = pd.Timestamp(asof).tz_localize(None)
    hist = frame[(frame["date"] >= asof - pd.Timedelta(days=window_days)) &
                 (frame["date"] < asof) &
                 (frame["date"] >= pd.Timestamp(floor))].copy()
    if len(hist) < 15:
        hist = frame[(frame["date"] < asof) & (frame["date"] >= pd.Timestamp(floor))].tail(60)
    pred = gbm.predict(hist[feats].values)
    return BayesT().update(hist["tmax"].values - pred)


def regime_of(row: dict | pd.Series) -> str:
    """Frontal vs calm day from pressure tendency + dewpoint depression.

    Springer takeaway: storm-regime days carry most of the error mass.
    |mslp_chg24| >= 4 hPa or dry 850hPa (dpd >= 15C... use F: dpd850 stored C;
    threshold ~8C) -> frontal. Calm otherwise. Missing drivers -> calm.
    """
    try:
        chg = abs(float(row.get("mslp_chg24", 0) or 0))
    except (TypeError, ValueError):
        chg = 0.0
    try:
        dpd = float(row.get("dpd850", 0) or 0)
    except (TypeError, ValueError):
        dpd = 0.0
    return "frontal" if (chg >= 4.0 or dpd >= 8.0) else "calm"


def _target_vals(frame: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "anomaly" and "clim" in frame.columns:
        return frame["tmax"] - frame["clim"]
    return frame["tmax"]


def trailing_cal_regime(gbm, feats: list[str], frame: pd.DataFrame, asof,
                        row: dict | pd.Series, window_days: int | None = None,
                        floor: str = "2025-01-01",
                        pred_override: pd.Series | None = None,
                        mode: str = "absolute") -> tuple["BayesT", str]:
    """Regime-conditional calibration: separate residual posterior for the
    target day's regime (frontal days get their own wide sigma instead of
    polluting calm-day precision). Falls back to pooled when thin.
    pred_override: precomputed predictions aligned to frame (pooled models).
    """
    import os as _os
    if window_days is None:
        window_days = int(_os.getenv("WX_CAL_WINDOW", "45"))
    reg = regime_of(row)
    asof = pd.Timestamp(asof).tz_localize(None)
    hist = frame[(frame["date"] >= asof - pd.Timedelta(days=window_days)) &
                 (frame["date"] < asof) &
                 (frame["date"] >= pd.Timestamp(floor))].copy()
    if len(hist) < 15:
        hist = frame[(frame["date"] < asof) & (frame["date"] >= pd.Timestamp(floor))].tail(90)
    if pred_override is not None:
        pred = pred_override.loc[hist.index].values
    else:
        pred = gbm.predict(hist[feats].values)
    hist = hist.assign(_res=_target_vals(hist, mode).values - pred,
                       _reg=hist.apply(regime_of, axis=1))
    sub = hist[hist["_reg"] == reg]
    if len(sub) < 12:  # thin regime -> pooled (honest fallback)
        return BayesT().update(hist["_res"].values), reg + "+pooled"
    return BayesT().update(sub["_res"].values), reg
