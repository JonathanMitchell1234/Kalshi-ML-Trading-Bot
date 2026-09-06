"""Dataset v2 for 15-min DIRECTION: P(close[t+1] > close[t]).

Grid: BTC 15-min closes (decision points, all features strictly causal).
Feature blocks:
  A. base 15m technicals (reuse features.add_features, raw levels excluded)
  B. 1-min microstructure aggregates over the last 15 closed 1m candles:
     micro-vol, path return, up-minute share, EMA9/21 cross, volume spike,
     last-5m vs first-10m momentum split, wick dominance
  C. cross-asset leads (ETH/SOL/XRP): 15m returns/dist-to-EMA, 1m-based
     15-min path returns, BTC-relative strength, 5-min lead momentum
Label: y_up = 1 if next 15m close > current close.
"""
import numpy as np
import pandas as pd

from features import add_features, get_feature_cols

LEADERS = ("ETH", "SOL", "XRP")


M_KEYS = ["vol", "path", "upfrac", "ema921", "volsum", "late_early", "range", "wick",
          "cvd", "buyratio"]


def _m1_block_stats(m1: pd.DataFrame) -> dict:
    """Stats over recent closed 1m candles ending at grid time. Pure history.

    Tolerates gappy minutes (thin pairs): needs >=5 prints, else {} and the
    caller fills NaN (the GBM handles NaN natively).
    """
    c = m1["close"].values
    if len(c) < 12:
        return {}
    ret = np.diff(np.log(c))
    v = m1["volume"].values
    ema9 = pd.Series(c).ewm(span=9, adjust=False).mean().iloc[-1]
    ema21 = pd.Series(c).ewm(span=21, adjust=False).mean().iloc[-1]
    up = (ret > 0).mean() if len(ret) else 0.5
    first = np.log(c[min(9, len(c) - 1)] / c[0]) if len(c) > 10 else 0.0
    last = np.log(c[-1] / c[-6]) if len(c) >= 6 else 0.0
    rng = ((m1["high"] - m1["low"]) / m1["close"]).mean()
    body = (abs(m1["close"] - m1["open"]) / m1["close"]).mean()
    if {"tb", "tq", "qv", "volume"}.issubset(m1.columns) and float(m1["volume"].sum()) > 0:
        m_cvd = float((2 * m1["tb"] - m1["volume"]).sum() / m1["volume"].sum())
        _qv = float(m1["qv"].sum())
        m_buy = float(m1["tq"].sum() / _qv) if _qv > 0 else 0.5
        if not np.isfinite(m_cvd):
            m_cvd = 0.0
        if not np.isfinite(m_buy):
            m_buy = 0.5
    else:
        m_cvd, m_buy = 0.0, 0.5  # venue lacks taker fields: neutral, not NaN
    return {
        "vol": float(ret.std()) if len(ret) else 0.0,
        "path": float(ret.sum()) if len(ret) else 0.0,
        "upfrac": float(up),
        "ema921": float(ema9 / ema21 - 1),
        "volsum": float(v.sum()),
        "late_early": float(last - first),
        "range": float(rng),
        "wick": float(max(rng - body, 0)),
        "cvd": m_cvd,
        "buyratio": m_buy,
    }


def _add_microstructure(grid: pd.DataFrame, m1: pd.DataFrame, prefix: str,
                        vol_med: pd.Series | None = None) -> pd.DataFrame:
    """Attach block stats + volume-spike ratio (vs causal trailing median)."""
    m1 = m1.sort_values("time").reset_index(drop=True)
    stats, vols = [], []
    for t in grid["time"]:
        block = m1[(m1["time"] > t - pd.Timedelta(minutes=16)) & (m1["time"] <= t)].tail(15)
        s = _m1_block_stats(block)
        stats.append(s)
        vols.append(s.get("volsum", np.nan))
    sdf = pd.DataFrame(stats, index=grid.index).add_prefix(prefix)
    grid = pd.concat([grid, sdf], axis=1)
    volsum = pd.Series(vols, index=grid.index)
    med = volsum.rolling(96, min_periods=24).median()  # causal trailing median (~24h)
    grid[f"{prefix}volspike"] = volsum / med.replace(0, np.nan)
    return grid


def _add_cross_asset(grid: pd.DataFrame, bars15: dict[str, pd.DataFrame],
                     bars1: dict[str, pd.DataFrame], leaders: tuple = LEADERS) -> pd.DataFrame:
    btc = grid.set_index("time")["close"]
    for a in leaders:
        d15 = bars15[a].set_index("time").reindex(btc.index, method="ffill", limit=2)
        grid[f"x_{a.lower()}_ret1"] = np.log(d15["close"] / d15["close"].shift(1)).values
        grid[f"x_{a.lower()}_ret4"] = np.log(d15["close"] / d15["close"].shift(4)).values
        ema8 = d15["close"].ewm(span=8, adjust=False).mean()
        grid[f"x_{a.lower()}_dist8"] = (d15["close"] / ema8 - 1).values
        grid[f"x_{a.lower()}_vold"] = (d15["volume"] / d15["volume"].rolling(24).mean().replace(0, np.nan)).values
        # 1m-based: asset path return over same 15m block, BTC-relative
        m1 = bars1[a].sort_values("time")
        paths = []
        for t in grid["time"]:
            blk = m1[(m1["time"] > t - pd.Timedelta(minutes=15)) & (m1["time"] <= t)]
            paths.append(float(np.log(blk["close"].iloc[-1] / blk["close"].iloc[0]))
                         if len(blk) >= 12 else np.nan)
        apath = pd.Series(paths, index=grid.index)
        bpath = grid["m_path"]
        grid[f"x_{a.lower()}_path15"] = apath
        grid[f"x_{a.lower()}_relstr"] = apath - (bpath if isinstance(bpath, pd.Series) else 0.0)
        # 5-min lead: asset's [t-20, t-5] move (what BTC hasn't 'seen' in last 5m)
        leads = []
        for t in grid["time"]:
            w = m1[(m1["time"] > t - pd.Timedelta(minutes=20)) & (m1["time"] <= t - pd.Timedelta(minutes=5))]
            leads.append(float(np.log(w["close"].iloc[-1] / w["close"].iloc[0]))
                         if len(w) >= 10 else np.nan)
        grid[f"x_{a.lower()}_lead5"] = leads
    # thin-minute gaps -> neutral 0 + known flag (never drop rows over this)
    for a in leaders:
        al = a.lower()
        grid[f"x_{al}_mknown"] = grid[f"x_{al}_path15"].notna().astype(int)
        for col in (f"x_{al}_path15", f"x_{al}_relstr", f"x_{al}_lead5"):
            grid[col] = grid[col].fillna(0.0)
    return grid


def build_dataset(btc15: pd.DataFrame, btc1: pd.DataFrame,
                  bars15: dict, bars1: dict, leaders: tuple = LEADERS,
                  asset: str = "BTC") -> tuple[pd.DataFrame, list[str]]:
    for df in [btc15, btc1, *bars15.values(), *bars1.values()]:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    grid = add_features(btc15).copy()
    grid = _add_microstructure(grid, btc1, prefix="m_")
    grid = _add_cross_asset(grid, bars15, bars1, leaders)
    grid["y_up"] = (grid["close"].shift(-1) > grid["close"]).astype(int)
    grid = _flow_features(grid, asset)
    grid = _add_orderflow(grid, btc15)
    # PCA market factors over the fixed universe (fit: train era only)
    _rets = _universe_rets(btc15, bars15, asset)
    _pcafit = _fit_pca(_rets)
    grid = _apply_pca(grid, _rets, _pcafit)
    grid = grid.drop(columns=["tb", "tq", "qv"], errors="ignore")  # raw taker levels: engineered, never feats
    grid = grid.dropna().reset_index(drop=True)
    base = get_feature_cols(grid)
    extra = [c for c in grid.columns if c.startswith(("m_", "x_"))]
    feats = [c for c in base + [c for c in extra if c not in base]
             if not c.startswith("y_")]  # never let the label leak in
    feats += [c for c in FLOW_FEATS.get(asset, []) if c in grid.columns]
    # orderflow + PCA cols are in base already (not raw/excluded); assert it
    for _c in ("cvd_ratio", "buyer_ratio", "trade_vel", "mkt_mom", "idio_div"):
        assert _c in feats, f"feature {_c} missing from feats"
    feats = list(dict.fromkeys(feats))
    return grid, feats, _pcafit


UNIVERSE = ("BTC", "ETH", "SOL", "XRP")


def _universe_rets(btc15: pd.DataFrame, bars15: dict, asset: str) -> pd.DataFrame:
    """15m log returns for the fixed 4-asset universe, indexed by grid time."""
    src = dict(bars15 or {})
    src[asset] = btc15
    out = {}
    for a in UNIVERSE:
        if a in src:
            s = src[a].set_index("time")["close"]
            out[a] = np.log(s / s.shift(1))
    return pd.DataFrame(out)


def _add_orderflow(grid: pd.DataFrame, bars15: pd.DataFrame) -> pd.DataFrame:
    """CVD / buyer-ratio / trade velocity from taker fields (kept, not dropped).

    cvd = buys - sells = 2*tb - volume. Missing (pre-refetch rows) -> 0 + flag.
    """
    grid = grid.copy()
    b = bars15.set_index("time")
    for col in ("tb", "tq", "qv", "count"):
        if col not in b.columns:
            b[col] = float("nan")
    g = grid.set_index("time")
    tb = b["tb"].reindex(g.index)
    tq = b["tq"].reindex(g.index)
    qv = b["qv"].reindex(g.index)
    ct = b["count"].reindex(g.index)
    base_vol = b["volume"].reindex(g.index)
    have_tb = tb.notna() & tq.notna() & qv.notna()
    grid["of_known"] = have_tb.astype(int).values
    cvd = (2 * tb - base_vol).where(have_tb, 0.0)
    grid["cvd_ratio"] = (cvd / base_vol.replace(0, np.nan)).fillna(0.0).values
    grid["cvd_sum4"] = cvd.rolling(4).sum().fillna(0.0).values
    grid["cvd_sum8"] = cvd.rolling(8).sum().fillna(0.0).values
    grid["buyer_ratio"] = (tq / qv.replace(0, np.nan)).where(have_tb, 0.5).fillna(0.5).values
    grid["trade_vel"] = (ct / ct.rolling(96, min_periods=24).mean().replace(0, np.nan)).fillna(1.0).values
    return grid


def _fit_pca(rets: pd.DataFrame) -> dict | None:
    """PCA(2) on 4-asset 15m returns: PC1 = market momentum, PC2 = idiosyncratic."""
    from sklearn.decomposition import PCA
    r = rets.dropna()
    if len(r) < 500 or r.shape[1] < 2:
        return None
    cut = int(len(r) * 0.70)
    pca = PCA(n_components=2, random_state=7).fit(r.iloc[:cut].values)
    return {"pca": pca, "cols": list(r.columns)}


def _apply_pca(grid: pd.DataFrame, rets: pd.DataFrame, fit: dict | None) -> pd.DataFrame:
    grid = grid.copy()
    grid["mkt_mom"] = 0.0
    grid["idio_div"] = 0.0
    grid["pca_known"] = 0
    if fit is None:
        return grid
    r = rets.reindex(grid.set_index("time").index)
    ok = r.notna().all(axis=1)
    if ok.sum():
        z = fit["pca"].transform(r.loc[ok, fit["cols"]].values)
        grid.loc[ok.values, "mkt_mom"] = z[:, 0]
        grid.loc[ok.values, "idio_div"] = z[:, 1] if z.shape[1] > 1 else 0.0
        grid.loc[ok.values, "pca_known"] = 1
    return grid


def _flow_features(grid: pd.DataFrame, asset: str) -> pd.DataFrame:
    """Hyperliquid funding + Coinbase premium. Pre-history -> 0 + known flag."""
    from fetch_data import DATA_DIR
    grid = grid.copy()
    grid["fund_known"] = 0
    grid["fund_rate"] = 0.0
    grid["fund_z"] = 0.0
    try:
        f = pd.read_csv(DATA_DIR / f"flow_{asset.lower()}_funding.csv", parse_dates=["time"])
        f["time"] = pd.to_datetime(f["time"], utc=True)
        f = f.sort_values("time")
        mu = f["funding"].rolling(168, min_periods=24).mean()
        sd = f["funding"].rolling(168, min_periods=24).std().replace(0, np.nan)
        f["z"] = (f["funding"] - mu) / sd
        g = grid.set_index("time")
        idx = f.set_index("time")
        grid["fund_rate"] = idx["funding"].reindex(g.index, method="ffill", limit=8).fillna(0.0).values
        grid["fund_z"] = idx["z"].reindex(g.index, method="ffill", limit=8).fillna(0.0).values
        grid["fund_known"] = idx["funding"].reindex(g.index, method="ffill", limit=8).notna().astype(int).values
    except Exception:
        pass
    # second venue (Bybit HL-spread): helps BTC (69% backtest), hurts ETH
    # (50% vs 56% without) -> BTC only. Missing -> neutral.
    grid["venue_known"] = 0
    for _c in ("bb_rate", "bb_z", "fund_spread", "spread_z"):
        grid[_c] = 0.0
    if asset in ("BTC",):
        try:
            _bb = pd.read_csv(DATA_DIR / f"flow_{asset.lower()}_bb.csv", parse_dates=["time"])
            _bb["time"] = pd.to_datetime(_bb["time"], utc=True)
            _bb = _bb.sort_values("time")
            _mu = _bb["funding"].rolling(56, min_periods=8).mean()  # 8h prints ~7d
            _sd = _bb["funding"].rolling(56, min_periods=8).std().replace(0, np.nan)
            _bb["z"] = (_bb["funding"] - _mu) / _sd
            _bi = _bb.set_index("time")
            _bbr = _bi["funding"].reindex(g.index, method="ffill", limit=8)
            _bbz = _bi["z"].reindex(g.index, method="ffill", limit=8)
            _fr = idx["funding"].reindex(g.index, method="ffill", limit=8)
            _sp = _fr - _bbr
            _spz = (_sp - _sp.rolling(672, min_periods=48).mean()) / _sp.rolling(672, min_periods=48).std().replace(0, np.nan)
            grid["bb_rate"] = _bbr.fillna(0.0).values
            grid["bb_z"] = _bbz.fillna(0.0).values
            grid["fund_spread"] = _sp.fillna(0.0).values
            grid["spread_z"] = _spz.fillna(0.0).values
            grid["venue_known"] = _bbr.notna().astype(int).values
        except Exception:
            pass
    if asset == "BTC":
        grid["prem_known"] = 0
        grid["prem_bps"] = 0.0
        grid["prem_chg"] = 0.0
        try:
            cb = pd.read_csv(DATA_DIR / "premium_btc_15m.csv", parse_dates=["time"])
            cb["time"] = pd.to_datetime(cb["time"], utc=True)
            m = pd.merge_asof(grid.sort_values("time"), cb.sort_values("time")[["time", "close"]].rename(columns={"close": "cb"}),
                              on="time", direction="backward", tolerance=pd.Timedelta("20min"))
            prem = (m["cb"] - m["close"]) / m["close"] * 1e4
            grid["prem_bps"] = prem.fillna(0.0).values
            grid["prem_chg"] = prem.fillna(0.0).diff(4).fillna(0.0).values
            grid["prem_known"] = m["cb"].notna().astype(int).values
        except Exception:
            pass
    return grid


FLOW_FEATS = {"BTC": ["fund_rate", "fund_z", "fund_known", "prem_bps", "prem_chg", "prem_known",
                         "bb_rate", "bb_z", "fund_spread", "spread_z", "venue_known"],
              "ETH": ["fund_rate", "fund_z", "fund_known",
                      "bb_rate", "bb_z", "fund_spread", "spread_z", "venue_known"],
              "SOL": ["fund_rate", "fund_z", "fund_known"],
              "XRP": ["fund_rate", "fund_z", "fund_known"]}


def time_splits(n: int, fracs=(0.70, 0.775, 0.85), times=None,
                 purge_td: str = "3D", embargo_td: str = "5D") -> tuple:
    """Time-ordered (train, calibA, calibB, test) index arrays.

    With times: purge rows within purge_td before each eval cut (longest
    lookback can't reach across) + embargo_td gap after calibB (synoptic/
    volatility persistence must not leak into test).
    """
    a, b, c = (int(n * f) for f in fracs)
    idx = np.arange(n)
    tr, cA, cB, te = idx[:a], idx[a:b], idx[b:c], idx[c:]
    if times is not None:
        t = pd.to_datetime(pd.Series(times)).reset_index(drop=True)
        tr = tr[t.iloc[tr] <= t.iloc[cA].min() - pd.Timedelta(purge_td)]
        te = te[t.iloc[te] >= t.iloc[cB].max() + pd.Timedelta(embargo_td)]
    return tr, cA, cB, te


def latest_row(btc15: pd.DataFrame, btc1: pd.DataFrame,
               bars15: dict, bars1: dict, feats: list[str], leaders: tuple = LEADERS,
               asset: str = "BTC", pcafit: dict | None = None) -> pd.Series:
    """Fast single-row feature build for live inference (same logic, last grid time)."""
    for df in [btc15, btc1, *bars15.values(), *bars1.values()]:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    base = add_features(btc15)
    row = base.iloc[[-1]].copy().reset_index(drop=True)
    T = row["time"].iloc[0]

    b1 = btc1.sort_values("time")
    blk = b1[(b1["time"] > T - pd.Timedelta(minutes=16)) & (b1["time"] <= T)].tail(15)
    s = _m1_block_stats(blk)
    for k in M_KEYS:  # never emit a partial key set (would KeyError downstream)
        row[f"m_{k}"] = s.get(k, np.nan)
    hist_vol = b1[b1["time"] <= T].tail(96)["volume"]
    row["m_volspike"] = (blk["volume"].sum() / hist_vol.median()) if hist_vol.median() else np.nan

    btc_close = float(row["close"].iloc[0])
    for a in leaders:
        d15 = bars15[a].sort_values("time")
        past = d15[d15["time"] <= T]
        if len(past) < 30:
            continue
        last = past.iloc[-1]
        c = past["close"]
        row[f"x_{a.lower()}_ret1"] = float(np.log(last["close"] / c.iloc[-2]))
        row[f"x_{a.lower()}_ret4"] = float(np.log(last["close"] / c.iloc[-5]))
        row[f"x_{a.lower()}_dist8"] = float(last["close"] / c.ewm(span=8, adjust=False).mean().iloc[-1] - 1)
        row[f"x_{a.lower()}_vold"] = float(last["volume"] / past["volume"].tail(24).mean())
        m1 = bars1[a].sort_values("time")
        ab = m1[(m1["time"] > T - pd.Timedelta(minutes=15)) & (m1["time"] <= T)]
        apath = float(np.log(ab["close"].iloc[-1] / ab["close"].iloc[0])) if len(ab) >= 12 else np.nan
        row[f"x_{a.lower()}_path15"] = apath
        row[f"x_{a.lower()}_relstr"] = apath - s.get("path", 0.0)
        w = m1[(m1["time"] > T - pd.Timedelta(minutes=20)) & (m1["time"] <= T - pd.Timedelta(minutes=5))]
        row[f"x_{a.lower()}_lead5"] = float(np.log(w["close"].iloc[-1] / w["close"].iloc[0])) if len(w) >= 10 else np.nan
    for a in leaders:  # same neutral-0 + known-flag convention as training
        al = a.lower()
        r0 = row.iloc[0]
        row[f"x_{al}_mknown"] = 0 if pd.isna(r0[f"x_{al}_path15"]) else 1
        for col in (f"x_{al}_path15", f"x_{al}_relstr", f"x_{al}_lead5"):
            if pd.isna(r0[col]):
                row[col] = 0.0
    row = _flow_features(row, asset)
    try:
        row = _add_orderflow(row, btc15)
    except Exception:
        for _c in ("cvd_ratio", "cvd_sum4", "cvd_sum8", "buyer_ratio",
                   "trade_vel", "of_known"):
            if _c not in row.columns:
                row[_c] = {"buyer_ratio": 0.5, "trade_vel": 1.0}.get(_c, 0.0)
    try:
        row = _apply_pca(row, _universe_rets(btc15, bars15, asset), pcafit)
    except Exception:
        for _c in ("mkt_mom", "idio_div", "pca_known"):
            if _c not in row.columns:
                row[_c] = 0.0 if _c != "pca_known" else 0
    return row.iloc[0]


if __name__ == "__main__":
    from fetch_data import load_or_fetch
    btc15 = load_or_fetch("BTC", "15m", days=150)
    btc1 = load_or_fetch("BTC", "1m", days=120)
    b15 = {a: load_or_fetch(a, "15m", days=150) for a in LEADERS}
    b1 = {a: load_or_fetch(a, "1m", days=120) for a in LEADERS}
    df, feats, _ = build_dataset(btc15, btc1, b15, b1)
    print(len(df), "rows x", len(feats), "features,", df['time'].min(), "->", df['time'].max())
    print("base up-rate:", df["y_up"].mean().round(4))
