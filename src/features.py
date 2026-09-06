"""Feature engineering for BTC 15m bars. No TA-Lib dependency — pure pandas/numpy. No lookahead."""
import numpy as np
import pandas as pd

TARGET_HORIZON = 1  # predict close t+1 (15 min ahead)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    line = ema_f - ema_s
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def _atr(df: pd.DataFrame, period=14) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("time").reset_index(drop=True)
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]

    # --- returns / momentum ---
    for lag in [1, 2, 3, 4, 8, 12, 24, 48, 96]:
        df[f"ret_{lag}"] = close.pct_change(lag)
    df["logret_1"] = np.log(close / close.shift(1))
    for w in [4, 8, 24, 96]:
        df[f"mom_{w}"] = close - close.shift(w)
        df[f"roc_{w}"] = close.pct_change(w)

    # --- moving averages & position ---
    for w in [4, 8, 12, 24, 48, 96]:
        sma = close.rolling(w).mean()
        ema = close.ewm(span=w, adjust=False).mean()
        df[f"sma_{w}"] = sma
        df[f"ema_{w}"] = ema
        df[f"dist_sma_{w}"] = close / sma - 1
        df[f"dist_ema_{w}"] = close / ema - 1
    df["sma_8_24_ratio"] = df["sma_8"] / df["sma_24"] - 1
    df["ema_12_26_ratio"] = df["ema_12"] / df["ema_24"] - 1

    # --- volatility ---
    for w in [8, 24, 96]:
        df[f"vol_std_{w}"] = df["logret_1"].rolling(w).std()
        df[f"range_pct_{w}"] = ((high - low) / close).rolling(w).mean()
    df["atr_14"] = _atr(df) / close
    df["hl_pct"] = (high - low) / close
    df["oc_pct"] = (close - df["open"]) / df["open"]
    df["upper_wick"] = (high - close.clip(lower=df["open"])) / close
    df["lower_wick"] = (df["open"].clip(lower=close) - low) / close if False else (pd.concat([df["open"], close], axis=1).min(axis=1) - low) / close

    # --- RSI / MACD / Bollinger / Stochastic ---
    for p in [7, 14, 28]:
        df[f"rsi_{p}"] = _rsi(close, p)
    macd, sig, hist = _macd(close)
    df["macd"], df["macd_sig"], df["macd_hist"] = macd, sig, hist
    for w in [20]:
        sma = close.rolling(w).mean()
        std = close.rolling(w).std()
        df["bb_upper"] = (close - (sma + 2 * std)) / (4 * std.replace(0, np.nan))
        df["bb_width"] = (4 * std) / sma
        df["bb_pos"] = (close - (sma - 2 * std)) / (4 * std.replace(0, np.nan))
    k_w = 14
    ll, hh = low.rolling(k_w).min(), high.rolling(k_w).max()
    df["stoch_k"] = (close - ll) / (hh - ll).replace(0, np.nan)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # --- volume ---
    for w in [8, 24, 96]:
        df[f"vol_ratio_{w}"] = vol / vol.rolling(w).mean().replace(0, np.nan)
        df[f"vol_chg_{w}"] = vol.pct_change(w)
    df["dollar_vol"] = close * vol
    for w in [8, 24]:
        df[f"dollar_vol_ratio_{w}"] = df["dollar_vol"] / df["dollar_vol"].rolling(w).mean()

    # --- time features (UTC; Kalshi BTC settles on its own clock — UTC is the sane base) ---
    t = pd.to_datetime(df["time"], utc=True)
    df["hour"] = t.dt.hour
    df["dow"] = t.dt.dayofweek
    df["is_weekend"] = (t.dt.dayofweek >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["quarter_hour"] = t.dt.minute // 15
    df["overnight_us"] = ((t.dt.hour < 13) | (t.dt.hour >= 21)).astype(int)  # thin-liquidity proxy

    # --- targets (t+1 = 15 min ahead) ---
    df["target_close"] = close.shift(-TARGET_HORIZON)
    df["target_ret"] = df["target_close"] / close - 1
    df["target_logret"] = np.log(df["target_close"] / close)
    df["target_up"] = (df["target_close"] > close).astype(int)
    df["target_move_bps"] = df["target_ret"] * 1e4

    return df


FEATURE_BLACKLIST_PREFIXES = ("target_",)
NON_FEATURE_COLS = {"time"}


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c in NON_FEATURE_COLS:
            continue
        if c.startswith(FEATURE_BLACKLIST_PREFIXES):
            continue
        if c in ("open", "high", "low", "close", "vwap", "volume", "count",
                   "tb", "tq", "qv"):  # raw taker fields: ratios are feats, levels aren't
            continue  # raw levels leak scale; use derived features only
        cols.append(c)
    return cols
