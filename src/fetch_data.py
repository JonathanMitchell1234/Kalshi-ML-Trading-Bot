"""Multi-asset klines from Binance.US (free, no key, full history).

Assets: BTC, ETH, SOL, XRP. Intervals: 1m, 15m.
Cache: data_cache/{asset}_{interval}.csv  (e.g. btc_1m.csv).
Binance.US symbols are USD-quoted (BTCUSD...); fallback to USDT.

Backwards compatible: fetch_history(days) / load_or_fetch(days) still mean BTC 15m.
"""
import os
import time
import fcntl
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
_DD = Path(os.getenv("DATA_DIR", str(ROOT / "data_cache")))
DATA_DIR = _DD if _DD.is_absolute() else ROOT / _DD
DATA_DIR.mkdir(parents=True, exist_ok=True)

BASE = "https://api.binance.us"
INTERVALS = {"1m": 60 * 1000, "15m": 15 * 60 * 1000}
LIMIT = 1000
ASSETS = ("BTC", "ETH", "SOL", "XRP")


def _symbols(asset: str) -> list[str]:
    a = asset.upper()
    return [f"{a}USD", f"{a}USDT"]


def _pick_symbol(asset: str) -> str:
    for s in _symbols(asset):
        try:
            r = requests.get(f"{BASE}/api/v3/klines",
                             params={"symbol": s, "interval": "15m", "limit": 1}, timeout=15)
            if r.status_code == 200:
                return s
        except Exception:
            continue
    raise RuntimeError(f"Binance.US unreachable for {asset}")


def cache_path(asset: str = "BTC", interval: str = "15m") -> Path:
    return DATA_DIR / f"{asset.lower()}_{interval}.csv"


def fetch_chunk(symbol: str, interval: str, start_ms: int, retries: int = 4) -> pd.DataFrame:
    last = None
    for i in range(retries):
        try:
            r = requests.get(f"{BASE}/api/v3/klines",
                             params={"symbol": symbol, "interval": interval,
                                     "startTime": start_ms, "limit": LIMIT}, timeout=30)
            if r.status_code in (429,) + tuple(range(500, 600)):
                last = f"HTTP {r.status_code}"
                time.sleep(2 * (i + 1))
                continue
            r.raise_for_status()
            rows = r.json()
            break
        except requests.RequestException as e:
            last = str(e)[:100]
            time.sleep(2 * (i + 1))
    else:
        raise RuntimeError(f"klines failed after {retries} tries ({symbol} {interval}): {last}")
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume",
                                     "close_time", "qav", "trades", "taker_base", "taker_quote", "ignore"])
    df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df["count"] = df["trades"].astype(int)
    df["vwap"] = df["close"]
    # order-flow fields (review: CVD/buyer-ratio need these — keep them)
    df["tb"] = df["taker_base"].astype(float)    # taker BUY base volume
    df["tq"] = df["taker_quote"].astype(float)   # taker buy quote volume
    df["qv"] = df["qav"].astype(float)           # total quote volume
    return df[["time", "open", "high", "low", "close", "vwap", "volume", "count",
               "tb", "tq", "qv"]]


def fetch_history(symbol: str | None = None, interval: str = "15m", days: int = 365,
                 sleep: float = 0.4, verbose: bool = True, asset: str = "BTC") -> pd.DataFrame:
    symbol = symbol or _pick_symbol(asset)
    step = INTERVALS[interval]
    start_ms = int(time.time() * 1000) - days * 24 * 3600 * 1000
    now_ms = int(time.time() * 1000)
    frames = []
    while start_ms < now_ms - step:
        df = fetch_chunk(symbol, interval, start_ms)
        if df.empty:
            break
        frames.append(df)
        last_open = int(pd.Timestamp(df["time"].iloc[-1]).timestamp() * 1000)
        if verbose and len(frames) % 10 == 0:
            print(f"  ...{df['time'].iloc[-1]} (total {sum(map(len, frames))})")
        if last_open <= start_ms - step:
            break
        start_ms = last_open + step
        time.sleep(sleep)
        if len(frames) > 400:
            break
    full = pd.concat(frames, ignore_index=True).drop_duplicates("time").sort_values("time").reset_index(drop=True)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    full = full[full["time"] >= cutoff].reset_index(drop=True)
    full = full[full["time"] <= pd.Timestamp.now(tz="UTC").floor("15min") - pd.Timedelta(minutes=15)].reset_index(drop=True)
    return full


def _atomic_save(df: pd.DataFrame, path: Path):
    tmp = path.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def fetch_latest(asset: str = "BTC", interval: str = "15m", n: int = 10,
                 symbol: str | None = None) -> pd.DataFrame:
    """Last n CLOSED candles. Single API request — cheap enough to call often."""
    symbol = symbol or _pick_symbol(asset)
    r = requests.get(f"{BASE}/api/v3/klines",
                     params={"symbol": symbol, "interval": interval, "limit": n + 2},
                     timeout=20)
    r.raise_for_status()
    rows = r.json()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume",
                                     "close_time", "qav", "trades", "taker_base", "taker_quote", "ignore"])
    df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df["count"] = df["trades"].astype(int)
    df["vwap"] = df["close"]
    df = df[["time", "open", "high", "low", "close", "vwap", "volume", "count"]]
    # drop the still-forming candle
    step_ms = INTERVALS[interval]
    df = df[df["time"] <= pd.Timestamp.now(tz="UTC").floor(interval.replace("m", "min")) - pd.Timedelta(milliseconds=step_ms)]
    return df.reset_index(drop=True)


def refresh_all() -> dict:
    """Refresh tails of every dataset the signal stack reads. ~8 requests."""
    out = {}
    for asset, interval in [("BTC", "15m"), ("BTC", "1m"), ("ETH", "15m"), ("ETH", "1m"),
                            ("SOL", "15m"), ("SOL", "1m"), ("XRP", "15m"), ("XRP", "1m")]:
        try:
            out[f"{asset}_{interval}"] = refresh_tail(asset, interval)
        except Exception as e:
            out[f"{asset}_{interval}"] = f"ERR {e}"[:80]
    return out


def refresh_tail(asset: str = "BTC", interval: str = "15m") -> int:
    """Merge latest closed candles into cache (locked, atomic). Returns # added.

    Also heals gaps: any hole >3 min in the last 24h is refetched (capped),
    so thin-minute pairs converge to complete history over time.
    """
    cp = cache_path(asset, interval)
    lock_path = cp.with_suffix(".lock")
    with open(lock_path, "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX)
        except Exception:
            pass
        try:
            tail = fetch_latest(asset, interval)
            if tail.empty:
                return 0
            if cp.exists():
                df = pd.read_csv(cp, parse_dates=["time"])
                before = len(df)
                df = pd.concat([df, tail], ignore_index=True)
            else:
                before = 0
                df = tail
            df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
            df = df.dropna(subset=["time"]).drop_duplicates("time").sort_values("time").reset_index(drop=True)
            added = len(df) - before
            # heal recent holes (thin minutes): refetch gaps >3min in last 24h
            try:
                if interval == "1m":
                    day = df[df["time"] >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=26)]
                    if len(day) > 10:
                        gaps = day["time"].diff().dt.total_seconds()
                        holes = day[gaps > 210].index.tolist()[:8]
                        for ix in holes:
                            start = int(day["time"].iloc[ix - 1].timestamp() * 1000) if ix > 0 else None
                            if start is None:
                                continue
                            fill = fetch_chunk(_pick_symbol(asset), interval, start)
                            if not fill.empty:
                                df = pd.concat([df, fill], ignore_index=True)
                        if holes:
                            df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
                            df = df.dropna(subset=["time"]).drop_duplicates("time").sort_values("time").reset_index(drop=True)
                            added = len(df) - before
            except Exception:
                pass
            _atomic_save(df, cp)
            return added
        finally:
            try:
                fcntl.flock(lock, fcntl.LOCK_UN)
            except Exception:
                pass


def load_or_fetch(asset: str = "BTC", interval: str = "15m", days: int = 365,
                  force_refresh: bool = False, symbol: str | None = None,
                  min_refresh_s: int = 600) -> pd.DataFrame:
    """Full cache kept on disk; returns the requested window.

    Compat: load_or_fetch(days) / load_or_fetch(365) still loads BTC 15m.
    Concurrency-safe (fcntl lock + atomic replace). Tail refresh throttled by
    file mtime so dashboard loops don't hammer the API or the cache file.
    Corrupt time rows are coerced away instead of crashing callers.
    """
    if isinstance(asset, int):  # load_or_fetch(365)
        days = asset
        asset = "BTC"
    cp = cache_path(asset, interval)
    lock_path = cp.with_suffix(".lock")
    with open(lock_path, "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX)
        except Exception:
            pass
        try:
            return _load_locked(asset, interval, days, force_refresh, symbol, cp, min_refresh_s)
        finally:
            try:
                fcntl.flock(lock, fcntl.LOCK_UN)
            except Exception:
                pass


def _load_locked(asset, interval, days, force_refresh, symbol, cp, min_refresh_s):
    import time as _t
    if cp.exists() and not force_refresh:
        df = pd.read_csv(cp, parse_dates=["time"])
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.dropna(subset=["time"]).drop_duplicates("time").sort_values("time").reset_index(drop=True)
        fresh = (_t.time() - cp.stat().st_mtime) < min_refresh_s
        if not fresh:
            try:
                tail = fetch_history(symbol or _pick_symbol(asset), interval,
                                     days=2 if interval == "1m" else 8,
                                     verbose=False, asset=asset)
                df = pd.concat([df, tail], ignore_index=True)
                df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
                df = df.dropna(subset=["time"]).drop_duplicates("time").sort_values("time").reset_index(drop=True)
                _atomic_save(df, cp)
            except Exception as e:
                print(f"Tail refresh failed ({e}), using cache.")
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
        view = df[df["time"] >= cutoff].reset_index(drop=True)
        print(f"Loaded {asset} {interval}: {len(view)} rows [cache {len(df)}]")
        return view
    print(f"Fetching {asset} {interval} ~{days}d from Binance.US...")
    df = fetch_history(symbol or _pick_symbol(asset), interval, days=days, asset=asset)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).drop_duplicates("time").sort_values("time").reset_index(drop=True)
    _atomic_save(df, cp)
    print(f"Saved {len(df)} rows to {cp}")
    return df


# ---- backwards-compat wrappers (BTC 15m) ----
CACHE_CSV = cache_path("BTC", "15m")


def _compat_history(days=365, sleep=0.3, verbose=True):
    return fetch_history(_pick_symbol("BTC"), "15m", days=days, sleep=sleep, verbose=verbose, asset="BTC")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="BTC")
    ap.add_argument("--interval", default="15m")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--all", action="store_true", help="fetch all assets (1m 120d + 15m 365d)")
    a = ap.parse_args()
    if a.all:
        for ass in ASSETS:
            load_or_fetch(ass, "1m", days=120, force_refresh=a.force)
        for ass in ASSETS:
            load_or_fetch(ass, "15m", days=365, force_refresh=a.force)
    else:
        df = load_or_fetch(a.asset, a.interval, days=a.days, force_refresh=a.force)
        print(df.tail(2).to_string())
