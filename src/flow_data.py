"""Order-flow features: Hyperliquid funding + Coinbase premium (both free).

Funding (hourly, perps): rate level + z-score vs trailing 7d. Persistent
positive funding = crowded longs (contrarian short-horizon signal).
Premium: Coinbase BTC-USD minus Binance.US BTCUSD, 15m closes (level in bps
+ 1h change). Coinbase premium leads short moves in literature.
Cache: data_cache/flow_btc_funding.csv, flow_eth_funding.csv, premium_btc_15m.csv
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

HL = "https://api.hyperliquid.xyz/info"


def dydx_funding(market: str = "BTC-USD", days: int = 365) -> pd.DataFrame:
    """dYdX v4 hourly funding history (public indexer, no key)."""
    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(days=days)
    rows, cursor = [], None
    while True:
        p = {"limit": 100}
        if cursor:
            p["createdBeforeOrAt"] = cursor
        r = requests.get(f"https://indexer.dydx.trade/v4/historicalFunding/{market}",
                         params=p, timeout=30)
        r.raise_for_status()
        chunk = r.json().get("historicalFunding", [])
        if not chunk:
            break
        rows += chunk
        oldest = pd.Timestamp(chunk[-1]["effectiveAt"])
        cursor = chunk[-1]["effectiveAt"]
        if oldest <= start or len(chunk) < 100:
            break
        time.sleep(0.3)
        if len(rows) > 12000:
            break
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["effectiveAt"], utc=True)
    df["funding"] = df["rate"].astype(float)
    df = df[df["time"] >= start].drop_duplicates("time").sort_values("time").reset_index(drop=True)
    return df[["time", "funding"]]


def bybit_funding(symbol: str = "BTCUSDT", days: int = 365) -> pd.DataFrame:
    """Bybit linear-perp funding history (8h prints). Requires non-US route."""
    import datetime as _dt
    end = int(time.time() * 1000)
    start = end - days * 24 * 3600 * 1000
    rows, cursor = [], end
    while cursor > start:
        r = requests.get("https://api.bybit.com/v5/market/funding/history",
                         params={"category": "linear", "symbol": symbol, "limit": 200,
                                 "endTime": cursor},
                         timeout=30)
        r.raise_for_status()
        j = r.json()
        if j.get("retCode") != 0:
            raise RuntimeError(f"bybit: {j.get('retMsg')}")
        chunk = j["result"]["list"]  # newest-first
        if not chunk:
            break
        rows += [c for c in chunk if int(c["fundingRateTimestamp"]) >= start]
        cursor = int(chunk[-1]["fundingRateTimestamp"]) - 1
        if len(chunk) < 200:
            break
        time.sleep(0.3)
        if len(rows) > 3000:
            break
    df = pd.DataFrame(rows).drop_duplicates("fundingRateTimestamp")
    df["time"] = pd.to_datetime(df["fundingRateTimestamp"].astype(int), unit="ms", utc=True)
    df["funding"] = df["fundingRate"].astype(float)
    return df[["time", "funding"]].sort_values("time").reset_index(drop=True)


def bybit_oi(symbol: str = "BTCUSDT", days: int = 180) -> pd.DataFrame:
    """Bybit hourly open interest (USDT). Requires non-US route."""
    end = int(time.time() * 1000)
    start = end - days * 24 * 3600 * 1000
    rows, cur, cursor = [], start, ""
    while True:
        p = {"category": "linear", "symbol": symbol, "intervalTime": "60",
             "limit": 200, "startTime": max(cur, start), "endTime": end}
        if cursor:
            p["cursor"] = cursor
        r = requests.get("https://api.bybit.com/v5/market/open-interest", params=p, timeout=30)
        r.raise_for_status()
        j = r.json()
        if j.get("retCode") != 0:
            raise RuntimeError(f"bybit OI: {j.get('retMsg')}")
        chunk = j["result"]["list"]
        if not chunk:
            break
        rows += chunk
        cursor = j["result"].get("nextPageCursor", "")
        cur = int(chunk[-1]["timestamp"]) + 1
        if not cursor or cur >= end:
            break
        time.sleep(0.3)
        if len(rows) > 5000:
            break
    df = pd.DataFrame(rows).drop_duplicates("timestamp")
    df["time"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True)
    df["oi"] = df["openInterest"].astype(float)
    return df[["time", "oi"]].sort_values("time").reset_index(drop=True)


def hl_funding(coin: str, days: int = 365) -> pd.DataFrame:
    now = int(time.time() * 1000)
    start = now - days * 24 * 3600 * 1000
    rows = []
    cur = start
    while cur < now:
        r = requests.post(HL, json={"type": "fundingHistory", "coin": coin,
                                    "startTime": cur, "endTime": min(cur + 30 * 24 * 3600 * 1000, now)},
                          timeout=30)
        r.raise_for_status()
        chunk = r.json()
        if not chunk:
            break
        rows += chunk
        cur = chunk[-1]["time"] + 1
        if len(chunk) < 2:
            break
        time.sleep(0.3)
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df["funding"] = df["fundingRate"].astype(float)
    return df[["time", "funding"]].drop_duplicates("time").sort_values("time").reset_index(drop=True)


def coinbase_15m(days: int = 365) -> pd.DataFrame:
    from datetime import datetime, timezone, timedelta
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    frames = []
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=3), end)  # <=300 candles per call
        r = requests.get("https://api.exchange.coinbase.com/products/BTC-USD/candles",
                         params={"granularity": 900,
                                 "start": cur.replace(microsecond=0).isoformat(),
                                 "end": nxt.replace(microsecond=0).isoformat()}, timeout=30)
        r.raise_for_status()
        chunk = r.json()
        if chunk:
            f = pd.DataFrame(chunk, columns=["time", "low", "high", "open", "close", "vol"])
            frames.append(f)
        cur = nxt
        time.sleep(0.4)
    df = pd.concat(frames, ignore_index=True).drop_duplicates("time").sort_values("time")
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df.reset_index(drop=True)


def _save(df: pd.DataFrame, name: str):
    p = DATA_DIR / name
    tmp = p.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    os.replace(tmp, p)
    print(f"saved {name}: {len(df)} rows {df['time'].min()} -> {df['time'].max()}")


if __name__ == "__main__":
    _save(hl_funding("BTC"), "flow_btc_funding.csv")
    _save(hl_funding("ETH"), "flow_eth_funding.csv")
    _save(coinbase_15m(), "premium_btc_15m.csv")
