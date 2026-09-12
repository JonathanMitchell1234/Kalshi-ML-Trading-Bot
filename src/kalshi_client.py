"""Kalshi public REST client (real markets) + authenticated client scaffolding.

Public endpoints need no key: events, markets, market snapshots.
Authenticated endpoints (balance / orders) use RSA-PSS signing per Kalshi docs.
Paper trading is the default — no real orders are placed by this codebase.
"""
import os
import time
import base64
import requests
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

load_dotenv()

BASE = os.getenv("KALSHI_API_BASE", "https://api.elections.kalshi.com/trade-api/v2").rstrip("/")
SERIES = "KXBTC"      # hourly BTC range brackets
SERIES15 = "KXBTC15M"  # 15-min BTC up/down binaries (YES = expire >= target)
SERIES15_MAP = {"BTC": "KXBTC15M", "ETH": "KXETH15M", "SOL": "KXSOL15M", "XRP": "KXXRP15M"}


def _et_now() -> datetime:
    """Current time in America/New_York (Kalshi crypto hours)."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        # Fallback: EDT (UTC-4). Wrong by 1h during EST — verify via API instead.
        return datetime.now(timezone.utc).astimezone() - timedelta(hours=0) \
            if False else (datetime.now(timezone.utc) - timedelta(hours=4))


def event_ticker_for(dt_et: datetime) -> str:
    """KXBTC-YYMONDDHH where HH = closing hour ET (e.g. KXBTC-26SEP0421)."""
    return f"{SERIES}-{dt_et.strftime('%y').upper()}{dt_et.strftime('%b').upper()}{dt_et.day:02d}{dt_et.hour:02d}"


def current_event_ticker() -> str:
    """Guess the live event: next ET hour close, verified against the API."""
    now_et = _et_now()
    for delta_h in (1, 0, 2):
        guess = now_et.replace(minute=0, second=0, microsecond=0) + timedelta(hours=delta_h)
        t = event_ticker_for(guess)
        try:
            ev = get_event(t)
            statuses = {m.get("status") for m in ev.get("markets", [])}
            if "active" in statuses:
                return t
        except Exception:
            continue
    # last resort: return the +1h guess and let callers handle it
    return event_ticker_for(now_et.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))


def _get(path: str, params: dict | None = None, auth_headers: dict | None = None) -> dict:
    r = requests.get(BASE + path, params=params or {}, headers=auth_headers or {}, timeout=25)
    r.raise_for_status()
    return r.json()


def get_event(event_ticker: str) -> dict:
    return _get(f"/events/{event_ticker}")


def get_market(ticker: str) -> dict:
    return _get(f"/markets/{ticker}").get("market", {})


def _cents(x) -> int | None:
    if x is None:
        return None
    try:
        return int(round(float(x) * 100))
    except (TypeError, ValueError):
        return None


def parse_bracket(m: dict) -> dict:
    """Normalize one Kalshi bracket market into plain ints (cents)."""
    return {
        "ticker": m.get("ticker"),
        "subtitle": m.get("subtitle"),
        "strike_type": m.get("strike_type"),  # less | between | greater
        "floor": m.get("floor_strike"),
        "cap": m.get("cap_strike"),
        "status": m.get("status"),
        "result": m.get("result") or None,
        "yes_bid": _cents(m.get("yes_bid_dollars")),
        "yes_ask": _cents(m.get("yes_ask_dollars")),
        "last": _cents(m.get("last_price_dollars")),
        "volume_24h": m.get("volume_24h_fp"),
        "open_interest": m.get("open_interest_fp"),
        "close_time": m.get("close_time"),
        "expiration_value": m.get("expiration_value"),
        "expiration_ts": m.get("expiration_time"),
    }


def get_brackets(event_ticker: str, near_floor: float | None = None, window: int = 1500) -> tuple[dict, list[dict]]:
    """Event meta + brackets, optionally filtered to ±window of a price level."""
    ev = get_event(event_ticker)
    meta = ev.get("event", {}) or {}
    out = [parse_bracket(m) for m in ev.get("markets", [])]
    if near_floor is not None:
        out = [b for b in out
               if b["floor"] is not None and abs(b["floor"] - near_floor) <= window
               or b["strike_type"] in ("less", "greater")]
    # sort by floor (tails first/last)
    out.sort(key=lambda b: (b["floor"] is None, b["floor"] or 0))
    return meta, out


def list_events(limit_pages: int = 5) -> list[dict]:
    """Recent KXBTC events (newest first by ticker)."""
    evs, cursor = [], None
    for _ in range(limit_pages):
        p = {"series_ticker": SERIES, "limit": 200}
        if cursor:
            p["cursor"] = cursor
        j = _get("/events", p)
        evs += j.get("events", [])
        cursor = j.get("cursor")
        if not cursor:
            break
    return evs


# ------------------------------------------------- 15-min up/down binaries
def m15_ticker_for(dt_et: datetime, asset: str = "BTC") -> str:
    """{SERIES}-YYMONDDHHMM in ET (suffix = window CLOSE)."""
    series = SERIES15_MAP.get(asset, SERIES15)
    return f"{series}-{dt_et.strftime('%y').upper()}{dt_et.strftime('%b').upper()}{dt_et.day:02d}{dt_et.hour:02d}{dt_et.minute:02d}"


def _et_floor15(now_et: datetime) -> datetime:
    return now_et.replace(second=0, microsecond=0).replace(minute=(now_et.minute // 15) * 15)


class MarketAbsent(Exception):
    """The expected event ticker doesn't exist (Kalshi skips low-liquidity
    slots overnight). Not a transport error — callers should skip quietly."""


_live_cache: dict = {}


def current_15m_ticker(asset: str = "BTC") -> str:
    """Live 15-min event: nearest slot at/before now+15m with an ACTIVE market.

    Kalshi skips some overnight slots entirely (404), so scan a 2h window
    instead of trusting one guess. Cached 4 min to spare the API.
    Raises MarketAbsent when nothing is tradeable right now.
    """
    import time as _t
    from datetime import timedelta as _td
    key = (asset, int(_t.time() // 240))
    if key in _live_cache:
        return _live_cache[key]
    base_close = _et_floor15(_et_now())
    tried = []
    for back_min in (15, 30, 0, 45, 60, 75, 90, 105, 120, -15):
        t = m15_ticker_for(base_close - _td(minutes=back_min - 15), asset)
        tried.append(t)
        try:
            mk = get_event(t).get("markets", [])
        except Exception:
            continue  # transport blip: keep scanning, don't conclude
        if not mk:
            continue  # 404/empty: slot was never listed
        if any(m.get("status") == "active" for m in mk):
            _live_cache[key] = t
            return t
    raise MarketAbsent(f"no active {asset} 15m event near {base_close} (tried {tried[-1]}..{tried[0]})")


def parse_updown(m: dict, event_ticker: str = "") -> dict:
    no_ask = _cents(m.get("no_ask_dollars"))
    yes_bid = _cents(m.get("yes_bid_dollars"))
    if no_ask is None and yes_bid is not None:
        no_ask = 100 - yes_bid
    def _sz(x):
        try:
            return int(float(x))  # fp units track contracts at our sizes; documented assumption
        except (TypeError, ValueError):
            return None  # lift the resting NO bid
    return {
        "event": event_ticker,
        "ticker": m.get("ticker"),
        "target": m.get("floor_strike"),
        "status": m.get("status"),
        "result": m.get("result") or None,
        "yes_bid": yes_bid,
        "yes_ask": _cents(m.get("yes_ask_dollars")),
        "no_bid": _cents(m.get("no_bid_dollars")),
        "no_ask": no_ask,
        "yes_bid_size": _sz(m.get("yes_bid_size_fp")),
        "yes_ask_size": _sz(m.get("yes_ask_size_fp")),
        "no_bid_size": _sz(m.get("no_bid_size_fp")),
        "no_ask_size": _sz(m.get("no_ask_size_fp")),
        "close_time": m.get("close_time"),
        "open_time": m.get("open_time"),
        "expiration_value": m.get("expiration_value"),
    }


def get_15m(event_ticker: str) -> dict:
    """The single up/down market of a 15-min event, parsed."""
    ev = get_event(event_ticker)
    mk = ev.get("markets", [])
    if not mk:
        raise RuntimeError(f"no markets in {event_ticker}")
    d = parse_updown(mk[0], event_ticker)
    d["title"] = (ev.get("event") or {}).get("title")
    return d


def recent_15m_tickers(n: int, asset: str = "BTC") -> list[str]:
    """Recently CLOSED 15-min tickers in ET, newest first (for backtests)."""
    from datetime import timedelta as _td
    floor = _et_floor15(_et_now())
    return [m15_ticker_for(floor - _td(minutes=15 * (i - 1)), asset) for i in range(1, n + 1)]


# ---------------------------------------------------------------- authenticated
class KalshiAuth:
    """RSA-PSS request signer. Used for balance / (future) real orders only."""

    def __init__(self, key_id: str | None = None, key_path: str | None = None):
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding
        self._hashes, self._serialization, self._padding = hashes, serialization, padding
        self.key_id = key_id or os.getenv("KALSHI_API_KEY_ID", "")
        self.key_path = key_path or os.getenv("KALSHI_API_PRIVATE_KEY_PATH", "")
        self._key = None

    def _load(self):
        if self._key is None:
            data = Path(self.key_path).expanduser().read_bytes()
            self._key = self._serialization.load_pem_private_key(data, password=None)
        return self._key

    def headers(self, method: str, path: str) -> dict:
        ts = str(int(time.time() * 1000))
        msg = f"{ts}{method}{path}".encode()
        sig = self._load().sign(
            msg,
            self._padding.PSS(mgf=self._padding.MGF1(self._hashes.SHA256()),
                              salt_length=self._padding.PSS.DIGEST_LENGTH),
            self._hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
        }

    def balance(self) -> dict:
        h = self.headers("GET", "/trade-api/v2/portfolio/balance")
        return _get("/portfolio/balance", auth_headers=h).get("balance", {})


if __name__ == "__main__":
    t = current_event_ticker()
    print("live event:", t)
    meta, brackets = get_brackets(t, near_floor=80000, window=500)
    print("brackets:", len(brackets))
    for b in brackets:
        if b["floor"] and 79400 <= b["floor"] <= 80000:
            print(f"  {b['ticker']} {b['subtitle']} bid={b['yes_bid']} ask={b['yes_ask']} last={b['last']}")
