"""Gate: static-Normal vs Garman-Klass + Student-t bracket density.

For settled KXBTC hourly bracket events: center = Binance spot 30 min before
close (tests the VOL/DIST choice, not the center). Compare mean winner-bracket
mass under legacy static-Normal vs dynamic GK + t(5).
"""
import math
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import kalshi_client as K
from fetch_data import load_or_fetch


def _parse(m: dict) -> dict:
    return {"ticker": m.get("ticker"), "strike_type": m.get("strike_type"),
            "floor": m.get("floor_strike"), "cap": m.get("cap_strike"),
            "status": m.get("status"), "result": m.get("result") or None}


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _mass(brackets, center, sigma, t_nu=None) -> dict:
    from scipy.stats import t as _t
    out = {}
    for b in brackets:
        st = b["strike_type"]
        if t_nu:
            s = sigma * math.sqrt((t_nu - 2) / t_nu)
            F = lambda z: float(_t.cdf(z, t_nu))
        else:
            s, F = sigma, _norm_cdf
        if st == "less":
            out[b["ticker"]] = F((float(b["cap"] or b["floor"] or 0) - center) / s)
        elif st == "greater":
            out[b["ticker"]] = 1 - F((float(b["floor"] or 0) - center) / s)
        else:
            fl = float(b["floor"] or 0)
            cap = float(b.get("cap") or fl + 100)
            out[b["ticker"]] = F((cap - center) / s) - F((fl - center) / s)
    return out


def gk_at(ts, minutes: float = 60.0) -> float | None:
    import numpy as np
    m1 = BBB.copy()
    m1 = m1[m1["time"] <= ts].tail(int(minutes) + 5)
    if len(m1) < 10:
        return None
    hl = np.log(m1["high"].values / m1["low"].values)
    co = np.log(m1["close"].values / m1["open"].values)
    v = float(np.mean(0.5 * hl * hl - (2 * np.log(2) - 1) * co * co))
    if not np.isfinite(v) or v <= 0:
        return None
    return float(m1["close"].iloc[-1] * np.sqrt(v * minutes))


BBB = None


def main(n: int = 100):
    global BBB
    import signals as S
    BBB = load_or_fetch("BTC", "1m", days=10)
    BBB["time"] = pd.to_datetime(BBB["time"], utc=True)
    rmse15 = S.sigma_for_holdout()
    et = ZoneInfo("America/New_York")
    now_et = datetime.now(et).replace(minute=0, second=0, microsecond=0)
    ms, mw = [], []
    got = scanned = 0
    h = 1
    while got < n and scanned < n + 30:
        scanned += 1
        dt = now_et - timedelta(hours=h)
        h += 1
        t = f"KXBTC-{dt.strftime('%y').upper()}{dt.strftime('%b').upper()}{dt.day:02d}{dt.hour:02d}"
        try:
            ev = K.get_event(t)
        except Exception:
            continue
        mk = ev.get("markets", [])
        if not mk or any(m.get("status") not in ("finalized", "settled") for m in mk):
            continue
        w = [m["ticker"] for m in mk if (m.get("result") or "") == "yes"]
        if len(w) != 1:
            continue
        close = datetime.fromisoformat(str(mk[0]["close_time"]).replace("Z", "+00:00"))
        sig = close - timedelta(minutes=30)
        px = BBB[BBB["time"] <= sig]
        if px.empty:
            continue
        center = float(px["close"].iloc[-1])
        br = [_parse(m) for m in mk]
        sig_static = rmse15 * 2  # sqrt(60/15)
        gk = gk_at(sig, 60.0)
        sig_gk = max(gk or 0, 0.5 * sig_static) if (gk or 0) > 0 else sig_static
        ms.append(_mass(br, center, sig_static).get(w[0], 0.0))
        mw.append(_mass(br, center, sig_gk, 5).get(w[0], 0.0))
        got += 1
    import numpy as np
    print(f"n={got} static-N winner-mass={np.mean(ms):.4f} GK-t winner-mass={np.mean(mw):.4f}")


if __name__ == "__main__":
    main()
