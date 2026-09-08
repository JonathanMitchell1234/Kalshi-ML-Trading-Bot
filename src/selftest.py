"""Boot-time self-test: exercise every critical path, report loudly.

Runs in a background thread at startup (never blocks serving). Results are
kept in memory + returned by /api/health so the UI can banner failures.
This exists because a one-line live-path bug once blanked the weather tab
while all manual checks passed.
"""
import time
import traceback

RESULTS: dict = {"ran_at": None, "checks": {}, "ok": None}


def _check(name, fn):
    t0 = time.time()
    try:
        fn()
        RESULTS["checks"][name] = {"ok": True, "ms": int((time.time() - t0) * 1000)}
    except Exception as e:
        RESULTS["checks"][name] = {"ok": False, "error": str(e)[:200],
                                   "trace": traceback.format_exc(limit=3)[-400:]}


def run_all():
    import signals as S
    import economics as E
    import execution as X
    import tracker as T
    import kalshi_client as K
    from pathlib import Path as _P

    _check("reg_model", lambda: S.load_models())
    for asset in ("BTC", "ETH"):
        _check(f"direction_{asset}", lambda a=asset: S.load_direction_stack(a))
    _check("wx_models", lambda: [__import__("wx_signals").load_wx(c) for c in ("NYC", "CHI")])
    _check("fee_schedule", lambda: (
        abs(E.taker_fee_cents(10, 1) - 0.63) < 1e-9 or (_ for _ in ()).throw(AssertionError("taker 10c")),
        abs(E.taker_fee_cents(50, 1) - 1.75) < 1e-9 or (_ for _ in ()).throw(AssertionError("taker 50c")),
        abs(X.maker_fee_cents(50, 1) - 0.44) < 0.011 or (_ for _ in ()).throw(AssertionError("maker 50c")),
    ))
    _check("kalshi_public", lambda: K.get_event("KXBTC-26SEP0421"))
    _check("db_writable", lambda: T.conn().execute("SELECT 1").fetchone())
    _check("static_files", lambda: (
        [(_P(__file__).parent / "static" / f).exists() or (_ for _ in ()).throw(AssertionError(f))
         for f in ("dashboard.html", "weather.html", "app.js", "app.css")]
    ))
    import datetime
    RESULTS["ran_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    RESULTS["ok"] = all(v["ok"] for v in RESULTS["checks"].values())
    return RESULTS


if __name__ == "__main__":
    import json
    print(json.dumps(run_all(), indent=1, default=str))
