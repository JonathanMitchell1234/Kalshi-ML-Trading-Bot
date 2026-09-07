"""Risk layer verification: pure-function unit tests + live halt-path drill.

Run: python src/test_risk.py   (restores all state it touches)
"""
import risk as R
import tracker as T


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    assert cond, name


def unit():
    h, r = R.check_daily_halt(-50001)
    check("daily halt trips at -$500.01", h and "500" in r)
    check("daily halt quiet at -$499", not R.check_daily_halt(-49900)[0])
    check("daily halt quiet in profit", not R.check_daily_halt(100)[0])
    check("exposure trips at 10 open", R.check_exposure(10, 100)[0])
    check("exposure trips on $100.01 trade", R.check_exposure(0, 10001)[0])
    check("exposure quiet normally", not R.check_exposure(2, 500)[0])
    check("drift trips 8/20 @60% implied", R.check_drift(8, 20, 0.60)[0])
    check("drift quiet when unconfident", not R.check_drift(8, 20, 0.50)[0])
    check("drift needs n>=20", not R.check_drift(2, 5, 0.90)[0])
    check("drift quiet when winning", not R.check_drift(12, 20, 0.60)[0])
    check("stale trips", R.check_fresh(45.0)[0])
    check("fresh ok", not R.check_fresh(5.0)[0])
    try:
        R.require_live()
        check("paper lock blocks live", False)
    except RuntimeError:
        check("paper lock blocks live", True)


def live_drill():
    import sqlite3
    with sqlite3.connect(T.DB_PATH) as c:
        before_c = c.execute("SELECT enabled FROM auto_config WHERE id=1").fetchone()
        before_w = c.execute("SELECT enabled FROM wx_config WHERE id=1").fetchone()
    T.trip_halt("all", "TEST DRILL - ignore")
    st = T.halt_state()
    check("halt recorded + visible", any(r["reason"].startswith("TEST DRILL") for r in st))
    with sqlite3.connect(T.DB_PATH) as c:
        check("crypto disabled by halt",
              c.execute("SELECT enabled FROM auto_config WHERE id=1").fetchone()[0] == 0)
        check("wx disabled by halt",
              c.execute("SELECT enabled FROM wx_config WHERE id=1").fetchone()[0] == 0)
    T.clear_halts()
    check("halt cleared", not [r for r in T.halt_state() if r["reason"].startswith("TEST DRILL")])
    with sqlite3.connect(T.DB_PATH) as c:
        c.execute("UPDATE auto_config SET enabled=? WHERE id=1", (before_c[0],))
        c.execute("UPDATE wx_config SET enabled=? WHERE id=1", (before_w[0],))
        c.commit()
    print("state restored")


def helpers():
    dp = T.day_pnl()
    check("day_pnl shape", set(dp) >= {"total_cents", "n_open", "realized_cents"})
    tp = T.trailing_perf(20, "crypto")
    check("trailing shape", set(tp) >= {"wins", "n", "mean_implied"})
    print("day:", dp, "trail-crypto:", tp)


if __name__ == "__main__":
    unit()
    helpers()
    live_drill()
    print("ALL RISK TESTS PASSED")
