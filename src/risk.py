"""Risk layer: pure, unit-testable guardrails. Paper-enforced today.

Reads thresholds from env (safe defaults). No network. Halts by flipping
the traders' own enabled flags + a global halt record, so the existing
start/stop UI keeps working and every halt shows up in the cycle logs.
"""
import os
from datetime import datetime, timezone

MAX_DAILY_LOSS_C = abs(float(os.getenv("MAX_DAILY_LOSS", "500"))) * 100  # cents, positive number
MAX_OPEN = int(os.getenv("MAX_OPEN_POSITIONS", "10"))
MAX_POS_C = float(os.getenv("MAX_POSITION_SIZE", "100")) * 100
DRIFT_N = int(os.getenv("DRIFT_WINDOW", "20"))
DRIFT_FLOOR = float(os.getenv("DRIFT_MIN_WIN", "0.45"))
STALE_BARS_MIN = float(os.getenv("STALE_BARS_MIN", "30"))
LIVE_OK = os.getenv("LIVE_TRADING", "false").lower() in ("1", "true", "yes")


def require_live():
    """Any future real-order path must call this. Paper cannot trip it."""
    if not LIVE_OK:
        raise RuntimeError("LIVE_TRADING is off: real orders are locked (paper mode).")


def utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def check_daily_halt(day_pnl_cents: float) -> tuple[bool, str]:
    """True = halt. day_pnl includes open risk (conservative)."""
    if day_pnl_cents <= -MAX_DAILY_LOSS_C:
        return True, f"daily loss {day_pnl_cents/100:.0f} breached -${MAX_DAILY_LOSS_C/100:.0f}"
    return False, ""


def check_exposure(n_open: int, cost_cents: float) -> tuple[bool, str]:
    if n_open >= MAX_OPEN:
        return True, f"{n_open} open positions (max {MAX_OPEN})"
    if cost_cents > MAX_POS_C:
        return True, f"cost ${cost_cents/100:.2f} exceeds ${MAX_POS_C/100:.0f} single-trade cap"
    return False, ""


def check_drift(wins: int, n: int, mean_implied: float | None) -> tuple[bool, str]:
    """Halt when the model is proven wrong (not merely unlucky): trailing
    window wins below floor while its own prices implied better."""
    if n < DRIFT_N or mean_implied is None:
        return False, ""
    if mean_implied < 0.55:
        return False, ""  # model wasn't confident; nothing disproven
    if wins / n < DRIFT_FLOOR:
        return True, f"drift: {wins}/{n} trailing vs {mean_implied:.0%} implied"
    return False, ""


def check_fresh(age_min: float | None, limit_min: float = STALE_BARS_MIN) -> tuple[bool, str]:
    if age_min is None:
        return True, "unknown data age"
    if age_min > limit_min:
        return True, f"data {age_min:.0f}m old (limit {limit_min:.0f}m)"
    return False, ""
