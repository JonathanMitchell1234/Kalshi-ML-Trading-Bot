"""Maker-mode paper execution.

Taker (status quo): lift the ask, pay taker fee, always filled.
Maker: rest a limit at mid-price, pay maker fee (25% of taker), fill only
what the visible book can absorb (conservative queue proxy).

No historical depth exists, so maker backtests are LABELED SCENARIOS, never
blended into headline stats. Live paper records both legs when EXEC_MODE=both.
Fee schedule (July 2026): taker = ceil(7c*P*(1-P)), maker = ceil(1.75c*P*(1-P)).
"""
import os
import math

EXEC_MODE = os.getenv("EXEC_MODE", "both")  # taker | maker | both
MAKER_COEF = 0.0175


def maker_fee_cents(price_cents: float, contracts: int = 1) -> float:
    p = min(max(price_cents / 100.0, 0.01), 0.99)
    return math.ceil(10000 * MAKER_COEF * p * (1 - p) - 1e-9) / 100 * contracts


def mid_price(bid_cents: int | None, ask_cents: int | None) -> int | None:
    """Limit price: mid, rounded to ticks. None when book is one-sided."""
    if bid_cents is None or ask_cents is None:
        return None
    if ask_cents - bid_cents < 2:
        return None  # no room to improve the quote
    return (bid_cents + ask_cents) // 2


def maker_fill(contracts: int, depth_contracts: int | None) -> tuple[int, int]:
    """(filled, unfilled). No depth info -> assume no fill (conservative)."""
    if depth_contracts is None or depth_contracts <= 0:
        return 0, contracts
    filled = min(contracts, int(depth_contracts))
    return filled, contracts - filled


def maker_edge(fair: float, mid_cents: int, contracts: int = 1) -> float:
    """Fee-aware edge of a maker fill (per $1 notional)."""
    allin = (mid_cents * contracts + maker_fee_cents(mid_cents, contracts)) / contracts
    return fair - allin / 100.0


if __name__ == "__main__":
    # schedule spot-checks: taker 10c->0.63, 50c->1.75, 90c->0.63
    from economics import taker_fee_cents
    for px, exp in ((10, 0.63), (50, 1.75), (90, 0.63)):
        got = taker_fee_cents(px, 1)
        assert abs(got - exp) < 1e-9, (px, got, exp)
    assert abs(maker_fee_cents(50, 1) - 0.44) < 1e-9 + 0.01
    assert mid_price(40, 50) == 45 and mid_price(40, 41) is None and mid_price(None, 50) is None
    assert maker_fill(10, 4) == (4, 6) and maker_fill(10, None) == (0, 10)
    assert maker_edge(0.60, 45) > 0.60 - 0.46
    print("maker math OK")
