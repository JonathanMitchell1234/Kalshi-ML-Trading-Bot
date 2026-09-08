"""Kalshi transaction economics (event contracts, July 2026 schedule).

Taker fee per execution:  fee = ceil(M * 0.07 * C * P * (1-P)) to the centicent,
P = price in dollars, C = contracts, M = series multiplier (default 1).
E.g. 10c -> 0.63c/contract, 50c -> 1.75c, 90c -> 0.63c.
Paper trades are takers (we lift the ask). Holding to settlement = 1 fee leg.
Verify against the live order ticket before real trading.
"""
import math
import os

MULT = float(os.getenv("KALSHI_FEE_MULT", "1"))


def taker_fee_cents(price_cents: float, contracts: int = 1, mult: float = MULT) -> float:
    p = min(max(price_cents / 100.0, 0.01), 0.99)
    per_contract = math.ceil(mult * 0.07 * p * (1 - p) * 10000 - 1e-9) / 10000  # dollars (eps: float dust vs schedule)
    return per_contract * 100 * contracts  # cents


def all_in_ask_cents(ask_cents: int) -> float:
    """Ask + taker fee for 1 contract, in cents."""
    return ask_cents + taker_fee_cents(ask_cents, 1)


def net_edge(fair_prob: float, ask_cents: int) -> float:
    """Edge after fees: fair P - all-in cost (per $1 notional)."""
    return fair_prob - all_in_ask_cents(ask_cents) / 100.0
