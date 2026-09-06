"""Fractional-Kelly sizing for binary prediction markets.

For a contract costing c (fraction, fee-included) with model fair value p:
  decimal odds b = (1 - c) / c
  full-Kelly fraction of bankroll f = (p - c) / (c * (1 - c))
Contracts = floor(bankroll * f * KELLY_FRACTION / cost_per_contract),
clamped to [1, max_contracts] and to MAX_POSITION_PCT of bankroll by cost.

Defaults are deliberately timid (tenth-Kelly, 2% cap): at our evidence
levels the job is survival + measurement, not maximization.
Env: PAPER_BANKROLL_DOLLARS (1000), KELLY_FRACTION (0.10),
     MAX_POSITION_PCT (0.02), MAX_CONTRACTS (10).
"""
import os

BANKROLL_C = int(float(os.getenv("PAPER_BANKROLL_DOLLARS", "1000")) * 100)
KELLY_FRAC = float(os.getenv("KELLY_FRACTION", "0.10"))
MAX_POS_PCT = float(os.getenv("MAX_POSITION_PCT", "0.02"))
MAX_CONTRACTS = int(os.getenv("MAX_CONTRACTS", "10"))


def kelly_fraction(p: float, cost_frac: float) -> float:
    """Full-Kelly fraction of bankroll. <=0 means no bet."""
    c = min(max(cost_frac, 0.01), 0.99)
    if p <= c:
        return 0.0
    return (p - c) / (c * (1 - c))


def size_contracts(p: float, cost_cents: float, bankroll_cents: int = BANKROLL_C,
                   kelly_frac: float = KELLY_FRAC, max_pct: float = MAX_POS_PCT,
                   max_contracts: int = MAX_CONTRACTS) -> dict:
    f = kelly_fraction(p, cost_cents / 100.0)
    if f <= 0:
        return {"contracts": 0, "kelly_f": 0.0, "reason": "no edge"}
    raw = bankroll_cents * f * kelly_frac / cost_cents
    cap_by_pct = (bankroll_cents * max_pct) // cost_cents
    n = max(1, min(int(raw), int(cap_by_pct), max_contracts))
    return {"contracts": n, "kelly_f": round(f * kelly_frac, 4),
            "bankroll_pct": round(n * cost_cents / bankroll_cents * 100, 2)}


if __name__ == "__main__":
    # sanity: p=0.60 @ 50c+fee -> f=(0.10)/(0.25)=0.40 full, tenth=0.04 -> $40 on $1000
    print(size_contracts(0.60, 51.75))
    print(size_contracts(0.51, 50.0))   # dust edge -> 1 contract floor
    print(size_contracts(0.40, 50.0))   # no edge -> 0
