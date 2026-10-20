# app/services/recommend_service.py
from __future__ import annotations

from typing import Optional

import asyncpg

from ..core.database import rows_to_list
from ..core.query import FilterParams, build_filter_where
from ..core.scoring import HARD_FILTER_PRIORITIES, PRIORITY_SQL_EXPR
from ..core.shaping import shape_cards
from ..core.sql_fragments import CARD_SELECT

# Legacy price tiers, still accepted via ?tier=s|a|b|c|d
TIER_BOUNDS: dict[str, tuple[float, float | None]] = {
    "s": (1000, None),
    "a": (700, 999),
    "b": (400, 699),
    "c": (200, 399),
    "d": (0, 199),
}

# Tier price ranges used by /pick. Half-open [min, upper): a $699.50 phone must
# land in exactly one tier, so the upper bound is exclusive (no 699 -> 700 gap).
# TIER_BOUNDS above is kept untouched because trade-in imports it.
TIER_RANGES: dict[str, tuple[float, float | None]] = {
    "s": (1000, None),
    "a": (700, 1000),
    "b": (400, 700),
    "c": (200, 400),
    "d": (0, 200),
}

# The dial value is a CEILING. Range = 90% .. 105% of it. Keep in sync with
# lib/priceTiers.ts (BUDGET_FLOOR_RATIO / BUDGET_CEILING_RATIO / DIAL_MAX).
BUDGET_FLOOR_RATIO = 0.90
BUDGET_CEILING_RATIO = 1.05
BUDGET_UNCAPPED_AT = 2000
# When NO phone matches (for the chosen priorities), the floor drops to 85%, never lower.
# The ceiling never moves.
BUDGET_WIDEN_FLOOR_RATIO = 0.85
LEGACY_WIDEN_FLOOR_FACTOR = 0.65

CANDIDATE_POOL_MULTIPLIER = 4
MIN_CANDIDATE_POOL = 20
MAX_PER_BRAND = 2


def _round_half_up(x: float) -> int:
    # Match JS Math.round (Python's round() is banker's rounding: 577.5 vs 2.5 differ).
    return int(x + 0.5)


def budget_bounds(budget: float) -> tuple[float, float | None]:
    """Dial -> (floor, ceiling). Rounded exactly like lib/priceTiers.ts so the
    range the user sees is the range that is queried."""
    ceiling = None if budget >= BUDGET_UNCAPPED_AT else _round_half_up(budget * BUDGET_CEILING_RATIO)
    return _round_half_up(budget * BUDGET_FLOOR_RATIO), ceiling


def _apply_brand_cap(phones: list[dict], cap: int) -> list[dict]:
    counts: dict[str, int] = {}
    out: list[dict] = []
    for p in phones:
        brand = (p.get("brand_slug") or p.get("brand") or "").strip().lower()
        if counts.get(brand, 0) >= cap:
            continue
        counts[brand] = counts.get(brand, 0) + 1
        out.append(p)
    return out


async def recommend(
    conn: asyncpg.Connection,
    *,
    priorities: list[str],
    min_price: Optional[float],
    max_price: Optional[float],
    limit: int,
    region: Optional[str] = None,
    budget: Optional[float] = None,
    tier: Optional[str] = None,
) -> dict:
    """Price scope is EITHER a tier OR the dial, never both (tier wins):
      tier   -> that tier's fixed range, no widening, dial ignored
      budget -> 90%..105% of the dial value, floor widens to 85% only if nothing matches
    Ranking: overall_score when no priorities are given; the average of the
    chosen priority scores (overall_score as tiebreak) when they are."""
    hard_ids = [p for p in priorities if p in HARD_FILTER_PRIORITIES]
    soft_ids = [p for p in priorities if p in PRIORITY_SQL_EXPR]

    hard_clause = " AND ".join(HARD_FILTER_PRIORITIES[h] for h in hard_ids)

    # Exclusive upper bound only exists in tier mode.
    upper_exclusive: Optional[float] = None
    mode = "range"
    if tier in TIER_RANGES:
        mode, budget = "tier", None
        min_price, upper_exclusive = TIER_RANGES[tier]
        max_price = None
    elif budget is not None:
        mode = "dial"
        min_price, max_price = budget_bounds(budget)

    if priorities and not hard_ids and not soft_ids:
        return {
            "phones": [], "priorities": [], "hard_filters": [], "budget": budget, "mode": mode, "tier": tier if mode == "tier" else None,
            "requested_price_range": {"min": min_price, "max": max_price},
            "effective_price_range": {"min": min_price, "max": max_price},
            "budget_widened": False, "insufficient_matches": True,
        }

    if soft_ids:
        combined_expr = "(" + " + ".join(PRIORITY_SQL_EXPR[p] for p in soft_ids) + f") / {len(soft_ids)}.0"
        order_by = (
            "match_score DESC NULLS LAST, COALESCE(p.overall_score, 0) DESC, "
            "p.trend_score DESC NULLS LAST, p.id DESC"
        )
    else:
        combined_expr = "NULL::numeric"
        order_by = "COALESCE(p.overall_score, 0) DESC, p.trend_score DESC NULLS LAST, p.id DESC"

    requested_min, requested_max = min_price, max_price
    effective_min, effective_max = requested_min, requested_max
    # What the UI shows as the top of a tier: $700 exclusive -> "$699".
    display_max = requested_max if upper_exclusive is None else upper_exclusive - 1
    widened = False
    pool_size = max(limit * CANDIDATE_POOL_MULTIPLIER, MIN_CANDIDATE_POOL)

    async def _run(lo: Optional[float], hi: Optional[float]) -> list[dict]:
        where, params, _ = build_filter_where(
            FilterParams(min_price=lo, max_price=hi, region=region)
        )
        if hard_clause:
            where = f"{where} AND {hard_clause}"
        if upper_exclusive is not None:
            params.append(upper_exclusive)
            where = f"{where} AND p.price_usd < ${len(params)}"
        i = len(params) + 1
        rows = await conn.fetch(
            f"SELECT {CARD_SELECT}, ({combined_expr}) AS match_score FROM phone_catalog p "
            f"WHERE {where} ORDER BY {order_by} LIMIT ${i}",
            *params, pool_size,
        )
        return _apply_brand_cap(rows_to_list(rows), MAX_PER_BRAND)

    phones = await _run(requested_min, requested_max)

    # Tier mode is a hard range: never widened. Dial mode lowers the floor only, and only when
    # nothing matches at all. Legacy min/max mode keeps its old "fewer than limit" rule.
    needs_widen = (not phones) if mode == "dial" else (len(phones) < limit)
    if mode != "tier" and needs_widen and requested_min is not None:
        trial_min = (
            budget * BUDGET_WIDEN_FLOOR_RATIO if budget is not None
            else requested_min * LEGACY_WIDEN_FLOOR_FACTOR
        )
        wider = await _run(trial_min, requested_max)
        if len(wider) > len(phones):
            phones, effective_min, widened = wider, trial_min, True

    phones = phones[:limit]

    for p in phones:
        raw = p.pop("match_score", None)
        p["match_score"] = round(min(float(raw), 10.0), 1) if raw is not None else None
        price = p.get("price_usd")
        # Only meaningful for the dial: a phone between the dial value and +5%.
        # No ceiling at the top of the dial, so nothing can be "over budget" there.
        p["over_budget"] = bool(
            mode == "dial" and budget < BUDGET_UNCAPPED_AT and price is not None and price > budget
        )
        p["below_range"] = bool(
            mode == "dial" and price is not None and requested_min is not None and price < requested_min
        )
        p["in_requested_budget"] = (
            price is None
            or (
                (requested_min is None or price >= requested_min)
                and (requested_max is None or price <= requested_max)
                and (upper_exclusive is None or price < upper_exclusive)
            )
        )

    return {
        "phones": shape_cards(phones),
        "priorities": hard_ids + soft_ids,
        "hard_filters": hard_ids,
        "budget": budget,
        "mode": mode,
        "tier": tier if mode == "tier" else None,
        "requested_price_range": {"min": requested_min, "max": display_max},
        "effective_price_range": {"min": effective_min, "max": display_max},
        "budget_widened": widened,
        "insufficient_matches": len(phones) < limit,
    }