from __future__ import annotations

from typing import Optional

import asyncpg

from ..core.database import rows_to_list
from ..core.query import FilterParams, build_filter_where
from ..core.scoring import HARD_FILTER_PRIORITIES, PRIORITY_SQL_EXPR
from ..core.shaping import attach_computed_fields, pop_smart_score
from ..core.sql_fragments import PHONE_JOIN, PHONE_LIST_SELECT

TIER_BOUNDS: dict[str, tuple[float, float | None]] = {
    "s": (1000, None),
    "a": (700, 999),
    "b": (400, 699),
    "c": (200, 399),
    "d": (0, 199),
}

# Single-step widen when a padded/narrow query returns nothing. Not a
# ladder anymore — one honest step, then report what happened.
ZERO_RESULT_WIDEN_FACTOR = 0.35

# A range is "wide" (explicit two-sided intent) once max exceeds min by
# more than this ratio. Below it, the range is treated as effectively a
# single value and collapses to point-price behavior.
WIDE_RANGE_RATIO = 1.5

# Padding applied when a query collapses to a single effective price
# point (dial drag, or a narrow custom range).
POINT_PRICE_LOW_PAD = 0.15
POINT_PRICE_HIGH_PAD = 0.10


def _is_wide_range(min_price: Optional[float], max_price: Optional[float]) -> bool:
    if min_price is None or max_price is None or min_price <= 0:
        return False
    return (max_price / min_price) >= WIDE_RANGE_RATIO


def resolve_point_price_bounds(value: float) -> tuple[float, float]:
    """Dial drag, or a narrow custom range collapsed to one number.
    Single rule used everywhere a 'point price' needs padding."""
    return value * (1 - POINT_PRICE_LOW_PAD), value * (1 + POINT_PRICE_HIGH_PAD)


def collapse_to_point_price(min_price: Optional[float], max_price: Optional[float]) -> Optional[float]:
    """A narrow custom range (not wide enough to express real min/max
    intent) collapses to a single point, using max — consistent with the
    max-weighted bias applied to wide ranges."""
    if max_price is None:
        return None
    if min_price is None:
        return max_price
    if _is_wide_range(min_price, max_price):
        return None
    return max_price


def _widen_bounds(
    min_price: Optional[float], max_price: Optional[float], factor: float
) -> tuple[Optional[float], Optional[float]]:
    new_min = None if min_price is None else max(0.0, min_price * (1 - factor))
    new_max = None if max_price is None else max_price * (1 + factor)
    return new_min, new_max


async def recommend(
    conn: asyncpg.Connection,
    *,
    priorities: list[str],
    min_price: Optional[float],
    max_price: Optional[float],
    limit: int,
) -> dict:
    hard_ids = [p for p in priorities if p in HARD_FILTER_PRIORITIES]
    soft_ids = [p for p in priorities if p in PRIORITY_SQL_EXPR]

    if not hard_ids and not soft_ids:
        return {
            "phones": [], "priorities": [], "hard_filters": [],
            "requested_price_range": {"min": min_price, "max": max_price},
            "effective_price_range": {"min": min_price, "max": max_price},
            "budget_widened": False, "insufficient_matches": True,
        }

    hard_clause = " AND ".join(HARD_FILTER_PRIORITIES[h] for h in hard_ids)

    # Point-price collapse: narrow custom ranges behave exactly like a
    # dial drag — one number, padded asymmetrically.
    point_price = collapse_to_point_price(min_price, max_price)
    if point_price is not None:
        min_price, max_price = resolve_point_price_bounds(point_price)

    wide_range = _is_wide_range(min_price, max_price)

    if soft_ids:
        combined_expr = "(" + " + ".join(PRIORITY_SQL_EXPR[p] for p in soft_ids) + f") / {len(soft_ids)}.0"
        if wide_range:
            # Max-weighted: bias toward phones priced near the top of an
            # explicit wide range, not a flat match-score-only order.
            proximity_expr = (
                f"(1 - LEAST(ABS(COALESCE(p.price_usd, {max_price}) - {max_price}) "
                f"/ NULLIF({max_price}, 0), 1))"
            )
            order_by = (
                f"(match_score * 0.7 + {proximity_expr} * 10 * 0.3) DESC NULLS LAST, "
                "p.popularity DESC NULLS LAST, p.id DESC"
            )
        else:
            order_by = "match_score DESC NULLS LAST, p.popularity DESC NULLS LAST, p.id DESC"
    else:
        combined_expr = "NULL::numeric"
        if wide_range:
            proximity_expr = (
                f"(1 - LEAST(ABS(COALESCE(p.price_usd, {max_price}) - {max_price}) "
                f"/ NULLIF({max_price}, 0), 1))"
            )
            order_by = f"{proximity_expr} DESC NULLS LAST, p.popularity DESC NULLS LAST, p.id DESC"
        else:
            order_by = "COALESCE(sc.overall_score, 0) DESC, p.popularity DESC NULLS LAST, p.id DESC"

    requested_min, requested_max = min_price, max_price
    effective_min, effective_max = requested_min, requested_max
    widened = False
    phones: list[dict] = []

    async def _run(trial_min: Optional[float], trial_max: Optional[float]) -> list[dict]:
        where, params, _ = build_filter_where(FilterParams(min_price=trial_min, max_price=trial_max))
        if hard_clause:
            where = f"{where} AND {hard_clause}"
        i = len(params) + 1
        rows = await conn.fetch(
            f"""
            SELECT {PHONE_LIST_SELECT},
                   ({combined_expr}) AS match_score
            {PHONE_JOIN}
            WHERE {where}
            ORDER BY {order_by}
            LIMIT ${i}
            """,
            *params, limit,
        )
        return rows_to_list(rows)

    phones = await _run(requested_min, requested_max)

    # Single honest widen step whenever results fall short — regardless of
    # whether a hard filter is present. Previously this only triggered for
    # hard-filter starvation; a plain price-only search that happened to
    # land in a sparse pocket of the catalog returned however few (or
    # zero) results it found, with no attempt to recover.
    if len(phones) < limit:
        trial_min, trial_max = _widen_bounds(requested_min, requested_max, ZERO_RESULT_WIDEN_FACTOR)
        widened_phones = await _run(trial_min, trial_max)
        if len(widened_phones) > len(phones):
            phones = widened_phones
            effective_min, effective_max = trial_min, trial_max
            widened = True

    for p in phones:
        raw_match = p.pop("match_score", None)
        p["match_score"] = round(min(float(raw_match), 10.0), 1) if raw_match is not None else None
        price = p.get("price_usd")
        p["in_requested_budget"] = (
            price is None
            or (
                (requested_min is None or price >= requested_min)
                and (requested_max is None or price <= requested_max)
            )
        )

    attach_computed_fields(phones)
    for p in phones:
        p["smart_score"] = pop_smart_score(p)

    ordered_ids = hard_ids + soft_ids
    return {
        "phones": phones,
        "priorities": ordered_ids,
        "hard_filters": hard_ids,
        "requested_price_range": {"min": requested_min, "max": requested_max},
        "effective_price_range": {"min": effective_min, "max": effective_max},
        "budget_widened": widened,
        "insufficient_matches": len(phones) < limit,
    }
