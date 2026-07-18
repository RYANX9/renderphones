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

# Price-band relaxation steps, tried in order, ONLY when a hard filter
# (e.g. foldable) leaves fewer than `limit` matches inside budget. None on
# the last step means "drop the price filter entirely." A soft-only search
# never widens: returning 3 phones instead of 5 for a sparse combination is
# an honest signal about the catalog, not something to paper over.
WIDEN_STEPS: tuple[Optional[float], ...] = (0.0, 0.20, 0.40, None)


def _widen_bounds(
    min_price: Optional[float], max_price: Optional[float], factor: Optional[float]
) -> tuple[Optional[float], Optional[float]]:
    if factor is None:
        return None, None
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

    if soft_ids:
        # Average, not sum — keeps match_score a true 0-10 regardless of
        # how many priorities were picked.
        combined_expr = "(" + " + ".join(PRIORITY_SQL_EXPR[p] for p in soft_ids) + f") / {len(soft_ids)}.0"
        order_by = "match_score DESC NULLS LAST, p.popularity DESC NULLS LAST, p.id DESC"
    else:
        combined_expr = "NULL::numeric"
        order_by = "COALESCE(sc.overall_score, 0) DESC, p.popularity DESC NULLS LAST, p.id DESC"

    requested_min, requested_max = min_price, max_price
    effective_min, effective_max = requested_min, requested_max
    widen_level = 0
    phones: list[dict] = []

    for step_idx, factor in enumerate(WIDEN_STEPS):
        trial_min, trial_max = _widen_bounds(requested_min, requested_max, factor)

        where, params = build_filter_where(FilterParams(min_price=trial_min, max_price=trial_max))
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
        phones = rows_to_list(rows)
        effective_min, effective_max = trial_min, trial_max
        widen_level = step_idx

        if not hard_ids or len(phones) >= limit or factor is None:
            break

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
        "budget_widened": widen_level > 0,
        "insufficient_matches": len(phones) < limit,
    }
