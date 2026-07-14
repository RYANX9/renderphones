from __future__ import annotations

import logging
from typing import Optional

import anyio
from fastapi import APIRouter, HTTPException, Query

from cache import cached
from config import settings
from database import (
    get_pool,
    row_to_dict,
    rows_to_list,
    PHONE_SCORED_SELECT,
    PHONE_SCORED_FROM,
    SCORED_SORT_COL_MAP,
)
from utils.query import build_search_where
from utils.scoring import attach_computed_fields, PRIORITY_SQL_EXPR, HARD_FILTER_PRIORITIES
from .recommend_copy import generate_match_copy
from .compare_copy import generate_compare_verdict

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/phones", tags=["phones"])

_SMART_KEYS = (
    "smart_overall_score", "smart_value_score", "smart_camera_score",
    "smart_performance_score", "smart_battery_score", "smart_display_score",
    "smart_build_score", "smart_strengths", "smart_weaknesses",
    "smart_reasoning", "smart_model_version", "smart_scored_at", "smart_tier",
)

# Peer-group size used to normalise the value_score fallback for a single
# phone lookup (get_phone). Must be called AFTER attach_computed_fields —
# that function reads smart_value_score off the same dict to compute
# value_score, and _pop_smart_score removes it.
_VALUE_PEER_LIMIT = 40

_PRIORITY_LABEL = {
    "camera": "Camera Quality",
    "battery": "Battery Life",
    "performance": "Performance",
    "gaming": "Gaming Performance",
    "compact": "Compact Size",
    "lightweight": "Lightweight",
    "display": "Display Quality",
    "smooth_display": "High Refresh Rate",
    "fast_charging": "Fast Charging",
    "wireless_charging": "Wireless Charging",
    "foldable": "Foldable Design",
    "durability": "Water/Dust Resistance",
    "value": "Best Value",
}


def _pop_smart_score(d: dict) -> Optional[dict]:
    has_score = d.get("smart_overall_score") is not None
    out = None
    if has_score:
        out = {
            "overall_score":     d.get("smart_overall_score"),
            "camera_score":      d.get("smart_camera_score"),
            "performance_score": d.get("smart_performance_score"),
            "battery_score":     d.get("smart_battery_score"),
            "display_score":     d.get("smart_display_score"),
            "build_score":       d.get("smart_build_score"),
            "value_score":       d.get("smart_value_score"),
            "strengths":         d.get("smart_strengths"),
            "weaknesses":        d.get("smart_weaknesses"),
            "reasoning":         d.get("smart_reasoning"),
            "tier":              d.get("smart_tier"),
            "model_version":     d.get("smart_model_version"),
            "scored_at":         d.get("smart_scored_at"),
        }
    for k in _SMART_KEYS:
        d.pop(k, None)
    return out


def _resolve_sort(sort_by: str, sort_order: str) -> tuple[str, str]:
    expr = SCORED_SORT_COL_MAP.get(sort_by, SCORED_SORT_COL_MAP["release_ts"])
    order = "DESC" if sort_order.lower() == "desc" else "ASC"
    return expr, order


def _round_money(v) -> Optional[float]:
    if v is None:
        return None
    return round(float(v), 2)


async def _latest_price(conn, phone_id: int) -> Optional[dict]:
    row = await conn.fetchrow(
        """
        SELECT price_usd, price_original, scope, snapshot_date
        FROM   price_points
        WHERE  phone_id = $1
        ORDER  BY (scope = 'global') DESC, snapshot_date DESC
        LIMIT  1
        """,
        phone_id,
    )
    if not row:
        return None
    d = row_to_dict(row)
    d["price_usd"] = _round_money(d["price_usd"])
    d["price_original"] = _round_money(d.get("price_original"))
    return d


def _apply_latest_price(target: dict, price: Optional[dict]) -> None:
    """A price_points row with a NULL price_usd (phone currently
    untracked / out of stock) must not clobber the denormalised
    phones.price_usd fallback."""
    if price is None or price.get("price_usd") is None:
        return
    target["price_usd"] = price["price_usd"]
    target["price_original"] = price.get("price_original")
    target["price_updated_at"] = str(price["snapshot_date"])
    target["price_scope"] = price["scope"]


async def _fetch_value_peers(conn, phone: dict) -> list[dict]:
    """Real comparison set for the value_score fallback on a single-phone
    lookup — same price-band-plus-brand logic as /similar, so the number
    shown on the detail page is computed the same way as on list pages
    instead of degenerating to a peer group of one."""
    price = phone.get("price_usd")
    lo = price * 0.65 if price else None
    hi = price * 1.45 if price else None

    rows = await conn.fetch(
        f"""
        SELECT {PHONE_SCORED_SELECT}
        {PHONE_SCORED_FROM}
        WHERE  p.id != $1
          AND  ($2::numeric IS NULL OR p.price_usd BETWEEN $2 AND $3)
        ORDER  BY
            CASE WHEN p.brand = $4 THEN 0 ELSE 1 END,
            ABS(COALESCE(p.price_usd, 0) - COALESCE($5, 0)),
            p.popularity DESC NULLS LAST
        LIMIT  $6
        """,
        phone["id"], lo, hi, phone.get("brand"), price, _VALUE_PEER_LIMIT,
    )
    return rows_to_list(rows)


@router.get("/search")
async def search_phones(
    q:                  Optional[str]   = Query(None),
    brand:              Optional[str]   = Query(None),
    min_price:          Optional[float] = Query(None),
    max_price:          Optional[float] = Query(None),
    min_ram:            Optional[int]   = Query(None),
    min_battery:        Optional[int]   = Query(None),
    min_camera_mp:      Optional[int]   = Query(None),
    min_screen_size:    Optional[float] = Query(None),
    max_screen_size:    Optional[float] = Query(None),
    min_year:           Optional[int]   = Query(None),
    max_weight:         Optional[int]   = Query(None),
    min_charging_w:     Optional[int]   = Query(None),
    chipset_tier_param: Optional[str]   = Query(None, alias="chipset_tier"),
    sort_by:            str = Query("release_ts"),
    sort_order:         str = Query("desc"),
    page:               int = Query(1, ge=1),
    page_size:          int = Query(24, ge=1, le=100),
):
    where, params = build_search_where(
        q=q,
        brand=brand,
        min_price=min_price,
        max_price=max_price,
        min_ram=min_ram,
        min_battery=min_battery,
        min_camera_mp=min_camera_mp,
        min_screen_size=min_screen_size,
        max_screen_size=max_screen_size,
        min_year=min_year,
        max_weight=max_weight,
        min_charging_w=min_charging_w,
        chipset_tier_filter=chipset_tier_param,
    )
    sort_expr, order = _resolve_sort(sort_by, sort_order)
    offset = (page - 1) * page_size

    async with get_pool().acquire() as conn:
        total = await conn.fetchval(
            f"SELECT COUNT(*) {PHONE_SCORED_FROM} WHERE {where}",
            *params,
        )
        rows = await conn.fetch(
            f"""
            SELECT {PHONE_SCORED_SELECT}
            {PHONE_SCORED_FROM}
            WHERE  {where}
            ORDER  BY {sort_expr} {order} NULLS LAST, p.id DESC
            LIMIT  {page_size} OFFSET {offset}
            """,
            *params,
        )

    phones = rows_to_list(rows)
    attach_computed_fields(phones)
    for p in phones:
        p["smart_score"] = _pop_smart_score(p)

    return {
        "total":     total,
        "page":      page,
        "page_size": page_size,
        "results":   phones,
    }


@router.get("/latest")
async def latest_phones(limit: int = Query(20, ge=1, le=100)):
    async def _fetch():
        async with get_pool().acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT {PHONE_SCORED_SELECT}
                {PHONE_SCORED_FROM}
                ORDER  BY p.release_year DESC NULLS LAST,
                          p.release_month DESC NULLS LAST,
                          p.release_day DESC NULLS LAST,
                          p.id DESC
                LIMIT  {limit}
                """
            )
        phones = rows_to_list(rows)
        attach_computed_fields(phones)
        for p in phones:
            p["smart_score"] = _pop_smart_score(p)
        return {"phones": phones}

    return await cached(f"phones:latest:{limit}", settings.cache_ttl_stable, _fetch)


@router.get("/trending")
async def trending_phones(limit: int = Query(10, ge=1, le=50)):
    async def _fetch():
        async with get_pool().acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT {PHONE_SCORED_SELECT}
                {PHONE_SCORED_FROM}
                ORDER  BY p.popularity DESC NULLS LAST,
                          p.fans DESC NULLS LAST,
                          COALESCE(s.overall_score, 0) DESC,
                          p.release_year DESC NULLS LAST,
                          p.id DESC
                LIMIT  {limit}
                """
            )
        phones = rows_to_list(rows)
        attach_computed_fields(phones)
        for p in phones:
            p["smart_score"] = _pop_smart_score(p)
        return {"phones": phones}

    return await cached(f"phones:trending:{limit}", settings.cache_ttl_trending, _fetch)


@router.get("/compare")
async def compare_phones(
    ids:   Optional[str] = Query(None),
    slugs: Optional[str] = Query(None),
):
    if not ids and not slugs:
        raise HTTPException(status_code=400, detail="Provide `ids` or `slugs`.")

    async with get_pool().acquire() as conn:
        if ids:
            id_list = [int(i) for i in ids.split(",") if i.strip().isdigit()]
            if not id_list:
                raise HTTPException(status_code=400, detail="No valid ids provided.")
            rows = await conn.fetch(
                f"""
                SELECT {PHONE_SCORED_SELECT}
                {PHONE_SCORED_FROM}
                WHERE  p.id = ANY($1::int[])
                """,
                id_list,
            )
        else:
            slug_list = [s.strip() for s in slugs.split(",") if s.strip()]
            if not slug_list:
                raise HTTPException(status_code=400, detail="No valid slugs provided.")
            rows = await conn.fetch(
                f"""
                SELECT {PHONE_SCORED_SELECT}
                {PHONE_SCORED_FROM}
                WHERE  p.slug = ANY($1::text[])
                """,
                slug_list,
            )

        phones = rows_to_list(rows)
        for p in phones:
            price = await _latest_price(conn, p["id"])
            _apply_latest_price(p, price)

    attach_computed_fields(phones)
    for p in phones:
        p["smart_score"] = _pop_smart_score(p)

    # Blocking HTTP call to Gemini — offloaded to a worker thread so it
    # doesn't stall the event loop for the duration of the round trip.
    verdict = None
    if len(phones) >= 2:
        verdict = await anyio.to_thread.run_sync(generate_compare_verdict, phones)

    return {"phones": phones, "verdict": verdict}


_TIER_BOUNDS: dict[str, tuple[float, float | None]] = {
    "s": (1000, None),
    "a": (700, 999),
    "b": (400, 699),
    "c": (200, 399),
    "d": (0, 199),
}

# Price-band relaxation steps tried, in order, ONLY when a hard filter (e.g.
# foldable) leaves fewer than `limit` matches inside the requested budget.
# None on the last step means "drop the price filter entirely." A soft-only
# search NEVER widens — returning 3 phones instead of 5 for a sparse category
# is an honest signal about the catalog, not a problem to paper over.
_WIDEN_STEPS: tuple[float | None, ...] = (0.0, 0.20, 0.40, None)


def _widen_bounds(
    min_price: Optional[float], max_price: Optional[float], factor: Optional[float]
) -> tuple[Optional[float], Optional[float]]:
    if factor is None:
        return None, None
    new_min = None if min_price is None else max(0.0, min_price * (1 - factor))
    new_max = None if max_price is None else max_price * (1 + factor)
    return new_min, new_max


@router.get("/recommend")
async def recommend_phones(
    priorities: str = Query(..., description="Comma-separated priority ids"),
    tier:       Optional[str]   = Query(None, description="s|a|b|c|d — overrides min_price/max_price when set"),
    min_price:  Optional[float] = Query(None),
    max_price:  Optional[float] = Query(None),
    limit:      int = Query(5, ge=1, le=20),
):
    if tier and tier in _TIER_BOUNDS:
        min_price, max_price = _TIER_BOUNDS[tier]

    requested_min, requested_max = min_price, max_price

    all_ids  = [p.strip() for p in priorities.split(",") if p.strip()]
    hard_ids = [p for p in all_ids if p in HARD_FILTER_PRIORITIES]
    soft_ids = [p for p in all_ids if p in PRIORITY_SQL_EXPR]

    if not hard_ids and not soft_ids:
        raise HTTPException(status_code=400, detail="No valid priorities provided.")

    hard_clause = " AND ".join(HARD_FILTER_PRIORITIES[h] for h in hard_ids)

    if soft_ids:
        # Average, not sum: keeps match_score a true 0-10 regardless of how
        # many priorities were picked, so it never needs post-hoc clamping
        # (the old min(raw_sum, 10.0) is why every result showed "10.0/10" —
        # a sum of 2-3 uncapped 0-10 terms crosses 10 almost immediately).
        combined_expr = "(" + " + ".join(PRIORITY_SQL_EXPR[p] for p in soft_ids) + f") / {len(soft_ids)}.0"
        order_by = "match_score DESC NULLS LAST, p.popularity DESC NULLS LAST, p.id DESC"
    else:
        combined_expr = "NULL::numeric"
        order_by = "COALESCE(s.overall_score, 0) DESC, p.popularity DESC NULLS LAST, p.id DESC"

    widen_level = 0
    phones: list[dict] = []
    effective_min, effective_max = requested_min, requested_max

    async with get_pool().acquire() as conn:
        for step_idx, factor in enumerate(_WIDEN_STEPS):
            trial_min, trial_max = _widen_bounds(requested_min, requested_max, factor)

            where, params = build_search_where(min_price=trial_min, max_price=trial_max)
            if hard_clause:
                where = f"{where} AND {hard_clause}"
            i = len(params) + 1

            rows = await conn.fetch(
                f"""
                SELECT {PHONE_SCORED_SELECT},
                       ({combined_expr}) AS match_score
                {PHONE_SCORED_FROM}
                WHERE  {where}
                ORDER  BY {order_by}
                LIMIT  ${i}
                """,
                *params,
                limit,
            )
            phones = rows_to_list(rows)
            effective_min, effective_max = trial_min, trial_max
            widen_level = step_idx

            # Only a hard filter justifies widening the budget. A soft-only
            # search stops after one pass no matter how few rows come back.
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
        p["smart_score"] = _pop_smart_score(p)

    ordered_ids = hard_ids + soft_ids
    priority_labels = [_PRIORITY_LABEL.get(p, p) for p in ordered_ids]
    if requested_min and requested_max:
        budget_label = f"${requested_min:.0f}-${requested_max:.0f}"
    elif requested_max:
        budget_label = f"under ${requested_max:.0f}"
    elif requested_min:
        budget_label = f"${requested_min:.0f}+"
    else:
        budget_label = "any budget"

    # Blocking HTTP call to Gemini — offloaded to a worker thread.
    match_copy = await anyio.to_thread.run_sync(
        generate_match_copy, phones, ordered_ids, budget_label
    )
    if match_copy:
        for p in phones:
            copy = match_copy.get(p["id"])
            if copy:
                p["match_line"] = copy["match_line"]
                p["tradeoff_line"] = copy["tradeoff_line"]

    return {
        "phones": phones,
        "priorities": ordered_ids,
        "hard_filters": hard_ids,
        "requested_price_range": {"min": requested_min, "max": requested_max},
        "effective_price_range": {"min": effective_min, "max": effective_max},
        "budget_widened": widen_level > 0,
        "insufficient_matches": len(phones) < limit,
    }


@router.get("/{phone_id}")
async def get_phone(phone_id: int):
    async with get_pool().acquire() as conn:
        row = await conn.fetchrow(
            f"""
            SELECT {PHONE_SCORED_SELECT}
            {PHONE_SCORED_FROM}
            WHERE  p.id = $1
            """,
            phone_id,
        )
        if row is None:
            raise HTTPException(status_code=404, detail=f"Phone {phone_id} not found.")

        phone = row_to_dict(row)

        price = await _latest_price(conn, phone_id)
        _apply_latest_price(phone, price)

        peers = await _fetch_value_peers(conn, phone) if phone.get("smart_value_score") is None else []

    attach_computed_fields([phone], peers=peers or [phone])
    phone["smart_score"] = _pop_smart_score(phone)
    return phone


@router.get("/{phone_id}/similar")
async def similar_phones(phone_id: int, limit: int = Query(12, ge=1, le=30)):
    async with get_pool().acquire() as conn:
        base = await conn.fetchrow(
            "SELECT price_usd, brand, screen_size FROM phones WHERE id = $1",
            phone_id,
        )
        if base is None:
            raise HTTPException(status_code=404, detail=f"Phone {phone_id} not found.")

        price = float(base["price_usd"]) if base["price_usd"] else None
        lo = price * 0.65 if price else None
        hi = price * 1.45 if price else None

        rows = await conn.fetch(
            f"""
            SELECT {PHONE_SCORED_SELECT}
            {PHONE_SCORED_FROM}
            WHERE  p.id != $1
              AND  ($2::numeric IS NULL OR p.price_usd BETWEEN $2 AND $3)
            ORDER  BY
                CASE WHEN p.brand = $4 THEN 0 ELSE 1 END,
                ABS(COALESCE(p.price_usd, 0) - COALESCE($5, 0)),
                p.popularity DESC NULLS LAST
            LIMIT  $6
            """,
            phone_id, lo, hi, base["brand"], price, limit,
        )

    phones = rows_to_list(rows)
    attach_computed_fields(phones)
    for p in phones:
        p["smart_score"] = _pop_smart_score(p)
    return {"phones": phones}


@router.get("/{phone_id}/price-history")
async def price_history(
    phone_id: int,
    condition: str = Query(
        "new",
        description="price_history row filter: 'new', 'used', or 'all'.",
    ),
    scope: str = Query(
        "global",
        description="price_points row filter: 'global', 'local', or 'all'.",
    ),
):
    async with get_pool().acquire() as conn:
        exists = await conn.fetchval("SELECT 1 FROM phones WHERE id = $1", phone_id)
        if not exists:
            raise HTTPException(status_code=404, detail=f"Phone {phone_id} not found.")

        if condition == "all":
            history_rows = await conn.fetch(
                """
                SELECT snapshot_date, condition, min_price_usd, max_price_usd,
                       avg_price_usd, listing_count
                FROM   price_history
                WHERE  phone_id = $1
                ORDER  BY snapshot_date ASC, condition ASC
                """,
                phone_id,
            )
        else:
            history_rows = await conn.fetch(
                """
                SELECT snapshot_date, condition, min_price_usd, max_price_usd,
                       avg_price_usd, listing_count
                FROM   price_history
                WHERE  phone_id = $1 AND condition = $2
                ORDER  BY snapshot_date ASC
                """,
                phone_id, condition,
            )

        if scope == "all":
            point_rows = await conn.fetch(
                """
                SELECT snapshot_date, scope, price_usd
                FROM   price_points
                WHERE  phone_id = $1
                ORDER  BY snapshot_date ASC, scope ASC
                """,
                phone_id,
            )
        else:
            point_rows = await conn.fetch(
                """
                SELECT snapshot_date, scope, price_usd
                FROM   price_points
                WHERE  phone_id = $1 AND scope = $2
                ORDER  BY snapshot_date ASC
                """,
                phone_id, scope,
            )

    points = rows_to_list(history_rows)
    for pt in points:
        pt["min_price_usd"] = _round_money(pt.get("min_price_usd"))
        pt["max_price_usd"] = _round_money(pt.get("max_price_usd"))
        pt["avg_price_usd"] = _round_money(pt.get("avg_price_usd"))

    price_pts = rows_to_list(point_rows)
    for pp in price_pts:
        pp["price_usd"] = _round_money(pp.get("price_usd"))

    return {
        "phone_id": phone_id,
        "points": points,
        "price_points": price_pts,
    }
