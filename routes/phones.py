from __future__ import annotations

import logging
from typing import Optional

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
from utils.scoring import attach_computed_fields, PRIORITY_SQL_EXPR

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/phones", tags=["phones"])

_SMART_KEYS = (
    "smart_overall_score", "smart_value_score", "smart_camera_score",
    "smart_performance_score", "smart_battery_score", "smart_display_score",
    "smart_build_score", "smart_strengths", "smart_weaknesses",
    "smart_reasoning", "smart_model_version", "smart_scored_at", "smart_tier",
)


def _pop_smart_score(d: dict) -> Optional[dict]:
    """Pulls the smart_* columns (from the phone_smart_scores join) off a
    row dict and returns them as a nested SmartScore-shaped dict, or None
    if the phone has never been scored. Mutates `d` in place."""
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
    """Every price column comes out of Postgres as a numeric with float-noise
    tails (1055.26729, 1489.8059999999998). Round once, at the API boundary,
    to 2dp everywhere a price leaves this service."""
    if v is None:
        return None
    return round(float(v), 2)


async def _latest_price(conn, phone_id: int) -> Optional[dict]:
    """price_points is the source of truth for current pricing —
    phones.price_usd is a denormalised snapshot that can lag behind it.
    'global' scope is the canonical price; only fall back to 'local' rows
    if the phone has never had a global snapshot at all. Within whichever
    scope wins, pick the most recent snapshot_date."""
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
    for p in phones:
        _pop_smart_score(p)
    attach_computed_fields(phones)

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
        for p in phones:
            _pop_smart_score(p)
        attach_computed_fields(phones)
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
        for p in phones:
            _pop_smart_score(p)
        attach_computed_fields(phones)
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
            p["smart_score"] = _pop_smart_score(p)
            price = await _latest_price(conn, p["id"])
            if price:
                p["price_usd"] = float(price["price_usd"])
                p["price_original"] = price.get("price_original")
                p["price_updated_at"] = str(price["snapshot_date"])
                p["price_scope"] = price["scope"]

    attach_computed_fields(phones)
    return {"phones": phones}


@router.get("/recommend")
async def recommend_phones(
    priorities: str = Query(..., description="Comma-separated priority ids"),
    min_price:  Optional[float] = Query(None),
    max_price:  Optional[float] = Query(None),
    limit:      int = Query(5, ge=1, le=20),
):
    priority_list = [p.strip() for p in priorities.split(",") if p.strip() in PRIORITY_SQL_EXPR]
    if not priority_list:
        raise HTTPException(status_code=400, detail="No valid priorities provided.")

    combined_expr = " + ".join(PRIORITY_SQL_EXPR[p] for p in priority_list)

    where, params = build_search_where(min_price=min_price, max_price=max_price)
    i = len(params) + 1

    async with get_pool().acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT {PHONE_SCORED_SELECT},
                   ({combined_expr}) AS match_score
            {PHONE_SCORED_FROM}
            WHERE  {where}
            ORDER  BY match_score DESC NULLS LAST, p.popularity DESC NULLS LAST, p.id DESC
            LIMIT  ${i}
            """,
            *params,
            limit,
        )

    phones = rows_to_list(rows)
    for p in phones:
        _pop_smart_score(p)
        raw_match = p.pop("match_score", None)
        p["match_score"] = round(min(float(raw_match), 10.0), 1) if raw_match is not None else None
    attach_computed_fields(phones)

    return {"phones": phones, "priorities": priority_list}


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
        phone["smart_score"] = _pop_smart_score(phone)

        price = await _latest_price(conn, phone_id)
        if price:
            phone["price_usd"] = float(price["price_usd"])
            phone["price_original"] = price.get("price_original")
            phone["price_updated_at"] = str(price["snapshot_date"])
            phone["price_scope"] = price["scope"]

    attach_computed_fields([phone])
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
    for p in phones:
        _pop_smart_score(p)
    attach_computed_fields(phones)
    return {"phones": phones}


@router.get("/{phone_id}/price-history")
async def price_history(
    phone_id: int,
    condition: str = Query(
        "new",
        description="price_history row filter: 'new', 'used', or 'all'. "
                     "price_history has one row per (date, condition) — "
                     "without this filter a graph gets two overlapping "
                     "series per day.",
    ),
    scope: str = Query(
        "global",
        description="price_points row filter: 'global', 'local', or 'all'. "
                     "Same duplication issue as `condition`, one row per "
                     "(date, scope).",
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
