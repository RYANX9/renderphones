from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

# CORRECT - matches your actual structure
from cache import cached
from config import get_settings
settings = get_settings()  
from database import get_pool, row_to_dict, rows_to_list, PHONE_LIST_SELECT, RELEASE_TS_EXPR
from utils.query import build_search_where, resolve_sort, parse_json_safe
from utils.scoring import chipset_tier, compute_value_score

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/brands", tags=["brands"])

_VALID_BRAND_SORTS = {
    "release_year", "price_usd", "battery_capacity", "main_camera_mp", "antutu_score",
}


@router.get("")
async def list_brands():
    """All brands with phone count. Cached 1 hour."""
    async def _fetch():
        async with get_pool().acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT brand, COUNT(*) AS phone_count
                FROM   phones
                WHERE  brand IS NOT NULL
                GROUP  BY brand
                ORDER  BY phone_count DESC
                """
            )
        return {"brands": [{"brand": r["brand"], "count": r["phone_count"]} for r in rows]}

    return await cached("brands:all", settings.cache_ttl_stable, _fetch)


@router.get("/{brand_slug}")
async def get_brand(brand_slug: str):
    """Brand stats + latest phone. Cached 1 hour."""
    brand_name = slug_to_words(brand_slug)

    async def _fetch():
        async with get_pool().acquire() as conn:
            stats = await conn.fetchrow(
                """
                SELECT
                    brand,
                    COUNT(*)                                             AS total,
                    MIN(price_usd)                                       AS min_price,
                    MAX(price_usd)                                       AS max_price,
                    ROUND(AVG(price_usd)::numeric, 0)                   AS avg_price,
                    ROUND(AVG(battery_capacity)::numeric, 0)            AS avg_battery,
                    MAX(release_year)                                    AS latest_year
                FROM   phones
                WHERE  LOWER(brand) = LOWER($1)
                GROUP  BY brand
                """,
                brand_name,
            )

            if not stats:
                return None

            latest = await conn.fetchrow(
                f"""
                SELECT {PHONE_LIST_SELECT}
                FROM   phones
                WHERE  LOWER(brand) = LOWER($1)
                ORDER  BY release_year DESC, release_month DESC NULLS LAST, id DESC
                LIMIT  1
                """,
                brand_name,
            )

        return {
            "brand":        stats["brand"],
            "total_phones": stats["total"],
            "price_range": {
                "min": float(stats["min_price"]) if stats["min_price"] else None,
                "max": float(stats["max_price"]) if stats["max_price"] else None,
                "avg": float(stats["avg_price"]) if stats["avg_price"] else None,
            },
            "avg_battery":  int(stats["avg_battery"]) if stats["avg_battery"] else None,
            "latest_year":  stats["latest_year"],
            "latest_phone": row_to_dict(latest) if latest else None,
        }

    result = await cached(f"brand:{brand_slug}", settings.cache_ttl_stable, _fetch)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Brand '{brand_slug}' not found.")
    return result


@router.get("/{brand_slug}/phones")
async def get_brand_phones(
    brand_slug: str,
    sort_by:    str = Query("release_year"),
    sort_order: str = Query("desc"),
    page:       int = Query(1,  ge=1),
    page_size:  int = Query(24, ge=1, le=100),
    # Allow all standard filters for brand pages too
    min_price:       Optional[float] = Query(None),
    max_price:       Optional[float] = Query(None),
    min_ram:         Optional[int]   = Query(None),
    min_battery:     Optional[int]   = Query(None),
    min_camera_mp:   Optional[int]   = Query(None),
    min_screen_size: Optional[float] = Query(None),
    max_screen_size: Optional[float] = Query(None),
    min_year:        Optional[int]   = Query(None),
    max_weight:      Optional[int]   = Query(None),
    min_charging_w:  Optional[int]   = Query(None),
    chipset_tier_param: Optional[str] = Query(None, alias="chipset_tier"),
):
    """Filtered, sorted, paginated brand phone list. Never cached — user-driven."""
    brand_name = slug_to_words(brand_slug)

    # Build where clause using the shared helper, locking brand
    where, params = build_search_where(
        brand=brand_name,
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

    safe_sort = sort_by if sort_by in _VALID_BRAND_SORTS else "release_year"
    sort_expr, order = resolve_sort(safe_sort, sort_order)
    offset = (page - 1) * page_size

    async with get_pool().acquire() as conn:
        total = await conn.fetchval(
            f"SELECT COUNT(*) FROM phones WHERE {where}",
            *params,
        )
        rows = await conn.fetch(
            f"""
            SELECT {PHONE_LIST_SELECT}
            FROM   phones
            WHERE  {where}
            ORDER  BY {sort_expr} {order} NULLS LAST, id DESC
            LIMIT  {page_size} OFFSET {offset}
            """,
            *params,
        )

    return {
        "total":     total,
        "page":      page,
        "page_size": page_size,
        "results":   rows_to_list(rows),
    }
