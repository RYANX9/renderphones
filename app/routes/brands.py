from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from ..core.cache import cached
from ..core.config import settings
from ..core.database import get_pool, row_to_dict, rows_to_list
from ..core.query import FilterParams, build_filter_where, resolve_sort
from ..core.shaping import attach_computed_fields, pop_smart_score
from ..core.sql_fragments import PHONE_JOIN, PHONE_LIST_SELECT

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/brands", tags=["brands"])


@router.get("")
async def list_brands():
    async def _fetch():
        async with get_pool().acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT brand, COUNT(*) AS phone_count
                FROM phones
                WHERE brand IS NOT NULL
                GROUP BY brand
                ORDER BY phone_count DESC
                """
            )
        return {"brands": [{"brand": r["brand"], "count": r["phone_count"]} for r in rows]}

    return await cached("brands:all", settings.cache_ttl_stable, _fetch)


@router.get("/{brand_name}")
async def get_brand(brand_name: str):
    brand_words = brand_name.replace("-", " ")

    async def _fetch():
        async with get_pool().acquire() as conn:
            stats = await conn.fetchrow(
                """
                SELECT
                    p.brand,
                    COUNT(*) AS total,
                    MIN(p.price_usd) AS min_price,
                    MAX(p.price_usd) AS max_price,
                    ROUND(AVG(p.price_usd)::numeric, 0) AS avg_price,
                    ROUND(AVG(s.battery_capacity)::numeric, 0) AS avg_battery,
                    MAX(p.release_year) AS latest_year
                FROM phones p
                JOIN phone_specs s ON s.phone_id = p.id
                WHERE LOWER(p.brand) = LOWER($1)
                GROUP BY p.brand
                """,
                brand_words,
            )
            if not stats:
                return None

            latest_row = await conn.fetchrow(
                f"""
                SELECT {PHONE_LIST_SELECT}
                {PHONE_JOIN}
                WHERE LOWER(p.brand) = LOWER($1)
                ORDER BY p.release_year DESC NULLS LAST,
                         p.release_month DESC NULLS LAST, p.id DESC
                LIMIT 1
                """,
                brand_words,
            )
            latest = row_to_dict(latest_row)
            if latest:
                attach_computed_fields([latest])
                latest["smart_score"] = pop_smart_score(latest)

        return {
            "brand": stats["brand"],
            "total_phones": stats["total"],
            "price_range": {
                "min": float(stats["min_price"]) if stats["min_price"] else None,
                "max": float(stats["max_price"]) if stats["max_price"] else None,
                "avg": float(stats["avg_price"]) if stats["avg_price"] else None,
            },
            "avg_battery": int(stats["avg_battery"]) if stats["avg_battery"] else None,
            "latest_year": stats["latest_year"],
            "latest_phone": latest,
        }

    result = await cached(f"brand:{brand_name}", settings.cache_ttl_stable, _fetch)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Brand '{brand_name}' not found.")
    return result


@router.get("/{brand_name}/phones")
async def get_brand_phones(
    brand_name: str,
    sort_by: str = Query("release_year"),
    sort_order: str = Query("desc"),
    page: int = Query(1, ge=1),
    page_size: int = Query(24, ge=1, le=100),
    min_price: Optional[float] = Query(None),
    max_price: Optional[float] = Query(None),
    min_ram: Optional[int] = Query(None),
    min_battery: Optional[int] = Query(None),
    min_camera_mp: Optional[int] = Query(None),
    min_screen_size: Optional[float] = Query(None),
    max_screen_size: Optional[float] = Query(None),
    min_year: Optional[int] = Query(None),
    max_weight: Optional[int] = Query(None),
    min_charging_w: Optional[int] = Query(None),
    chipset_tier: Optional[str] = Query(None),
):
    brand_words = brand_name.replace("-", " ")

    filters = FilterParams(
        brand=brand_words, min_price=min_price, max_price=max_price,
        min_ram=min_ram, min_battery=min_battery, min_camera_mp=min_camera_mp,
        min_screen_size=min_screen_size, max_screen_size=max_screen_size,
        min_year=min_year, max_weight=max_weight, min_charging_w=min_charging_w,
        chipset_tier=chipset_tier,
    )
    where, params = build_filter_where(filters)
    sort_expr, order = resolve_sort(sort_by, sort_order, has_query=False)
    offset = (page - 1) * page_size

    async with get_pool().acquire() as conn:
        total = await conn.fetchval(f"SELECT COUNT(*) {PHONE_JOIN} WHERE {where}", *params)
        i = len(params) + 1
        rows = await conn.fetch(
            f"""
            SELECT {PHONE_LIST_SELECT}
            {PHONE_JOIN}
            WHERE {where}
            ORDER BY {sort_expr} {order} NULLS LAST, p.id DESC
            LIMIT ${i} OFFSET ${i + 1}
            """,
            *params, page_size, offset,
        )

    phones = rows_to_list(rows)
    attach_computed_fields(phones)
    for p in phones:
        p["smart_score"] = pop_smart_score(p)

    return {"total": total, "page": page, "page_size": page_size, "results": phones}
