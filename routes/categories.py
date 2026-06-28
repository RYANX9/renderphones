from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

from cache import cached
from config import settings
from database import get_pool, row_to_dict, rows_to_list, PHONE_LIST_SELECT

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/categories", tags=["categories"])


def _category_sql(score_expr: str, where_clause: str) -> str:
    """
    Builds a CTE template string with two deferred placeholders:
      {cols}  — filled with PHONE_LIST_SELECT at request time
      {limit} — filled with the limit param at request time

    score_expr and where_clause are baked in at definition time (f-string).
    {{cols}} and {{limit}} survive as literal {cols}/{limit} for .format().
    """
    return f"""
        WITH base AS (
            SELECT {{cols}},
                   ({score_expr}) AS raw_score
            FROM   phones
            WHERE  {where_clause}
        ),
        top_n AS (
            SELECT *
            FROM   base
            ORDER  BY raw_score DESC
            LIMIT  {{limit}}
        )
        SELECT *,
               10.0 * raw_score / NULLIF(MAX(raw_score) OVER (), 0) AS category_score
        FROM   top_n
        ORDER  BY raw_score DESC
    """


CATEGORY_CONFIG: dict[str, dict] = {
    "camera-phones": {
        "title": "Best Camera Phones",
        "description": (
            "Multi-factor ranking: main sensor resolution (40%), "
            "AI chip performance (40%), and fast charging (20%)."
        ),
        "sql": _category_sql(
            score_expr="""
                COALESCE(main_camera_mp, 0)::float * 0.40
                + COALESCE(antutu_score, 0)::float / 200000.0 * 2.0
                + COALESCE(fast_charging_w, 0)::float * 0.05
            """,
            where_clause=(
                "main_camera_mp IS NOT NULL "
                "AND screen_size >= 5.5 "
                "AND release_year >= 2023"
            ),
        ),
    },
    "battery-life": {
        "title": "Best Battery Life",
        "description": "Ranked by raw battery capacity. 5000 mAh+ on efficient modern chips dominate.",
        "sql": _category_sql(
            score_expr="COALESCE(battery_capacity, 0)::float",
            where_clause=(
                "battery_capacity IS NOT NULL "
                "AND screen_size >= 5.5 "
                "AND release_year >= 2023"
            ),
        ),
    },
    "gaming-phones": {
        "title": "Best Gaming Phones",
        "description": (
            "Top AnTuTu benchmark scores from 2024 and newer. "
            "Snapdragon 8 Elite and Dimensity 9400+ class chips only."
        ),
        "sql": _category_sql(
            score_expr="COALESCE(antutu_score, 0)::float",
            where_clause=(
                "antutu_score IS NOT NULL "
                "AND screen_size >= 5.5 "
                "AND release_year >= 2024"
            ),
        ),
    },
    "under-300": {
        "title": "Best Phones Under $300",
        "description": "Maximum specs per dollar under $300. Composite of battery, camera, and performance.",
        "sql": _category_sql(
            score_expr="""
                COALESCE(battery_capacity, 0)::float / 500.0
                + COALESCE(main_camera_mp, 0)::float / 10.0
                + COALESCE(antutu_score, 0)::float / 100000.0
            """,
            where_clause=(
                "price_usd <= 300 "
                "AND price_usd > 0 "
                "AND screen_size >= 5.5 "
                "AND release_year >= 2022"
            ),
        ),
    },
    "under-500": {
        "title": "Best Phones Under $500",
        "description": "The mid-range sweet spot. Near-flagship specs at half the price.",
        "sql": _category_sql(
            score_expr="""
                COALESCE(battery_capacity, 0)::float / 500.0
                + COALESCE(main_camera_mp, 0)::float / 10.0
                + COALESCE(antutu_score, 0)::float / 100000.0
            """,
            where_clause=(
                "price_usd <= 500 "
                "AND price_usd > 0 "
                "AND screen_size >= 5.5 "
                "AND release_year >= 2022"
            ),
        ),
    },
    "lightweight": {
        "title": "Lightest Smartphones",
        "description": "Modern smartphones (5.5\"+ screen) between 100 g–185 g. Feature phones excluded.",
        "sql": _category_sql(
            score_expr="(200.0 - COALESCE(weight_g, 200))::float",
            where_clause=(
                "weight_g IS NOT NULL "
                "AND weight_g BETWEEN 100 AND 185 "
                "AND screen_size >= 5.5 "
                "AND release_year >= 2023"
            ),
        ),
    },
    "compact-phones": {
        "title": "Best Compact Phones",
        "description": "Smartphones with screens between 5.0\"–6.3\". Ranked by AnTuTu performance.",
        "sql": _category_sql(
            score_expr="COALESCE(antutu_score, 0)::float",
            where_clause=(
                "screen_size <= 6.3 "
                "AND screen_size >= 5.0 "
                "AND release_year >= 2023"
            ),
        ),
    },
    "fast-charging": {
        "title": "Fastest Charging Phones",
        "description": "Ranked by maximum wired charging wattage. 30 W minimum. 90 W+ is the 2026 premium benchmark.",
        "sql": _category_sql(
            score_expr="COALESCE(fast_charging_w, 0)::float",
            where_clause=(
                "fast_charging_w >= 30 "
                "AND screen_size >= 5.5 "
                "AND release_year >= 2023"
            ),
        ),
    },
}

_SLUG_ALIASES: dict[str, str] = {
    "camera":      "camera-phones",
    "battery":     "battery-life",
    "gaming":      "gaming-phones",
    "lightweight": "lightweight",
    "compact":     "compact-phones",
    "charging":    "fast-charging",
}


@router.get("")
async def list_categories():
    async def _fetch():
        return {
            "categories": [
                {"slug": slug, "title": cfg["title"], "description": cfg["description"]}
                for slug, cfg in CATEGORY_CONFIG.items()
            ]
        }

    return await cached("categories:list", settings.cache_ttl_stable, _fetch)


@router.get("/{category_slug}")
async def get_category(
    category_slug: str,
    limit: int = Query(10, ge=5, le=20),
):
    resolved = _SLUG_ALIASES.get(category_slug, category_slug)
    cfg = CATEGORY_CONFIG.get(resolved)
    if not cfg:
        raise HTTPException(status_code=404, detail=f"Category '{category_slug}' not found.")

    async def _fetch():
        sql = cfg["sql"].format(cols=PHONE_LIST_SELECT, limit=limit)
        async with get_pool().acquire() as conn:
            rows = await conn.fetch(sql)

        phones = []
        for r in rows:
            d = row_to_dict(r)
            d.pop("raw_score", None)
            raw_cs = d.pop("category_score", 0) or 0
            d["category_score"] = round(float(raw_cs), 2)
            phones.append(d)

        return {
            "slug":        resolved,
            "title":       cfg["title"],
            "description": cfg["description"],
            "phones":      phones,
        }

    return await cached(
        f"category:{resolved}:{limit}",
        settings.cache_ttl_stable,
        _fetch,
    )
