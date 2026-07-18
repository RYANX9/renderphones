from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

from ..core.cache import cached
from ..core.config import settings
from ..core.database import get_pool, row_to_dict, rows_to_list
from ..core.sql_fragments import RELEASE_TS_EXPR

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/categories", tags=["categories"])

_CATEGORY_COLS = f"""
p.id, p.model_name, p.brand, p.slug, p.main_image_url,
p.release_year, p.price_usd, p.availability_status, p.amazon_link, p.popularity,
{RELEASE_TS_EXPR} AS release_ts,

s.screen_size, s.battery_capacity, s.fast_charging_w, s.main_camera_mp,
s.chipset, s.antutu_score, s.has_ois, s.optical_zoom, s.weight_g,
s.is_premium_gaming, s.ram_options, s.storage_options,

sc.tier AS smart_tier,
sc.overall_score AS smart_overall_score,
sc.camera_score AS smart_camera_score,
sc.performance_score AS smart_performance_score,
sc.battery_score AS smart_battery_score,
sc.display_score AS smart_display_score,
sc.build_score AS smart_build_score,
sc.value_score AS smart_value_score,
sc.strengths AS smart_strengths,
sc.weaknesses AS smart_weaknesses,
sc.reasoning AS smart_reasoning,
sc.model_version AS smart_model_version,
sc.scored_at AS smart_scored_at
"""

_SMART_KEYS = (
    "smart_camera_score", "smart_performance_score", "smart_battery_score",
    "smart_display_score", "smart_build_score", "smart_value_score",
    "smart_strengths", "smart_weaknesses", "smart_reasoning",
    "smart_model_version", "smart_scored_at",
)


def _pop_smart_score(d: dict) -> dict | None:
    has_score = d.get("smart_overall_score") is not None
    out = None
    if has_score:
        out = {
            "overall_score": d.get("smart_overall_score"),
            "camera_score": d.get("smart_camera_score"),
            "performance_score": d.get("smart_performance_score"),
            "battery_score": d.get("smart_battery_score"),
            "display_score": d.get("smart_display_score"),
            "build_score": d.get("smart_build_score"),
            "value_score": d.get("smart_value_score"),
            "strengths": d.get("smart_strengths"),
            "weaknesses": d.get("smart_weaknesses"),
            "reasoning": d.get("smart_reasoning"),
            "tier": d.get("smart_tier"),
            "model_version": d.get("smart_model_version"),
            "scored_at": d.get("smart_scored_at"),
        }
    for k in _SMART_KEYS:
        d.pop(k, None)
    return out


def _category_sql(smart_expr: str, legacy_score_expr: str, where_clause: str) -> str:
    return f"""
        WITH candidates AS (
            SELECT {{cols}},
                   ({smart_expr}) AS smart_metric,
                   ({legacy_score_expr}) AS legacy_raw_score
            FROM phones p
            JOIN phone_specs s ON s.phone_id = p.id
            LEFT JOIN phone_smart_scores sc ON sc.phone_id = p.id
            WHERE {where_clause}
        ),
        normalized AS (
            SELECT *,
                   COALESCE(
                       smart_metric,
                       10.0 * legacy_raw_score / NULLIF(MAX(legacy_raw_score) OVER (), 0)
                   ) AS blended_score
            FROM candidates
        ),
        top_n AS (
            SELECT * FROM normalized
            ORDER BY blended_score DESC NULLS LAST
            LIMIT {{limit}}
        )
        SELECT *,
               10.0 * blended_score / NULLIF(MAX(blended_score) OVER (), 0) AS category_score
        FROM top_n
        ORDER BY blended_score DESC NULLS LAST
    """


CATEGORY_CONFIG: dict[str, dict] = {
    "camera-phones": {
        "title": "Best Camera Phones",
        "description": (
            "Ranked by our smart camera score where available (sensor quality, "
            "OIS, versatility, real-world imaging), falling back to a spec "
            "composite of resolution, chip performance, and charging speed "
            "for unscored phones."
        ),
        "sql": _category_sql(
            smart_expr="sc.camera_score",
            legacy_score_expr="""
                COALESCE(s.main_camera_mp, 0)::float * 0.40
                + COALESCE(s.antutu_score, 0)::float / 200000.0 * 2.0
                + COALESCE(s.fast_charging_w, 0)::float * 0.05
                + CASE WHEN s.has_ois THEN 3.0 ELSE 0 END
            """,
            where_clause=(
                "s.main_camera_mp IS NOT NULL "
                "AND s.screen_size >= 5.5 "
                "AND p.release_year >= 2023"
            ),
        ),
    },
    "battery-life": {
        "title": "Best Battery Life",
        "description": (
            "Ranked by our smart battery score where available (capacity, "
            "efficiency, real endurance), falling back to raw battery "
            "capacity for unscored phones."
        ),
        "sql": _category_sql(
            smart_expr="sc.battery_score",
            legacy_score_expr="COALESCE(s.battery_capacity, 0)::float",
            where_clause=(
                "s.battery_capacity IS NOT NULL "
                "AND s.screen_size >= 5.5 "
                "AND p.release_year >= 2023"
            ),
        ),
    },
    "gaming-phones": {
        "title": "Best Gaming Phones",
        "description": (
            "Ranked by our smart performance score where available, falling "
            "back to raw AnTuTu benchmark for unscored phones. 2024+ releases only, "
            "phones flagged as premium gaming hardware get a boost."
        ),
        "sql": _category_sql(
            smart_expr="sc.performance_score",
            legacy_score_expr=(
                "COALESCE(s.antutu_score, 0)::float"
                " + CASE WHEN s.is_premium_gaming THEN 200000 ELSE 0 END"
            ),
            where_clause=(
                "s.antutu_score IS NOT NULL "
                "AND s.screen_size >= 5.5 "
                "AND p.release_year >= 2024"
            ),
        ),
    },
    "under-300": {
        "title": "Best Phones Under $300",
        "description": (
            "Ranked by our smart value score where available, falling back "
            "to a specs-per-dollar composite for unscored phones."
        ),
        "sql": _category_sql(
            smart_expr="sc.value_score",
            legacy_score_expr="""
                COALESCE(s.battery_capacity, 0)::float / 500.0
                + COALESCE(s.main_camera_mp, 0)::float / 10.0
                + COALESCE(s.antutu_score, 0)::float / 100000.0
            """,
            where_clause=(
                "p.price_usd <= 300 AND p.price_usd > 0 "
                "AND s.screen_size >= 5.5 AND p.release_year >= 2022"
            ),
        ),
    },
    "under-500": {
        "title": "Best Phones Under $500",
        "description": (
            "Ranked by our smart value score where available, falling back "
            "to a specs-per-dollar composite for unscored phones."
        ),
        "sql": _category_sql(
            smart_expr="sc.value_score",
            legacy_score_expr="""
                COALESCE(s.battery_capacity, 0)::float / 500.0
                + COALESCE(s.main_camera_mp, 0)::float / 10.0
                + COALESCE(s.antutu_score, 0)::float / 100000.0
            """,
            where_clause=(
                "p.price_usd <= 500 AND p.price_usd > 0 "
                "AND s.screen_size >= 5.5 AND p.release_year >= 2022"
            ),
        ),
    },
    "lightweight": {
        "title": "Lightest Smartphones",
        "description": (
            "Modern smartphones (5.5\"+ screen) between 100g-185g. Purely a "
            "physical measurement — smart score does not apply here."
        ),
        "sql": _category_sql(
            smart_expr="NULL::numeric",
            legacy_score_expr="(200.0 - COALESCE(s.weight_g, 200))::float",
            where_clause=(
                "s.weight_g IS NOT NULL AND s.weight_g BETWEEN 100 AND 185 "
                "AND s.screen_size >= 5.5 AND p.release_year >= 2023"
            ),
        ),
    },
    "compact-phones": {
        "title": "Best Compact Phones",
        "description": (
            "Screens between 5.0\"-6.3\". Ranked by our smart performance "
            "score where available within the compact segment, falling back "
            "to raw AnTuTu."
        ),
        "sql": _category_sql(
            smart_expr="sc.performance_score",
            legacy_score_expr="COALESCE(s.antutu_score, 0)::float",
            where_clause="s.screen_size <= 6.3 AND s.screen_size >= 5.0 AND p.release_year >= 2023",
        ),
    },
    "fast-charging": {
        "title": "Fastest Charging Phones",
        "description": (
            "Ranked by maximum wired charging wattage. 30W minimum. Purely a "
            "physical measurement — smart score does not apply here."
        ),
        "sql": _category_sql(
            smart_expr="NULL::numeric",
            legacy_score_expr="COALESCE(s.fast_charging_w, 0)::float",
            where_clause=(
                "s.fast_charging_w >= 30 AND s.screen_size >= 5.5 AND p.release_year >= 2023"
            ),
        ),
    },
    "foldables": {
        "title": "Best Foldable Phones",
        "description": "Ranked by smart overall score, restricted to foldable form factors.",
        "sql": _category_sql(
            smart_expr="sc.overall_score",
            legacy_score_expr="COALESCE(s.antutu_score, 0)::float",
            where_clause="s.is_foldable IS TRUE",
        ),
    },
    "water-resistant": {
        "title": "Best Water/Dust Resistant Phones",
        "description": "IP68/IP67-rated phones ranked by smart overall score.",
        "sql": _category_sql(
            smart_expr="sc.overall_score",
            legacy_score_expr="COALESCE(s.antutu_score, 0)::float",
            where_clause="s.water_resistance ILIKE '%IP6%' OR s.water_resistance ILIKE '%IP5%'",
        ),
    },
}

_SLUG_ALIASES: dict[str, str] = {
    "camera": "camera-phones",
    "battery": "battery-life",
    "gaming": "gaming-phones",
    "lightweight": "lightweight",
    "compact": "compact-phones",
    "charging": "fast-charging",
    "foldable": "foldables",
    "waterproof": "water-resistant",
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
async def get_category(category_slug: str, limit: int = Query(10, ge=5, le=20)):
    resolved = _SLUG_ALIASES.get(category_slug, category_slug)
    cfg = CATEGORY_CONFIG.get(resolved)
    if not cfg:
        raise HTTPException(status_code=404, detail=f"Category '{category_slug}' not found.")

    async def _fetch():
        sql = cfg["sql"].format(cols=_CATEGORY_COLS, limit=limit)
        async with get_pool().acquire() as conn:
            rows = await conn.fetch(sql)

        phones = []
        for r in rows:
            d = row_to_dict(r)
            d.pop("smart_metric", None)
            d.pop("legacy_raw_score", None)
            d.pop("blended_score", None)
            raw_cs = d.pop("category_score", 0) or 0
            d["category_score"] = round(float(raw_cs), 2)
            d["smart_score"] = _pop_smart_score(d)
            phones.append(d)

        return {"slug": resolved, "title": cfg["title"], "description": cfg["description"], "phones": phones}

    return await cached(f"category:{resolved}:{limit}", settings.cache_ttl_stable, _fetch)
