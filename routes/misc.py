from __future__ import annotations

import logging
import re
import unicodedata

from fastapi import APIRouter
from fastapi.responses import Response, JSONResponse

from cache import cached
from config import settings
from database import get_pool

logger = logging.getLogger(__name__)

router = APIRouter(tags=["system"])


def _slugify(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


@router.get("/filters/stats")
async def get_filter_stats():
    async def _fetch():
        async with get_pool().acquire() as conn:
            stats = await conn.fetchrow(
                """
                SELECT
                    COUNT(*)               AS total,
                    COUNT(DISTINCT brand)  AS total_brands,
                    MIN(price_usd)         AS min_price,
                    MAX(price_usd)         AS max_price,
                    MIN(battery_capacity)  AS min_battery,
                    MAX(battery_capacity)  AS max_battery,
                    MIN(screen_size)       AS min_screen,
                    MAX(screen_size)       AS max_screen,
                    MIN(weight_g)          AS min_weight,
                    MAX(weight_g)          AS max_weight,
                    MIN(fast_charging_w)   AS min_charging,
                    MAX(fast_charging_w)   AS max_charging,
                    MIN(release_year)      AS min_year,
                    MAX(release_year)      AS max_year
                FROM phones
                WHERE price_usd > 0
                """
            )
            brands = await conn.fetch(
                """
                SELECT brand, COUNT(*) AS count
                FROM   phones
                WHERE  brand IS NOT NULL
                GROUP  BY brand
                ORDER  BY count DESC
                """
            )
            rams = await conn.fetch(
                """
                SELECT DISTINCT unnest(ram_options) AS ram
                FROM   phones
                WHERE  ram_options IS NOT NULL
                ORDER  BY ram
                """
            )
            years = await conn.fetch(
                """
                SELECT DISTINCT release_year
                FROM   phones
                WHERE  release_year IS NOT NULL
                ORDER  BY release_year DESC
                """
            )

        return {
            "total_phones":   stats["total"],
            "total_brands":   stats["total_brands"],
            "price_range":    {"min": float(stats["min_price"]   or 0),    "max": float(stats["max_price"]   or 5000)},
            "battery_range":  {"min": int(stats["min_battery"]   or 1000),  "max": int(stats["max_battery"]   or 10000)},
            "screen_range":   {"min": float(stats["min_screen"]  or 4.0),   "max": float(stats["max_screen"]  or 7.5)},
            "weight_range":   {"min": int(stats["min_weight"]    or 100),   "max": int(stats["max_weight"]    or 300)},
            "charging_range": {"min": int(stats["min_charging"]  or 5),     "max": int(stats["max_charging"]  or 240)},
            "year_range":     {"min": int(stats["min_year"]      or 2018),  "max": int(stats["max_year"]      or 2026)},
            "brands":         [{"brand": r["brand"], "count": r["count"]} for r in brands],
            "ram_options":    [r["ram"] for r in rams if r["ram"]],
            "release_years":  [r["release_year"] for r in years],
        }

    return await cached("filters:stats", settings.cache_ttl_stable, _fetch)


@router.get("/sitemap.xml", response_class=Response)
async def generate_sitemap():
    async def _fetch():
        async with get_pool().acquire() as conn:
            rows = await conn.fetch("SELECT brand, model_name FROM phones")

        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
            '<url><loc>https://mobylite.vercel.app/</loc><priority>1.0</priority></url>',
            '<url><loc>https://mobylite.vercel.app/compare</loc><priority>0.7</priority></url>',
            '<url><loc>https://mobylite.vercel.app/pick</loc><priority>0.7</priority></url>',
        ]

        seen_brands: set[str] = set()
        for r in rows:
            brand_s = _slugify(r["brand"])
            model_s = _slugify(r["model_name"])
            if brand_s not in seen_brands:
                seen_brands.add(brand_s)
                lines.append(
                    f'<url><loc>https://mobylite.vercel.app/brand/{brand_s}</loc>'
                    f'<priority>0.6</priority></url>'
                )
            lines.append(
                f'<url><loc>https://mobylite.vercel.app/brand/{brand_s}/{model_s}</loc>'
                f'<priority>0.8</priority></url>'
            )

        lines.append("</urlset>")
        return "\n".join(lines)

    xml = await cached("sitemap", settings.cache_ttl_phone_detail, _fetch)
    return Response(content=xml, media_type="application/xml")


@router.post("/history/views")
async def record_view(data: dict):
    phone_id = data.get("phone_id")
    if phone_id:
        logger.info("view phone_id=%s", phone_id)
    return {"success": True}


@router.get("/health")
async def health():
    try:
        async with get_pool().acquire() as conn:
            await conn.fetchval("SELECT 1")
        return {"status": "ok", "db": "connected"}
    except Exception as exc:
        logger.error("Health check failed: %s", exc)
        return JSONResponse(
            status_code=503,
            content={"status": "error", "db": "unreachable"},
        )


@router.get("/")
async def root():
    return {
        "status":  "online",
        "service": "Mobylite API",
        "version": settings.app_version,
        "docs":    "/docs",
    }
