from __future__ import annotations

import logging
import re
import unicodedata

from fastapi import APIRouter
from fastapi.responses import Response, JSONResponse

from ..core.cache import cached
from ..core.config import settings
from ..core.database import get_pool

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
                    COUNT(*) AS total,
                    COUNT(DISTINCT p.brand) AS total_brands,
                    MIN(p.price_usd) AS min_price,
                    MAX(p.price_usd) AS max_price,
                    MIN(s.battery_capacity) AS min_battery,
                    MAX(s.battery_capacity) AS max_battery,
                    MIN(s.screen_size) AS min_screen,
                    MAX(s.screen_size) AS max_screen,
                    MIN(s.weight_g) AS min_weight,
                    MAX(s.weight_g) AS max_weight,
                    MIN(s.fast_charging_w) AS min_charging,
                    MAX(s.fast_charging_w) AS max_charging,
                    MIN(p.release_year) AS min_year,
                    MAX(p.release_year) AS max_year,
                    MIN(s.antutu_score) AS min_antutu,
                    MAX(s.antutu_score) AS max_antutu,
                    MIN(s.refresh_rate_hz) AS min_refresh,
                    MAX(s.refresh_rate_hz) AS max_refresh
                FROM phones p
                JOIN phone_specs s ON s.phone_id = p.id
                WHERE p.price_usd > 0
                """
            )
            brands = await conn.fetch(
                """
                SELECT brand, COUNT(*) AS count
                FROM phones
                WHERE brand IS NOT NULL
                GROUP BY brand
                ORDER BY count DESC
                """
            )
            rams = await conn.fetch(
                """
                SELECT DISTINCT unnest(ram_options) AS ram
                FROM phone_specs
                WHERE ram_options IS NOT NULL
                ORDER BY ram
                """
            )
            storages = await conn.fetch(
                """
                SELECT DISTINCT unnest(storage_options) AS storage
                FROM phone_specs
                WHERE storage_options IS NOT NULL
                ORDER BY storage
                """
            )
            years = await conn.fetch(
                """
                SELECT DISTINCT release_year
                FROM phones
                WHERE release_year IS NOT NULL
                ORDER BY release_year DESC
                """
            )
            chipset_tiers = await conn.fetch(
                """
                SELECT tier, COUNT(*) AS count
                FROM phone_smart_scores
                WHERE tier IS NOT NULL
                GROUP BY tier
                ORDER BY count DESC
                """
            )

        return {
            "total_phones": stats["total"],
            "total_brands": stats["total_brands"],
            "price_range": {"min": float(stats["min_price"] or 0), "max": float(stats["max_price"] or 5000)},
            "battery_range": {"min": int(stats["min_battery"] or 1000), "max": int(stats["max_battery"] or 10000)},
            "screen_range": {"min": float(stats["min_screen"] or 4.0), "max": float(stats["max_screen"] or 7.5)},
            "weight_range": {"min": int(stats["min_weight"] or 100), "max": int(stats["max_weight"] or 300)},
            "charging_range": {"min": int(stats["min_charging"] or 5), "max": int(stats["max_charging"] or 240)},
            "year_range": {"min": int(stats["min_year"] or 2018), "max": int(stats["max_year"] or 2026)},
            "antutu_range": {"min": int(stats["min_antutu"] or 0), "max": int(stats["max_antutu"] or 3_000_000)},
            "refresh_rate_range": {"min": int(stats["min_refresh"] or 60), "max": int(stats["max_refresh"] or 165)},
            "brands": [{"brand": r["brand"], "count": r["count"]} for r in brands],
            "ram_options": [r["ram"] for r in rams if r["ram"]],
            "storage_options": [r["storage"] for r in storages if r["storage"]],
            "release_years": [r["release_year"] for r in years],
            "chipset_tiers": [{"tier": r["tier"], "count": r["count"]} for r in chipset_tiers],
        }

    return await cached("filters:stats", settings.cache_ttl_stable, _fetch)


@router.get("/sitemap.xml", response_class=Response)
async def generate_sitemap():
    async def _fetch():
        async with get_pool().acquire() as conn:
            rows = await conn.fetch("SELECT brand, model_name, slug FROM phones")

        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
            '<url><loc>https://specmob.vercel.app/</loc><priority>1.0</priority></url>',
            '<url><loc>https://specmob.vercel.app/compare</loc><priority>0.7</priority></url>',
            '<url><loc>https://specmob.vercel.app/pick</loc><priority>0.7</priority></url>',
        ]

        seen_brands: set[str] = set()
        for r in rows:
            brand_s = _slugify(r["brand"])
            phone_s = r["slug"] or _slugify(r["model_name"])
            if brand_s not in seen_brands:
                seen_brands.add(brand_s)
                lines.append(f'<url><loc>https://specmob.vercel.app/brand/{brand_s}</loc><priority>0.6</priority></url>')
            lines.append(f'<url><loc>https://specmob.vercel.app/brand/{brand_s}/{phone_s}</loc><priority>0.8</priority></url>')

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
        return JSONResponse(status_code=503, content={"status": "error", "db": "unreachable"})


@router.get("/")
async def root():
    return {
        "status": "online",
        "service": "Specmob API",
        "version": settings.app_version,
        "docs": "/docs",
    }
