"""Aggregated reads for whole pages, so a page needs ONE server call.

Every block carries a `slot` id; the frontend sends it back as `block_click`
so analytics can say which blocks earn their place. Block order and which
blocks exist is decided by `HOME_BLOCKS` below, so the home page can change
without touching components.
"""
from __future__ import annotations

import asyncio
import time
from typing import Optional

from fastapi import APIRouter, Query

from ..core.cache import cached
from ..core.category_config import CATEGORY_CONFIG, CATEGORY_GROUPS
from ..core.config import settings
from ..core.database import acquire_conn
from ..core.seo import category_indexable
from ..core.slugs import category_path
from ..services import next_steps, phone_repo, stats_repo
from .phones import _region

router = APIRouter(prefix="/discover", tags=["discover"])

HOME_BLOCKS: list[str] = ["categories", "most_compared", "most_viewed", "coming_soon", "brands"]
MOST_VIEWED_REFRESH_S = 5 * 86400


async def _category_tiles(conn, region: Optional[str], limit: int | None = None) -> list[dict]:
    counts = await phone_repo.category_counts(conn)
    tops = await phone_repo.category_tops(conn, 1, region)
    tiles = []
    for slug, cfg in CATEGORY_CONFIG.items():
        top = tops.get(slug)
        if not top:
            continue
        tiles.append({
            "slug": slug, "title": cfg["title"], "path": category_path(slug),
            "ranked_count": counts.get(slug, 0), "indexable": category_indexable(counts.get(slug, 0)),
            "top_phone": top[0],
        })
        if limit and len(tiles) >= limit:
            break
    return tiles


async def _most_viewed(region: Optional[str]) -> list[dict]:
    """Stable for 5 days: the list is rebuilt once per 5-day bucket, not on every
    home cache refresh, so it does not shuffle while people browse."""
    bucket = int(time.time() // MOST_VIEWED_REFRESH_S)

    async def fetch():
        async with acquire_conn() as conn:
            return await phone_repo.most_viewed(conn, 8, region)

    return await cached(f"discover:most-viewed:{bucket}:{region}", MOST_VIEWED_REFRESH_S, fetch)


async def _brands() -> list[dict]:
    brands = await phone_repo.cached_brand_list()
    return [{"brand": b["brand"], "brand_slug": b["brand_slug"], "path": b["path"], "phone_count": b["phone_count"]}
            for b in brands[:12]]


async def _block(block_id: str, region: Optional[str]) -> list[dict]:
    if block_id == "most_viewed":
        return await _most_viewed(region)
    if block_id == "brands":
        return await _brands()
    async with acquire_conn() as conn:
        if block_id == "categories":
            return await _category_tiles(conn, region, 6)
        if block_id == "most_compared":
            return await stats_repo.most_compared(conn, 6, settings.compare_display_min_sessions)
        if block_id == "coming_soon":
            return await phone_repo.upcoming_by_stage(conn, 8)
    return []


@router.get("/home")
async def home(region: Optional[str] = Query(None)):
    region = _region(region)

    async def fetch():
        results = await asyncio.gather(*(_block(b, region) for b in HOME_BLOCKS))
        return {"blocks": [
            {"id": b, "slot": f"home.{b}", "items": items} for b, items in zip(HOME_BLOCKS, results) if items
        ]}

    return await cached(f"discover:home:v4:{region}", settings.cache_ttl_trending, fetch)


@router.get("/best-hub")
async def best_hub(region: Optional[str] = Query(None)):
    region = _region(region)

    async def fetch():
        async with acquire_conn() as conn:
            tiles = {t["slug"]: t for t in await _category_tiles(conn, region)}
        groups = []
        for g in CATEGORY_GROUPS:
            items = [tiles[s] for s in g["slugs"] if s in tiles]
            if items:
                groups.append({"id": g["id"], "title": g["title"], "categories": items})
        steps = [
            next_steps.step("hub.pick", "pick", "Help me pick a phone", "/pick"),
            next_steps.step("hub.compare", "compare", "Compare two phones", "/compare"),
            next_steps.step("hub.brands", "hub", "Browse by brand", "/brands"),
        ]
        return {"groups": groups, "next_steps": steps}

    return await cached(f"discover:hub:v2:{region}", settings.cache_ttl_stable, fetch)


@router.get("/compare-landing")
async def compare_landing():
    async def fetch():
        async with acquire_conn() as conn:
            popular = await stats_repo.most_compared(conn, 9, settings.compare_display_min_sessions)
            tops = await phone_repo.category_tops(conn, 2, None)
        matchups = []
        for slug, cfg in CATEGORY_CONFIG.items():
            top = tops.get(slug, [])
            if len(top) >= 2:
                pair = sorted([top[0]["full_slug"], top[1]["full_slug"]])
                matchups.append({
                    "category": {"slug": slug, "title": cfg["title"]},
                    "path": f"/compare/{'-vs-'.join(pair)}",
                    "phones": top[:2],
                })
        return {
            "most_compared": popular, "category_matchups": matchups,
            "next_steps": [next_steps.step("compare_landing.pick", "pick", "Not sure what to compare? Help me pick", "/pick"),
                           next_steps.step("compare_landing.hub", "hub", "Browse best-of lists", next_steps.BEST_HUB)],
        }
    return await cached("discover:compare-landing:v2", settings.cache_ttl_trending, fetch)
