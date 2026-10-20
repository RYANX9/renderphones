"""Aggregated reads for whole pages, so a page needs ONE server call.

Every block carries a `slot` id; the frontend sends it back as `block_click`
so analytics can say which blocks earn their place. Block order and which
blocks exist is decided by `HOME_BLOCKS` below, so the home page can change
without touching components.
"""
from __future__ import annotations

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


async def _category_tiles(conn, region: Optional[str], limit: int | None = None) -> list[dict]:
    counts = await phone_repo.category_counts(conn)
    tiles = []
    for slug, cfg in CATEGORY_CONFIG.items():
        top = await phone_repo.category_rankings(conn, slug, 1, region)
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


@router.get("/home")
async def home(region: Optional[str] = Query(None)):
    region = _region(region)

    async def fetch():
        async with acquire_conn() as conn:
            builders = {
                "categories": lambda: _category_tiles(conn, region, 6),
                "most_compared": lambda: stats_repo.most_compared(conn, 6, settings.compare_display_min_sessions),
                "most_viewed": lambda: _most_viewed(region),
                "coming_soon": lambda: phone_repo.upcoming_by_stage(conn, 8),
                "brands": lambda: _brands(conn),
            }
            blocks = []
            for block_id in HOME_BLOCKS:
                items = await builders[block_id]()
                if items:
                    blocks.append({"id": block_id, "slot": f"home.{block_id}", "items": items})
        return {"blocks": blocks}

    return await cached(f"discover:home:v3:{region}", settings.cache_ttl_trending, fetch)


MOST_VIEWED_REFRESH_S = 5 * 86400


async def _most_viewed(region: Optional[str]) -> list[dict]:
    """Stable for 5 days: the list is rebuilt once per 5-day bucket, not on every
    home cache refresh, so it does not shuffle while people browse."""
    bucket = int(time.time() // MOST_VIEWED_REFRESH_S)

    async def fetch():
        async with acquire_conn() as conn:
            return await phone_repo.most_viewed(conn, 8, region)

    return await cached(f"discover:most-viewed:{bucket}:{region}", MOST_VIEWED_REFRESH_S, fetch)


async def _brands(conn) -> list[dict]:
    brands = await phone_repo.brand_list(conn)
    return [{"brand": b["brand"], "brand_slug": b["brand_slug"], "path": b["path"], "phone_count": b["phone_count"]}
            for b in brands[:12]]


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

    return await cached(f"discover:hub:{region}", settings.cache_ttl_stable, fetch)


@router.get("/compare-landing")
async def compare_landing():
    async def fetch():
        async with acquire_conn() as conn:
            popular = await stats_repo.most_compared(conn, 9, settings.compare_display_min_sessions)
            matchups = []
            for slug, cfg in CATEGORY_CONFIG.items():
                top = await phone_repo.category_rankings(conn, slug, 2, None)
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
    return await cached("discover:compare-landing", settings.cache_ttl_trending, fetch)
