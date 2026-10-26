from __future__ import annotations

import re

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from ..core.cache import cached
from ..core.category_config import (
    CATEGORY_BRAND_CAP, CATEGORY_CONFIG, CATEGORY_GROUPS, CATEGORY_POOL, SLUG_ALIASES,
)
from ..core.config import settings
from ..core.database import acquire_conn
from ..core.insights import best_for_for_category, category_reason
from ..core.market import cap_per_brand
from ..core.seo import category_indexable
from ..core.shaping import CATEGORY_EXTRA, SCORE_KEYS, strip_fields
from ..core.slugs import category_path
from ..jobs.refresh import refresh_category
from ..services import next_steps, phone_repo
from .phones import _region

router = APIRouter(prefix="/categories", tags=["categories"])

CROWDED_THRESHOLD = 50
CATEGORY_CACHE_VERSION = "camera-date-policy-v5-lean"
FALLBACK_WEAKNESS = "No major weakness identified from the supplied specification data."


SUMMARY_MIN_CHARS = 30
SUMMARY_MAX_CHARS = 180
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def _first_sentence(text: str | None) -> str | None:
    """First sentence of the analysis reasoning, capped for a card. None when too short to be useful."""
    if not text:
        return None
    clean = " ".join(text.split())
    sentence = _SENTENCE_END.split(clean, maxsplit=1)[0].strip()
    if len(sentence) > SUMMARY_MAX_CHARS:
        cut = sentence[:SUMMARY_MAX_CHARS - 1]
        sentence = cut[:cut.rfind(" ")].rstrip(",;: ") + "\u2026" if " " in cut else cut + "\u2026"
    return sentence if len(sentence) >= SUMMARY_MIN_CHARS else None


async def _attach_summaries(conn, phones: list[dict]) -> dict[int, list[str]]:
    """One query for the listed phones: summary sentence on each, weaknesses returned by phone id."""
    if not phones:
        return {}
    rows = await conn.fetch(
        "SELECT phone_id, reasoning, weaknesses FROM phone_analysis WHERE phone_id = ANY($1::int[])",
        [p["id"] for p in phones],
    )
    by_id = {r["phone_id"]: _first_sentence(r["reasoning"]) for r in rows}
    for p in phones:
        p["summary"] = by_id.get(p["id"])
    return {r["phone_id"]: list(r["weaknesses"] or []) for r in rows}


@router.get("")
async def list_categories():
    async def fetch():
        async with acquire_conn() as conn:
            counts = await phone_repo.category_counts(conn)
        return {
            "categories": [
                {"slug": slug, "title": cfg["title"], "description": cfg["description"], "path": category_path(slug),
                 "ranked_count": counts.get(slug, 0), "indexable": category_indexable(counts.get(slug, 0))}
                for slug, cfg in CATEGORY_CONFIG.items()
            ],
            "groups": CATEGORY_GROUPS,
        }
    return await cached("categories:list", settings.cache_ttl_stable, fetch)


@router.get("/{category_slug}")
async def get_category(
    category_slug: str,
    limit: int = Query(10, ge=5, le=200),
    brand_cap: int = Query(CATEGORY_BRAND_CAP, ge=1, le=CATEGORY_BRAND_CAP, description="Maximum phones per brand."),
    status: Optional[str] = Query(None, description="Comma-separated statuses; omit for every phone."),
    region: Optional[str] = Query(None),
):
    resolved = SLUG_ALIASES.get(category_slug, category_slug)
    cfg = CATEGORY_CONFIG.get(resolved)
    if not cfg:
        raise HTTPException(status_code=404, detail=f"Category '{category_slug}' not found.")
    region = _region(region)
    statuses = [x.strip() for x in status.split(",") if x.strip()] if status else None
    pool = max(CATEGORY_POOL, limit * 3)

    async def fetch():
        async with acquire_conn() as conn:
            rows = await phone_repo.category_rankings(conn, resolved, pool, region, statuses)
            if not rows:
                await refresh_category(conn, resolved)
                rows = await phone_repo.category_rankings(conn, resolved, pool, region, statuses)
            counts = await phone_repo.category_counts(conn)
            capped = cap_per_brand(rows, brand_cap)[:limit]
            for pos, p in enumerate(capped, start=1):
                p["position"] = pos  # list position; category_rank keeps gaps where the brand cap removed phones
                p["reason"] = category_reason(resolved, p)
                if p.get("best_for"):
                    p["best_for"] = best_for_for_category(resolved, p["best_for"], p)
            for p in capped:
                strip_fields(p, SCORE_KEYS + CATEGORY_EXTRA)
            weaknesses = await _attach_summaries(conn, capped)
            buy = {str(pid): offer for pid, offer in (await phone_repo.best_buy_offers(conn, [p["id"] for p in capped])).items()}
            steps = await next_steps.for_category(conn, resolved, capped)
        top_weakness = None
        if capped:
            top_weakness = next(
                (w for w in weaknesses.get(capped[0]["id"], []) if w and w.strip() and w.strip() != FALLBACK_WEAKNESS), None,
            )
        total_ranked = counts.get(resolved, 0)
        return {
            "slug": resolved,
            "title": cfg["title"],
            "description": cfg["description"],
            "path": category_path(resolved),
            "ranked_count": total_ranked,
            "shown": len(capped),
            "indexable": category_indexable(len(capped)),
            "crowded": total_ranked >= CROWDED_THRESHOLD,
            "phones": capped,
            "buy": buy,
            "top_weakness": top_weakness,
            "next_steps": steps,
        }

    result = dict(await cached(f"category:{CATEGORY_CACHE_VERSION}:{resolved}:{limit}:{brand_cap}:{status or 'all'}:{region}", settings.cache_ttl_stable, fetch))
    # Not part of the cached value: the alias used by THIS request. The frontend
    # answers with a 301 to `path` when it is set.
    result["alias_of"] = category_slug if category_slug != resolved else None
    return result