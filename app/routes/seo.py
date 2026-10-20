"""Data for the Next.js sitemap (the only sitemap; the backend no longer emits
one). Chunked so you never hit the 50,000-URL limit. `lastmod` is a real
timestamp (never now()); phones/brands use updated_at, which you should bump
only on meaningful changes."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ..core.cache import cached
from ..core.category_config import CATEGORY_CONFIG
from ..core.config import settings
from ..core.database import acquire_conn
from ..core.seo import brand_indexable, category_indexable
from ..core.slugs import brand_path, category_path, compare_path, phone_path
from ..core.sql_fragments import LISTABLE, NOT_REGIONAL
from ..services import stats_repo

router = APIRouter(prefix="/seo", tags=["seo"])
SECTIONS = ("phones", "brands", "categories", "compares")


@router.get("/sitemap")
async def sitemap_data(section: str = Query(...), page: int = Query(1, ge=1), page_size: int = Query(5000, ge=1, le=50000)):
    if section not in SECTIONS:
        raise HTTPException(status_code=404, detail=f"section must be one of {SECTIONS}")

    async def fetch():
        async with acquire_conn() as conn:
            if section == "phones":
                total = await conn.fetchval(f"SELECT count(*) FROM phone_catalog p WHERE {NOT_REGIONAL} AND (p.status <> 'discontinued' OR p.status IS NULL)")
                rows = await conn.fetch(
                    f"""SELECT p.brand_slug, p.slug, GREATEST(p.updated_at, p.price_updated_at, p.scored_at) AS lastmod, p.main_image_url
FROM phone_catalog p WHERE {NOT_REGIONAL} AND (p.status <> 'discontinued' OR p.status IS NULL)
ORDER BY p.id LIMIT $1 OFFSET $2""", page_size, (page - 1) * page_size)
                entries = [{"path": phone_path(r["brand_slug"], r["slug"]), "lastmod": r["lastmod"], "image": r["main_image_url"]} for r in rows]
            elif section == "brands":
                rows = await conn.fetch(
                    f"""SELECT p.brand_slug, count(*)::int AS n, max(p.updated_at) AS lastmod
FROM phone_catalog p WHERE {LISTABLE} AND {NOT_REGIONAL} GROUP BY p.brand_slug ORDER BY p.brand_slug""")
                data = [r for r in rows if brand_indexable(r["n"])]
                total = len(data)
                entries = [{"path": brand_path(r["brand_slug"]), "lastmod": r["lastmod"]}
                           for r in data[(page - 1) * page_size: page * page_size]]
            elif section == "categories":
                rows = await conn.fetch(
                    """SELECT r.category_slug AS slug, count(*)::int AS n, max(p.updated_at) AS lastmod
FROM category_rankings r JOIN phones p ON p.id = r.phone_id GROUP BY r.category_slug""")
                data = [r for r in rows if r["slug"] in CATEGORY_CONFIG and category_indexable(r["n"])]
                total = len(data)
                entries = [{"path": category_path(r["slug"]), "lastmod": r["lastmod"]} for r in data]
            else:
                curated = await stats_repo.curated_pair_rows(conn)
                popular = await conn.fetch(
                    "SELECT phone_a, phone_b FROM compare_pair_stats WHERE uniq_30d >= $1", settings.compare_popular_min_sessions)
                pairs = {(r["phone_a"], r["phone_b"]) for r in curated} | {(r["phone_a"], r["phone_b"]) for r in popular}
                ids = sorted({i for p in pairs for i in p})
                slugs = {r["id"]: (r["full_slug"], r["updated_at"]) for r in await conn.fetch(
                    "SELECT id, full_slug, updated_at FROM phones WHERE id = ANY($1::int[])", ids)}
                entries_all = []
                for a, b in sorted(pairs):
                    if a in slugs and b in slugs:
                        entries_all.append({"path": compare_path([slugs[a][0], slugs[b][0]]),
                                            "lastmod": max(slugs[a][1], slugs[b][1])})
                total = len(entries_all)
                entries = entries_all[(page - 1) * page_size: page * page_size]
        return {"section": section, "page": page, "page_size": page_size, "total": total,
                "pages": max(1, -(-total // page_size)), "entries": entries}

    return await cached(f"seo:{section}:{page}:{page_size}", 3600, fetch)
