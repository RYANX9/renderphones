from __future__ import annotations

from ..core.category_config import CANDIDATE_SQL, CATEGORY_CONFIG, CATEGORY_STORE_LIMIT, REQUIRED_MODEL_VERSION
from ..core.category_scoring import rank_category
from ..core.database import rows_to_list


async def refresh_category(conn, slug: str, pool: int | None = CATEGORY_STORE_LIMIT, rows: list[dict] | None = None) -> int:
    """Recompute one category ranking from phone_catalog and replace its rows."""
    if rows is None:
        rows = rows_to_list(await conn.fetch(CANDIDATE_SQL, REQUIRED_MODEL_VERSION))
    ranked = rank_category(slug, rows, limit=pool)
    async with conn.transaction():
        await conn.execute("DELETE FROM category_rankings WHERE category_slug = $1", slug)
        if ranked:
            await conn.executemany(
                "INSERT INTO category_rankings (category_slug, rank, phone_id, score, breakdown) VALUES ($1, $2, $3, $4, $5)",
                [(slug, i["rank"], i["phone_id"], i["category_score"], i.get("breakdown") or {}) for i in ranked],
            )
    return len(ranked)


async def refresh_all(conn) -> dict:
    """Candidates are fetched once and shared by every category."""
    rows = rows_to_list(await conn.fetch(CANDIDATE_SQL, REQUIRED_MODEL_VERSION))
    counts = {slug: await refresh_category(conn, slug, rows=rows) for slug in CATEGORY_CONFIG}
    return {"categories": counts, "candidates": len(rows)}