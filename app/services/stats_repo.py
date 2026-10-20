"""Reads over the tracking rollups (compare_pair_stats, phone_popularity) and
curated compare pairs. Request handlers never write these tables; the jobs in
app/jobs/rollups.py do."""
from __future__ import annotations

from typing import Any

import asyncpg

from ..core.database import rows_to_list
from ..core.slugs import compare_path, pair_slug
from ..core.sql_fragments import LISTABLE, NOT_REGIONAL
from . import phone_repo


def _pair_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


async def _attach(conn: asyncpg.Connection, rows: list[dict], **extra: Any) -> list[dict]:
    minis = await phone_repo.minis_by_ids(conn, [i for r in rows for i in (r["phone_a"], r["phone_b"])])
    out = []
    for r in rows:
        a, b = minis.get(r["phone_a"]), minis.get(r["phone_b"])
        if not a or not b:
            continue
        slugs = [a["full_slug"], b["full_slug"]]
        out.append({
            "pair_slug": pair_slug(slugs),
            "path": compare_path(slugs),
            "phones": sorted([a, b], key=lambda m: m["full_slug"]),
            "sessions_30d": r.get("uniq_30d"),
            **extra,
        })
    return out


async def top_pairs(conn: asyncpg.Connection, limit: int, min_sessions: int) -> list[dict]:
    rows = await conn.fetch(
        f"""SELECT s.phone_a, s.phone_b, s.uniq_30d, s.uniq_7d
FROM compare_pair_stats s
JOIN phone_catalog a ON a.id = s.phone_a AND {LISTABLE.replace('p.', 'a.')}
JOIN phone_catalog b ON b.id = s.phone_b AND {LISTABLE.replace('p.', 'b.')}
WHERE s.uniq_30d >= $2
ORDER BY s.uniq_30d DESC, s.uniq_7d DESC LIMIT $1""",
        limit, min_sessions,
    )
    return await _attach(conn, rows_to_list(rows), source="popular")


async def pairs_for_phone(conn: asyncpg.Connection, phone_id: int, limit: int, min_sessions: int = 1) -> list[dict]:
    rows = await conn.fetch(
        f"""SELECT s.phone_a, s.phone_b, s.uniq_30d, s.uniq_7d
FROM compare_pair_stats s
JOIN phone_catalog o ON o.id = CASE WHEN s.phone_a = $1 THEN s.phone_b ELSE s.phone_a END
  AND {LISTABLE.replace('p.', 'o.')}
WHERE (s.phone_a = $1 OR s.phone_b = $1) AND s.uniq_30d >= $3
ORDER BY s.uniq_30d DESC, s.uniq_7d DESC LIMIT $2""",
        phone_id, limit, min_sessions,
    )
    return await _attach(conn, rows_to_list(rows), source="popular")


async def pairs_for_phones(conn: asyncpg.Connection, phone_ids: list[int], exclude: tuple[int, int] | None,
                           limit: int, min_sessions: int = 1) -> list[dict]:
    """Most-compared pairs that contain any of these phones, excluding one pair."""
    rows = await conn.fetch(
        f"""SELECT s.phone_a, s.phone_b, s.uniq_30d, s.uniq_7d
FROM compare_pair_stats s
JOIN phone_catalog a ON a.id = s.phone_a AND {LISTABLE.replace('p.', 'a.')}
JOIN phone_catalog b ON b.id = s.phone_b AND {LISTABLE.replace('p.', 'b.')}
WHERE (s.phone_a = ANY($1::int[]) OR s.phone_b = ANY($1::int[])) AND s.uniq_30d >= $3
ORDER BY s.uniq_30d DESC, s.uniq_7d DESC LIMIT $2""",
        phone_ids, limit + 1, min_sessions,
    )
    data = [r for r in rows_to_list(rows) if exclude is None or _pair_key(r["phone_a"], r["phone_b"]) != exclude]
    return await _attach(conn, data[:limit], source="popular")


async def curated_pair_rows(conn: asyncpg.Connection, top_n: int = 3) -> list[dict]:
    """Pairs among the top-N phones of each category (rank 1 vs 2, 1 vs 3, 2 vs 3)."""
    rows = await conn.fetch(
        """SELECT DISTINCT LEAST(a.phone_id, b.phone_id) AS phone_a, GREATEST(a.phone_id, b.phone_id) AS phone_b
FROM category_rankings a JOIN category_rankings b
  ON a.category_slug = b.category_slug AND a.rank < b.rank
WHERE b.rank <= $1 AND a.phone_id <> b.phone_id""",
        top_n,
    )
    return rows_to_list(rows)


async def curated_pairs(conn: asyncpg.Connection, limit: int) -> list[dict]:
    rows = await curated_pair_rows(conn)
    rows = [{**r, "uniq_30d": None} for r in rows]
    return (await _attach(conn, rows, source="curated"))[:limit]


async def same_tier_pairs(conn: asyncpg.Connection, limit: int) -> list[dict]:
    """Fallback when there is no compare data yet: adjacent top-rated phones inside
    the same tier, so 'Most compared' is never an empty block."""
    rows = await conn.fetch(
        f"""SELECT p.id, p.tier FROM phone_catalog p
WHERE {LISTABLE} AND {NOT_REGIONAL} AND p.overall_score IS NOT NULL AND p.tier IS NOT NULL
ORDER BY p.overall_score DESC, p.id LIMIT 60"""
    )
    by_tier: dict[str, list[int]] = {}
    for r in rows:
        by_tier.setdefault(r["tier"], []).append(r["id"])
    pairs: list[dict] = []
    for ids in by_tier.values():
        for x, y in zip(ids, ids[1:3]):
            a, b = _pair_key(x, y)
            pairs.append({"phone_a": a, "phone_b": b, "uniq_30d": None})
    return (await _attach(conn, pairs, source="same_tier"))[:limit]


async def top_sets(conn: asyncpg.Connection, limit: int, min_sessions: int) -> list[dict]:
    rows = rows_to_list(await conn.fetch(
        """SELECT set_key, set_size, phone_ids, uniq_30d, uniq_7d FROM compare_set_stats
WHERE uniq_30d >= $2 ORDER BY uniq_30d DESC, uniq_7d DESC, set_size DESC, set_key LIMIT $1""",
        limit * 3, min_sessions,
    ))
    if not rows:
        return []
    ids = sorted({i for r in rows for i in r["phone_ids"]})
    listable = {r["id"] for r in await conn.fetch(
        f"SELECT p.id FROM phone_catalog p WHERE p.id = ANY($1::int[]) AND {LISTABLE} AND {NOT_REGIONAL}", ids,
    )}
    minis = await phone_repo.minis_by_ids(conn, ids)
    out: list[dict] = []
    for r in rows:
        if not all(i in listable and i in minis for i in r["phone_ids"]):
            continue
        phones = sorted((minis[i] for i in r["phone_ids"]), key=lambda m: m["full_slug"])
        slugs = [m["full_slug"] for m in phones]
        out.append({
            "pair_slug": pair_slug(slugs), "path": compare_path(slugs), "phones": phones,
            "sessions_30d": r["uniq_30d"], "source": "popular",
        })
        if len(out) >= limit:
            break
    return out


async def most_compared(conn: asyncpg.Connection, limit: int, min_sessions: int) -> list[dict]:
    out = await top_sets(conn, limit, min_sessions)
    seen = {p["pair_slug"] for p in out}
    sources = (
        lambda: top_pairs(conn, limit * 2, min_sessions),
        lambda: curated_pairs(conn, limit * 2),
        lambda: same_tier_pairs(conn, limit * 2),
    )
    for make in sources:
        if len(out) >= limit:
            break
        for p in await make():
            if p["pair_slug"] not in seen:
                out.append(p)
                seen.add(p["pair_slug"])
            if len(out) >= limit:
                break
    return out[:limit]


async def pair_is_indexable(conn: asyncpg.Connection, a: int, b: int, min_sessions: int) -> bool:
    x, y = _pair_key(a, b)
    popular = await conn.fetchval(
        "SELECT EXISTS(SELECT 1 FROM compare_pair_stats WHERE phone_a = $1 AND phone_b = $2 AND uniq_30d >= $3)",
        x, y, min_sessions,
    )
    if popular:
        return True
    return bool(await conn.fetchval(
        """SELECT EXISTS(SELECT 1 FROM category_rankings p1 JOIN category_rankings p2
  ON p1.category_slug = p2.category_slug AND p1.rank <= 3 AND p2.rank <= 3
WHERE p1.phone_id = $1 AND p2.phone_id = $2)""", x, y,
    ))
