from __future__ import annotations

from typing import Any, Optional

import asyncpg
import time

from ..core.sql_fragments import PHONE_JOIN, PHONE_LIST_SELECT, PHONE_DETAIL_SELECT, RELEASE_TS_EXPR

from ..core.database import rows_to_list, row_to_dict
from ..core.query import FilterParams, build_filter_where, resolve_sort
from ..core.scoring import similarity_score
from ..core.shaping import attach_computed_fields, pop_smart_score
from ..core.market import phone_allowed_for_region, cap_per_brand

SIMILAR_BRAND_CAP = 2
# Peer-group size used to normalise value_score for a single-phone lookup.
VALUE_PEER_LIMIT = 40
# Candidate pool pulled before similarity re-ranking in /similar.
SIMILAR_CANDIDATE_POOL = 60
# Trending candidate pool pulled before quality+recency re-ranking, same
# pattern as SIMILAR_CANDIDATE_POOL — cheap SQL ordering gets a wide
# enough pool, the real ranking happens in Python once value_score is
# resolved for phones with no AI scoring at all.
TRENDING_CANDIDATE_MULTIPLIER = 6
TRENDING_MIN_POOL = 60

# Weight given to "well-scored" vs "recently released" in the trending
# blend. Tune here, not scattered through the sort key.
TRENDING_QUALITY_WEIGHT = 0.6
TRENDING_RECENCY_WEIGHT = 0.4
TRENDING_RECENCY_DECAY_DAYS = 365.0  # linear decay to 0 over one year

async def search(
    conn: asyncpg.Connection,
    *,
    filters: FilterParams,
    sort_by: str,
    sort_order: str,
    page: int,
    page_size: int,
) -> tuple[int, list[dict]]:
    where, params, relevance_expr = build_filter_where(filters)
    sort_expr, order = resolve_sort(sort_by, sort_order, has_query=bool(filters.q), relevance_expr=relevance_expr)
    offset = (page - 1) * page_size

    total = await conn.fetchval(f"SELECT COUNT(*) {PHONE_JOIN} WHERE {where}", *params)

    i = len(params) + 1
    rows = await conn.fetch(
        f"""
        SELECT {PHONE_LIST_SELECT}
        {PHONE_JOIN}
        WHERE {where}
        ORDER BY {sort_expr} {order} NULLS LAST, p.id DESC
        LIMIT ${i} OFFSET ${i + 1}
        """,
        *params, page_size, offset,
    )

    phones = rows_to_list(rows)
    attach_computed_fields(phones)
    for p in phones:
        p["smart_score"] = pop_smart_score(p)

    return total, phones
    


async def get_by_id_or_slug(conn: asyncpg.Connection, phone_id_or_slug: str) -> dict | None:
    if phone_id_or_slug.isdigit():
        where_clause, param = "p.id = $1", int(phone_id_or_slug)
    else:
        where_clause, param = "p.slug = $1", phone_id_or_slug

    row = await conn.fetchrow(
        f"SELECT {PHONE_DETAIL_SELECT} {PHONE_JOIN} WHERE {where_clause}",
        param,
    )
    return row_to_dict(row)


async def fetch_variants(conn: asyncpg.Connection, phone_id: int) -> list[dict]:
    rows = await conn.fetch(
        """
        SELECT id, ram_gb, storage_gb, price, url
        FROM phone_variants
        WHERE phone_id = $1
        ORDER BY storage_gb ASC NULLS LAST, ram_gb ASC NULLS LAST
        """,
        phone_id,
    )
    return rows_to_list(rows)

async def fetch_full_specifications(conn: asyncpg.Connection, phone_id: int) -> dict | None:
    row = await conn.fetchrow(
        "SELECT full_specifications FROM phone_full_specifications WHERE phone_id = $1",
        phone_id,
    )
    return row_to_dict(row)
    
async def fetch_images(conn: asyncpg.Connection, phone_id: int) -> list[dict]:
    rows = await conn.fetch(
        """
        SELECT id, image_url, sort_order
        FROM phone_images
        WHERE phone_id = $1
        ORDER BY sort_order ASC NULLS LAST, id ASC
        """,
        phone_id,
    )
    return rows_to_list(rows)


async def fetch_features(conn: asyncpg.Connection, phone_id: int) -> list[str]:
    rows = await conn.fetch(
        "SELECT feature_name FROM phone_features WHERE phone_id = $1 ORDER BY feature_name ASC",
        phone_id,
    )
    return [r["feature_name"] for r in rows]

async def fetch_retailer_links(conn: asyncpg.Connection, phone_id: int) -> list[dict]:
    rows = await conn.fetch(
        """
        SELECT id, phone_id, variant_id, retailer, region, url, price,
               currency, is_available, status, checked_at
        FROM retailer_links
        WHERE phone_id = $1
        ORDER BY region ASC, retailer ASC
        """,
        phone_id,
    )
    return rows_to_list(rows)

async def latest_price_point(conn: asyncpg.Connection, phone_id: int) -> dict | None:
    row = await conn.fetchrow(
        """
        SELECT price_usd, price_original, scope, snapshot_date
        FROM price_points
        WHERE phone_id = $1
        ORDER BY (scope = 'global') DESC, snapshot_date DESC
        LIMIT 1
        """,
        phone_id,
    )
    return row_to_dict(row)


def apply_latest_price(target: dict, price: dict | None) -> None:
    """price_usd is already resolved (variant floor / latest global
    price_point / phones.price_usd, whichever is lowest) by
    PRICE_RESOLVED_EXPR in the SELECT itself — this must NOT touch
    price_usd or price_original again, or it silently undoes that
    resolution for compare_by_ids/compare_by_slugs. All this does now is
    attach display metadata about the freshest tracked price_points
    snapshot, when one exists — a null price_usd on that snapshot (an
    untracked/out-of-stock row) just means no metadata to attach."""
    if price is None or price.get("price_usd") is None:
        return
    target["price_updated_at"] = str(price["snapshot_date"])
    target["price_scope"] = price["scope"]


async def fetch_value_peers(conn: asyncpg.Connection, phone: dict) -> list[dict]:
    """Real comparison set for a single-phone value_score fallback: same
    price band, brand-affinity ordering, so the number matches what the
    phone would show on a list page instead of degenerating to a
    peer group of one."""
    price = phone.get("price_usd")
    lo = price * 0.65 if price else None
    hi = price * 1.45 if price else None

    rows = await conn.fetch(
        f"""
        SELECT {PHONE_LIST_SELECT}
        {PHONE_JOIN}
        WHERE p.id != $1
          AND ($2::numeric IS NULL OR p.price_usd BETWEEN $2 AND $3)
        ORDER BY
            CASE WHEN p.brand = $4 THEN 0 ELSE 1 END,
            ABS(COALESCE(p.price_usd, 0) - COALESCE($5, 0)),
            p.popularity DESC NULLS LAST
        LIMIT $6
        """,
        phone["id"], lo, hi, phone.get("brand"), price, VALUE_PEER_LIMIT,
    )
    return rows_to_list(rows)



async def _fetch_retailer_links_bulk(conn: asyncpg.Connection, phone_ids: list[int]) -> dict[int, list[dict]]:
    if not phone_ids:
        return {}
    rows = await conn.fetch(
        "SELECT phone_id, retailer, region, is_available FROM retailer_links WHERE phone_id = ANY($1::int[])",
        phone_ids,
    )
    grouped: dict[int, list[dict]] = {}
    for r in rows:
        grouped.setdefault(r["phone_id"], []).append(dict(r))
    return grouped


async def similar_phones(
    conn: asyncpg.Connection, phone_id: int, limit: int, region: str | None = None,
) -> list[dict]:
    anchor_row = await conn.fetchrow(
        f"SELECT {PHONE_LIST_SELECT} {PHONE_JOIN} WHERE p.id = $1",
        phone_id,
    )
    if anchor_row is None:
        return []
    anchor = row_to_dict(anchor_row)

    price = anchor.get("price_usd")
    lo = price * 0.5 if price else None
    hi = price * 1.8 if price else None

    rows = await conn.fetch(
        f"""
        SELECT {PHONE_LIST_SELECT}
        {PHONE_JOIN}
        WHERE p.id != $1
          AND ($2::numeric IS NULL OR p.price_usd BETWEEN $2 AND $3)
        ORDER BY p.popularity DESC NULLS LAST
        LIMIT $4
        """,
        phone_id, lo, hi, SIMILAR_CANDIDATE_POOL,
    )
    candidates = rows_to_list(rows)

    if region:
        links_by_phone = await _fetch_retailer_links_bulk(conn, [c["id"] for c in candidates])
        candidates = [
            c for c in candidates
            if phone_allowed_for_region(links_by_phone.get(c["id"], []), region)
        ]

    scored = [(similarity_score(anchor, c), c) for c in candidates]
    scored.sort(key=lambda t: t[0], reverse=True)
    ranked = [c for _, c in scored]

    ranked = cap_per_brand(ranked, SIMILAR_BRAND_CAP)
    top = ranked[:limit]

    attach_computed_fields(top, peers=candidates)
    for p in top:
        p["smart_score"] = pop_smart_score(p)
    return top


async def compare_by_ids(conn: asyncpg.Connection, ids: list[int]) -> list[dict]:
    rows = await conn.fetch(
        f"SELECT {PHONE_DETAIL_SELECT} {PHONE_JOIN} WHERE p.id = ANY($1::int[])",
        ids,
    )
    phones = rows_to_list(rows)
    for p in phones:
        price = await latest_price_point(conn, p["id"])
        apply_latest_price(p, price)
        p["features"] = await fetch_features(conn, p["id"])

    attach_computed_fields(phones)
    for p in phones:
        p["smart_score"] = pop_smart_score(p)
    return phones


async def compare_by_slugs(conn: asyncpg.Connection, slugs: list[str]) -> list[dict]:
    rows = await conn.fetch(
        f"SELECT {PHONE_DETAIL_SELECT} {PHONE_JOIN} WHERE p.slug = ANY($1::text[])",
        slugs,
    )
    phones = rows_to_list(rows)
    for p in phones:
        price = await latest_price_point(conn, p["id"])
        apply_latest_price(p, price)
        p["features"] = await fetch_features(conn, p["id"])

    attach_computed_fields(phones)
    for p in phones:
        p["smart_score"] = pop_smart_score(p)
    return phones


async def latest(conn: asyncpg.Connection, limit: int) -> list[dict]:
    rows = await conn.fetch(
        f"""
        SELECT {PHONE_LIST_SELECT}
        {PHONE_JOIN}
        ORDER BY p.release_year DESC NULLS LAST,
                 p.release_month DESC NULLS LAST,
                 p.release_day DESC NULLS LAST,
                 p.id DESC
        LIMIT {int(limit)}
        """
    )
    phones = rows_to_list(rows)
    attach_computed_fields(phones)
    for p in phones:
        p["smart_score"] = pop_smart_score(p)
    return phones


async def trending(conn: asyncpg.Connection, limit: int) -> list[dict]:
    """Was ordered by p.popularity/p.fans — placeholder columns with no
    real signal behind them until view/account tracking exists. Until
    then, "trending" means what it should honestly mean: well-scored and
    recently released, not a fabricated popularity count."""
    pool_size = max(limit * TRENDING_CANDIDATE_MULTIPLIER, TRENDING_MIN_POOL)

    rows = await conn.fetch(
        f"""
        SELECT {PHONE_LIST_SELECT}
        {PHONE_JOIN}
        ORDER BY
            {RELEASE_TS_EXPR} DESC NULLS LAST,
            COALESCE(sc.overall_score, s.antutu_score / 100000.0, 0) DESC,
            p.id DESC
        LIMIT {int(pool_size)}
        """
    )

    phones = rows_to_list(rows)
    attach_computed_fields(phones)
    for p in phones:
        p["smart_score"] = pop_smart_score(p)

    now_ts = time.time()

    def trending_rank(p: dict) -> float:
        quality = p.get("value_score") or 0.0
        release_ts = p.get("release_ts")
        age_days = max((now_ts - release_ts) / 86_400, 0.0) if release_ts else TRENDING_RECENCY_DECAY_DAYS
        recency = max(0.0, 10.0 - (age_days / TRENDING_RECENCY_DECAY_DAYS) * 10.0)
        return quality * TRENDING_QUALITY_WEIGHT + recency * TRENDING_RECENCY_WEIGHT

    phones.sort(key=trending_rank, reverse=True)
    return phones[:limit]
