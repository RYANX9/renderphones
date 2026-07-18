from __future__ import annotations

from typing import Any, Optional

import asyncpg

from ..core.database import rows_to_list, row_to_dict
from ..core.query import FilterParams, build_filter_where, resolve_sort
from ..core.scoring import similarity_score
from ..core.shaping import attach_computed_fields, pop_smart_score
from ..core.sql_fragments import PHONE_JOIN, PHONE_LIST_SELECT, PHONE_DETAIL_SELECT

# Peer-group size used to normalise value_score for a single-phone lookup.
VALUE_PEER_LIMIT = 40
# Candidate pool pulled before similarity re-ranking in /similar.
SIMILAR_CANDIDATE_POOL = 60


async def search(
    conn: asyncpg.Connection,
    *,
    filters: FilterParams,
    sort_by: str,
    sort_order: str,
    page: int,
    page_size: int,
) -> tuple[int, list[dict]]:
    where, params = build_filter_where(filters)
    sort_expr, order = resolve_sort(sort_by, sort_order, has_query=bool(filters.q))
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
    """A price_points row with a NULL price_usd (untracked/out-of-stock
    snapshot) must not clobber the phones.price_usd fallback."""
    if price is None or price.get("price_usd") is None:
        return
    target["price_usd"] = price["price_usd"]
    target["price_original"] = price.get("price_original")
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


async def similar_phones(conn: asyncpg.Connection, phone_id: int, limit: int) -> list[dict]:
    """Multi-factor similarity: price band + brand + chipset tier + camera/
    battery/screen closeness, instead of a plain price-band-plus-brand SQL
    ORDER BY. Pulls a wider SQL candidate pool, then re-ranks in Python
    against the full similarity_score() weighting."""
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

    scored = [(similarity_score(anchor, c), c) for c in candidates]
    scored.sort(key=lambda t: t[0], reverse=True)
    top = [c for _, c in scored[:limit]]

    attach_computed_fields(top, peers=candidates)
    for p in top:
        p["smart_score"] = pop_smart_score(p)
    return top


async def compare_by_ids(conn: asyncpg.Connection, ids: list[int]) -> list[dict]:
    rows = await conn.fetch(
        f"SELECT {PHONE_LIST_SELECT} {PHONE_JOIN} WHERE p.id = ANY($1::int[])",
        ids,
    )
    phones = rows_to_list(rows)
    for p in phones:
        price = await latest_price_point(conn, p["id"])
        apply_latest_price(p, price)

    attach_computed_fields(phones)
    for p in phones:
        p["smart_score"] = pop_smart_score(p)
    return phones


async def compare_by_slugs(conn: asyncpg.Connection, slugs: list[str]) -> list[dict]:
    rows = await conn.fetch(
        f"SELECT {PHONE_LIST_SELECT} {PHONE_JOIN} WHERE p.slug = ANY($1::text[])",
        slugs,
    )
    phones = rows_to_list(rows)
    for p in phones:
        price = await latest_price_point(conn, p["id"])
        apply_latest_price(p, price)

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
    rows = await conn.fetch(
        f"""
        SELECT {PHONE_LIST_SELECT}
        {PHONE_JOIN}
        ORDER BY p.popularity DESC NULLS LAST,
                 p.fans DESC NULLS LAST,
                 COALESCE(sc.overall_score, 0) DESC,
                 p.release_year DESC NULLS LAST,
                 p.id DESC
        LIMIT {int(limit)}
        """
    )
    phones = rows_to_list(rows)
    attach_computed_fields(phones)
    for p in phones:
        p["smart_score"] = pop_smart_score(p)
    return phones
