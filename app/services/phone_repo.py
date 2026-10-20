from __future__ import annotations

from collections import defaultdict
from typing import Any

import asyncpg

from ..core.category_config import CATEGORY_CONFIG
from ..core.currency import to_local
from ..core.database import row_to_dict, rows_to_list
from ..core.market import allowed_scopes, cap_per_brand, retailer_info
from ..core.query import FilterParams, build_filter_where, resolve_sort
from ..core.scoring import similarity_score
from ..core.shaping import build_analysis, shape_card, shape_cards, shape_detail, shape_mini
from ..core.slugs import brand_path, category_path, phone_path
from ..core.sql_fragments import (
    CARD_SELECT, DETAIL_SELECT, LISTABLE, MINI_SELECT, NOT_REGIONAL, UPCOMING, UPCOMING_STAGE, UPCOMING_STAGE_RANK
)

SIMILAR_BRAND_CAP = 2
SIMILAR_CANDIDATE_POOL = 60


# ── lists ────────────────────────────────────────────────────────────────────

async def search(conn: asyncpg.Connection, *, filters: FilterParams, sort_by: str, sort_order: str,
                 page: int, page_size: int) -> tuple[int, list[dict]]:
    where, params, relevance_expr = build_filter_where(filters)
    sort_expr, order = resolve_sort(sort_by, sort_order, has_query=bool(filters.q), relevance_expr=relevance_expr)
    i = len(params) + 1
    rows = await conn.fetch(
        f"SELECT {CARD_SELECT}, COUNT(*) OVER() AS _total FROM phone_catalog p WHERE {where} "
        f"ORDER BY {sort_expr} {order} NULLS LAST, p.id DESC LIMIT ${i} OFFSET ${i + 1}",
        *params, page_size, (page - 1) * page_size,
    )
    results = rows_to_list(rows)
    if results:
        total = results[0]["_total"]
    elif page == 1:
        total = 0
    else:
        total = await conn.fetchval(f"SELECT COUNT(*) FROM phone_catalog p WHERE {where}", *params)
    return total, shape_cards(results)


async def _list(conn, where: str, order: str, limit: int, *params: Any) -> list[dict]:
    rows = await conn.fetch(
        f"SELECT {CARD_SELECT} FROM phone_catalog p WHERE {where} ORDER BY {order} LIMIT ${len(params) + 1}",
        *params, limit,
    )
    return shape_cards(rows_to_list(rows))


async def latest(conn: asyncpg.Connection, limit: int) -> list[dict]:
    return await _list(
        conn, f"{LISTABLE} AND {NOT_REGIONAL} AND p.release_ts <= EXTRACT(EPOCH FROM now())",
        "p.release_ts DESC NULLS LAST, p.id DESC", limit,
    )


async def upcoming(conn: asyncpg.Connection, limit: int) -> list[dict]:
    return await _list(conn, f"{UPCOMING} AND {NOT_REGIONAL}", "p.release_ts ASC NULLS LAST, p.id", limit)


async def _brand_capped(conn, where: str, order: str, limit: int, cap: int, region: str | None) -> list[dict]:
    params: list[Any] = []
    scopes = allowed_scopes(region)
    if scopes is not None:
        where += " AND (cardinality(p.offer_scopes) = 0 OR p.offer_scopes && $1::text[])"
        params.append(scopes)
    pool = await _list(conn, where, order, max(limit * 4, 30), *params)
    return cap_per_brand(pool, cap)[:limit]


async def top_rated(conn: asyncpg.Connection, limit: int, region: str | None = None) -> list[dict]:
    return await _brand_capped(
        conn, f"{LISTABLE} AND {NOT_REGIONAL} AND p.overall_score IS NOT NULL",
        "p.overall_score DESC, p.price_usd ASC NULLS LAST, p.id", limit, 2, region,
    )


TIER_ORDER = ("ultra_flagship", "flagship", "upper_mid_range", "mid_range", "budget")


async def top_rated_by_tier(conn: asyncpg.Connection, per_tier: int, region: str | None = None) -> list[dict]:
    out: list[dict] = []
    for tier in TIER_ORDER:
        out += await _brand_capped(
            conn, f"{LISTABLE} AND {NOT_REGIONAL} AND p.overall_score IS NOT NULL AND p.tier = '{tier}'",
            "p.overall_score DESC, p.price_usd ASC NULLS LAST, p.id", per_tier, 2, region,
        )
    return out


async def trending(conn: asyncpg.Connection, limit: int, region: str | None = None) -> list[dict]:
    return await _brand_capped(
        conn, f"{LISTABLE} AND {NOT_REGIONAL}", "p.trend_score DESC, p.overall_score DESC NULLS LAST, p.id", limit, 2, region,
    )

async def most_viewed(conn: asyncpg.Connection, limit: int, region: str | None = None) -> list[dict]:
    return await _brand_capped(
        conn, f"{LISTABLE} AND {NOT_REGIONAL}",
        "p.views_7d DESC NULLS LAST, p.trend_score DESC NULLS LAST, p.id", limit, 2, region,
    )


_UPCOMING_SELECT = f"{CARD_SELECT}, p.release_date_full, {UPCOMING_STAGE} AS stage"


async def upcoming_by_stage(conn: asyncpg.Connection, per_stage: int) -> list[dict]:
    rows = await conn.fetch(
        f"""SELECT * FROM (
SELECT {_UPCOMING_SELECT},
       row_number() OVER (
         PARTITION BY {UPCOMING_STAGE}
         ORDER BY p.release_ts ASC NULLS LAST, p.overall_score DESC NULLS LAST, p.id
       ) AS rn
FROM phone_catalog p
WHERE {UPCOMING} AND {NOT_REGIONAL}
) t
WHERE rn <= $1
ORDER BY CASE stage WHEN 'launching_soon' THEN 0 WHEN 'announced' THEN 1 ELSE 2 END, rn""",
        per_stage,
    )
    out = shape_cards(rows_to_list(rows))
    for row in out:
        row.pop("rn", None)
    return out


async def brand_upcoming(conn: asyncpg.Connection, brand_slug: str, limit: int = 12) -> list[dict]:
    rows = await conn.fetch(
        f"""SELECT {_UPCOMING_SELECT} FROM phone_catalog p
WHERE p.brand_slug = $1 AND {UPCOMING} AND {NOT_REGIONAL}
ORDER BY {UPCOMING_STAGE_RANK}, p.release_ts ASC NULLS LAST, p.overall_score DESC NULLS LAST, p.id
LIMIT $2""",
        brand_slug, limit,
    )
    return shape_cards(rows_to_list(rows))


async def minis_by_ids(conn: asyncpg.Connection, ids: list[int]) -> dict[int, dict]:
    if not ids:
        return {}
    rows = await conn.fetch(f"SELECT {MINI_SELECT} FROM phone_catalog p WHERE p.id = ANY($1::int[])", list(set(ids)))
    return {r["id"]: shape_mini(row_to_dict(r)) for r in rows}


async def slug_index(conn: asyncpg.Connection) -> list[dict]:
    rows = await conn.fetch(
        "SELECT id, brand_slug, slug, full_slug, display_name, status, is_regional_edition, updated_at "
        "FROM phones ORDER BY id"
    )
    return rows_to_list(rows)


async def resolve_full_slug(conn: asyncpg.Connection, full_slug: str) -> dict | None:
    row = await conn.fetchrow("SELECT id, brand_slug, slug FROM phones WHERE full_slug = $1", full_slug.lower())
    if row is None:
        return None
    d = row_to_dict(row)
    d["path"] = phone_path(d["brand_slug"], d["slug"])
    return d


async def phone_exists(conn: asyncpg.Connection, phone_id: int) -> bool:
    return await conn.fetchval("SELECT EXISTS(SELECT 1 FROM phones WHERE id = $1)", phone_id)


# ── detail ───────────────────────────────────────────────────────────────────

_VARIANTS_SQL = """
SELECT v.id, v.ram_gb, v.storage_gb, v.is_default,
       o.usd AS price_usd, o.url, o.retailer_key, o.region, o.is_placeholder,
       (SELECT count(*)::int FROM offers x WHERE x.variant_id = v.id AND x.is_available) AS offer_count
FROM variants v
LEFT JOIN LATERAL (
  SELECT f.usd, f.url, f.retailer_key, f.region, f.is_placeholder
  FROM offers f
  WHERE f.variant_id = v.id AND f.is_available AND f.usd IS NOT NULL AND f.usd > 0
  ORDER BY f.usd ASC, f.is_placeholder ASC, f.id
  LIMIT 1
) o ON true
WHERE v.phone_id = $1
ORDER BY v.ram_gb NULLS LAST, v.storage_gb NULLS LAST, v.id
"""

_OFFERS_SQL = """
SELECT o.id, o.variant_id, v.ram_gb, v.storage_gb, o.retailer_key,
       r.name AS retailer_name, r.domain, r.scope,
       COALESCE(o.region, r.region) AS region,
       o.usd, o.url, o.is_available, o.status, o.is_placeholder, o.price_source,
       o.updated_at, o.checked_at
FROM offers o
JOIN variants v ON v.id = o.variant_id
LEFT JOIN retailers r ON r.key = o.retailer_key
WHERE v.phone_id = $1
ORDER BY o.usd ASC NULLS LAST, o.id
"""


def _with_local(row: dict, usd_key: str, region_key: str = "region") -> dict:
    row["local"] = to_local(row.get(usd_key), row.get(region_key))
    return row


async def fetch_variants(conn: asyncpg.Connection, phone_id: int) -> list[dict]:
    return [_with_local(r, "price_usd") for r in rows_to_list(await conn.fetch(_VARIANTS_SQL, phone_id))]


async def fetch_offers(conn: asyncpg.Connection, phone_id: int) -> list[dict]:
    out = []
    for r in rows_to_list(await conn.fetch(_OFFERS_SQL, phone_id)):
        info = retailer_info(r["retailer_key"])
        r["retailer_name"] = r.get("retailer_name") or info.get("name") or r["retailer_key"]
        r["scope"] = r.get("scope") or info.get("scope") or "global"
        out.append(_with_local(r, "usd"))
    return out


async def fetch_analysis(conn: asyncpg.Connection, phone_id: int) -> dict | None:
    row = await conn.fetchrow(
        "SELECT reasoning, strengths, weaknesses, model_version, scored_at FROM phone_analysis WHERE phone_id = $1",
        phone_id,
    )
    return build_analysis(row_to_dict(row))


async def phone_categories(conn: asyncpg.Connection, phone_id: int) -> list[dict]:
    rows = await conn.fetch(
        "SELECT category_slug AS slug, rank, score FROM category_rankings WHERE phone_id = $1 "
        "ORDER BY rank, category_slug", phone_id,
    )
    out = []
    for r in rows_to_list(rows):
        cfg = CATEGORY_CONFIG.get(r["slug"])
        if cfg:
            out.append({**r, "title": cfg["title"], "path": category_path(r["slug"])})
    return out


async def get_detail(conn: asyncpg.Connection, *, phone_id: int | None = None,
                     brand_slug: str | None = None, slug: str | None = None) -> dict | None:
    if phone_id is not None:
        row = await conn.fetchrow(f"SELECT {DETAIL_SELECT} FROM phone_catalog p WHERE p.id = $1", phone_id)
    else:
        row = await conn.fetchrow(
            f"SELECT {DETAIL_SELECT} FROM phone_catalog p WHERE p.brand_slug = $1 AND p.slug = $2",
            (brand_slug or "").lower(), (slug or "").lower(),
        )
    if row is None:
        return None
    phone = shape_detail(row_to_dict(row))
    pid = phone["id"]

    phone["variants"] = await fetch_variants(conn, pid)
    offers = await fetch_offers(conn, pid)
    phone["_offer_rows"] = offers  # region-resolved by the route, never cached resolved
    phone["images"] = rows_to_list(await conn.fetch(
        "SELECT id, url, kind, alt, width, height, sort_order FROM phone_images WHERE phone_id = $1 "
        "ORDER BY (kind = 'main') DESC, sort_order, id", pid,
    ))
    phone["features"] = [r["feature_name"] for r in await conn.fetch(
        "SELECT feature_name FROM phone_features WHERE phone_id = $1 ORDER BY feature_name", pid,
    )]
    phone["price_history_points"] = await conn.fetchval(
        "SELECT count(*)::int FROM price_history ph JOIN variants v ON v.id = ph.variant_id WHERE v.phone_id = $1", pid,
    )
    phone["analysis"] = await fetch_analysis(conn, pid)
    phone["categories"] = await phone_categories(conn, pid)
    phone["brand_path"] = brand_path(phone["brand_slug"])
    return phone


async def fetch_full_specifications(conn: asyncpg.Connection, phone_id: int) -> dict | None:
    row = await conn.fetchrow(
        "SELECT full_specifications FROM phone_full_specifications WHERE phone_id = $1 ORDER BY id DESC LIMIT 1",
        phone_id,
    )
    return row_to_dict(row)


# ── similar / rivals ─────────────────────────────────────────────────────────

async def similar_phones(conn: asyncpg.Connection, phone_id: int, limit: int, region: str | None = None,
                         band: tuple[float, float] = (0.5, 1.8)) -> list[dict]:
    anchor_row = await conn.fetchrow(f"SELECT {CARD_SELECT} FROM phone_catalog p WHERE p.id = $1", phone_id)
    if anchor_row is None:
        return []
    anchor = row_to_dict(anchor_row)
    price = anchor.get("price_usd")
    lo = price * band[0] if price else None
    hi = price * band[1] if price else None

    params: list[Any] = [phone_id, lo, hi, SIMILAR_CANDIDATE_POOL]
    region_clause = ""
    scopes = allowed_scopes(region)
    if scopes is not None:
        region_clause = "AND (cardinality(p.offer_scopes) = 0 OR p.offer_scopes && $5::text[])"
        params.append(scopes)
    rows = await conn.fetch(
        f"""SELECT {CARD_SELECT} FROM phone_catalog p
WHERE p.id <> $1 AND {LISTABLE} AND {NOT_REGIONAL}
  AND ($2::numeric IS NULL OR p.price_usd BETWEEN $2 AND $3)
  {region_clause}
ORDER BY p.trend_score DESC NULLS LAST
LIMIT $4""",
        *params,
    )
    candidates = rows_to_list(rows)
    scored = sorted(((similarity_score(anchor, c), c) for c in candidates), key=lambda x: x[0], reverse=True)
    ranked = cap_per_brand([c for _, c in scored], SIMILAR_BRAND_CAP)
    return shape_cards(ranked[:limit])


async def rivals(conn: asyncpg.Connection, phone_id: int, limit: int = 6, region: str | None = None) -> list[dict]:
    """Phones priced within about 15% of this one. Widens once to 30% if the
    tight band is too thin to be useful."""
    found = await similar_phones(conn, phone_id, limit, region, band=(0.85, 1.15))
    if len(found) >= min(3, limit):
        return found
    return await similar_phones(conn, phone_id, limit, region, band=(0.70, 1.30))


# ── compare ──────────────────────────────────────────────────────────────────

async def compare(conn: asyncpg.Connection, *, ids: list[int] | None = None,
                  full_slugs: list[str] | None = None) -> list[dict]:
    if ids is not None:
        rows = await conn.fetch(
            f"SELECT {DETAIL_SELECT} FROM phone_catalog p WHERE p.id = ANY($1::int[]) "
            "ORDER BY array_position($1::int[], p.id)", ids,
        )
    else:
        rows = await conn.fetch(
            f"SELECT {DETAIL_SELECT} FROM phone_catalog p WHERE p.full_slug = ANY($1::text[]) "
            "ORDER BY array_position($1::text[], p.full_slug)", full_slugs,
        )
    phones = [shape_detail(r) for r in rows_to_list(rows)]
    if not phones:
        return phones
    pids = [p["id"] for p in phones]
    feats: dict[int, list[str]] = defaultdict(list)
    for r in await conn.fetch(
        "SELECT phone_id, feature_name FROM phone_features WHERE phone_id = ANY($1::int[]) ORDER BY feature_name", pids,
    ):
        feats[r["phone_id"]].append(r["feature_name"])
    analyses = {
        r["phone_id"]: build_analysis(row_to_dict(r))
        for r in await conn.fetch(
            "SELECT phone_id, reasoning, strengths, weaknesses, model_version, scored_at "
            "FROM phone_analysis WHERE phone_id = ANY($1::int[])", pids,
        )
    }
    for p in phones:
        p["features"] = feats.get(p["id"], [])
        p["analysis"] = analyses.get(p["id"])
    return phones


# ── price history (chart only) ───────────────────────────────────────────────

async def price_history(conn: asyncpg.Connection, phone_id: int) -> dict:
    """Used ONLY to draw the chart. Nothing else in the API reads price_history."""
    rows = await conn.fetch(
        """SELECT ph.variant_id, v.ram_gb, v.storage_gb, ph.on_date, ph.usd
FROM price_history ph JOIN variants v ON v.id = ph.variant_id
WHERE v.phone_id = $1 ORDER BY ph.on_date, ph.variant_id""", phone_id,
    )
    series: dict[int, dict] = {}
    lowest: dict[str, float] = {}
    for r in rows:
        s = series.setdefault(r["variant_id"], {
            "variant_id": r["variant_id"], "ram_gb": r["ram_gb"], "storage_gb": r["storage_gb"], "points": [],
        })
        d, usd = str(r["on_date"]), float(r["usd"])
        s["points"].append({"date": d, "usd": usd})
        lowest[d] = min(usd, lowest.get(d, usd))
    return {
        "phone_id": phone_id,
        "series": list(series.values()),
        "lowest": [{"date": d, "usd": v} for d, v in sorted(lowest.items())],
    }


# ── categories ───────────────────────────────────────────────────────────────

async def category_rankings(conn: asyncpg.Connection, slug: str, pool: int, region: str | None,
                            statuses: list[str] | None = None) -> list[dict]:
    params: list[Any] = [slug, pool, statuses]
    region_clause = ""
    scopes = allowed_scopes(region)
    if scopes is not None:
        region_clause = "AND (cardinality(p.offer_scopes) = 0 OR p.offer_scopes && $4::text[])"
        params.append(scopes)
    rows = await conn.fetch(
        f"""SELECT {CARD_SELECT}, r.score AS category_score, r.rank AS category_rank, r.breakdown AS breakdown
FROM category_rankings r JOIN phone_catalog p ON p.id = r.phone_id
WHERE r.category_slug = $1 AND ($3::text[] IS NULL OR p.status = ANY($3)) {region_clause}
ORDER BY r.rank LIMIT $2""",
        *params,
    )
    out = []
    for row in rows_to_list(rows):
        row["category_score"] = round(float(row["category_score"]), 2)
        out.append(shape_card(row))
    return out


async def category_overlap(conn: asyncpg.Connection, slug: str, top_n: int = 10) -> list[dict]:
    """Other categories ordered by how many of this category's top phones they share."""
    rows = await conn.fetch(
        """SELECT b.category_slug AS slug, count(*)::int AS shared
FROM category_rankings a JOIN category_rankings b
  ON b.phone_id = a.phone_id AND b.category_slug <> a.category_slug AND b.rank <= $2
WHERE a.category_slug = $1 AND a.rank <= $2
GROUP BY b.category_slug ORDER BY shared DESC, b.category_slug""",
        slug, top_n,
    )
    return rows_to_list(rows)


async def category_counts(conn: asyncpg.Connection) -> dict[str, int]:
    rows = await conn.fetch("SELECT category_slug, count(*)::int AS n FROM category_rankings GROUP BY 1")
    return {r["category_slug"]: r["n"] for r in rows}


# ── brands ───────────────────────────────────────────────────────────────────

_BRAND_STATS_SQL = f"""
SELECT p.brand, p.brand_slug, count(*)::int AS phone_count,
       min(p.price_usd) AS min_price, max(p.price_usd) AS max_price,
       round(avg(p.price_usd)) AS avg_price, round(avg(p.battery_capacity)) AS avg_battery,
       max(p.release_year) AS latest_year, max(p.updated_at) AS updated_at
FROM phone_catalog p
WHERE {LISTABLE} AND {NOT_REGIONAL}
GROUP BY p.brand, p.brand_slug
"""


async def brand_list(conn: asyncpg.Connection) -> list[dict]:
    rows = rows_to_list(await conn.fetch(_BRAND_STATS_SQL + " ORDER BY phone_count DESC, brand"))
    tops = await conn.fetch(
        f"""SELECT DISTINCT ON (p.brand_slug) {MINI_SELECT}
FROM phone_catalog p WHERE {LISTABLE} AND {NOT_REGIONAL} AND p.overall_score IS NOT NULL
ORDER BY p.brand_slug, p.overall_score DESC, p.price_usd ASC NULLS LAST""",
    )
    top_by_brand = {r["brand_slug"]: shape_mini(row_to_dict(r)) for r in tops}
    for r in rows:
        r["path"] = brand_path(r["brand_slug"])
        r["top_phone"] = top_by_brand.get(r["brand_slug"])
    return rows


async def brand_detail(conn: asyncpg.Connection, brand_slug: str) -> dict | None:
    row = await conn.fetchrow(_BRAND_STATS_SQL.replace("GROUP BY", "AND p.brand_slug = $1 GROUP BY"), brand_slug)
    if row is None:
        return None
    stats = row_to_dict(row)
    latest_row = await conn.fetchrow(
        f"SELECT {CARD_SELECT} FROM phone_catalog p WHERE p.brand_slug = $1 AND {LISTABLE} AND {NOT_REGIONAL} "
        "AND p.release_ts <= EXTRACT(EPOCH FROM now()) ORDER BY p.release_ts DESC NULLS LAST LIMIT 1", brand_slug,
    )
    top_rows = await conn.fetch(
        f"SELECT {CARD_SELECT} FROM phone_catalog p WHERE p.brand_slug = $1 AND {LISTABLE} AND {NOT_REGIONAL} "
        "AND p.overall_score IS NOT NULL ORDER BY p.overall_score DESC, p.price_usd ASC NULLS LAST LIMIT 3", brand_slug,
    )
    cat_rows = await conn.fetch(
        """SELECT r.category_slug AS slug, count(*)::int AS ranked, min(r.rank)::int AS best_rank
FROM category_rankings r JOIN phones p ON p.id = r.phone_id
WHERE p.brand_slug = $1 GROUP BY r.category_slug ORDER BY best_rank, ranked DESC""", brand_slug,
    )
    categories = [
        {**r, "title": CATEGORY_CONFIG[r["slug"]]["title"], "path": category_path(r["slug"])}
        for r in rows_to_list(cat_rows) if r["slug"] in CATEGORY_CONFIG
    ]
    return {
        "brand": stats["brand"],
        "brand_slug": stats["brand_slug"],
        "path": brand_path(stats["brand_slug"]),
        "total_phones": stats["phone_count"],
        "price_range": {"min": stats["min_price"], "max": stats["max_price"], "avg": stats["avg_price"]},
        "avg_battery": stats["avg_battery"],
        "latest_year": stats["latest_year"],
        "updated_at": stats["updated_at"],
        "latest_phone": shape_card(row_to_dict(latest_row)) if latest_row else None,
        "top_phones": shape_cards(rows_to_list(top_rows)),
        "categories": categories,
        "upcoming": await brand_upcoming(conn, brand_slug),
    }


async def catalog_stats(conn: asyncpg.Connection) -> dict:
    row = await conn.fetchrow(
        f"""SELECT count(*)::int AS total_phones, count(DISTINCT p.brand_slug)::int AS total_brands,
min(p.price_usd) AS min_price, max(p.price_usd) AS max_price,
min(p.release_year) AS min_year, max(p.release_year) AS max_year,
max(p.battery_capacity) AS max_battery, max(p.ram_max_gb) AS max_ram, max(p.storage_max_gb) AS max_storage,
max(p.fast_charging_w) AS max_charging_w, max(p.refresh_rate_hz) AS max_refresh_rate,
max(p.main_camera_mp) AS max_camera_mp, min(p.screen_size) AS min_screen, max(p.screen_size) AS max_screen,
min(p.weight_g) AS min_weight, max(p.weight_g) AS max_weight
FROM phone_catalog p WHERE {LISTABLE} AND {NOT_REGIONAL}"""
    )
    tiers = await conn.fetch(
        f"SELECT DISTINCT p.tier FROM phone_catalog p WHERE {LISTABLE} AND p.tier IS NOT NULL ORDER BY 1"
    )
    stats = row_to_dict(row)
    stats["tiers"] = [r["tier"] for r in tiers]
    return stats


# ── trade-in ─────────────────────────────────────────────────────────────────

async def tradein_phone(conn: asyncpg.Connection, phone_id: int) -> dict | None:
    row = await conn.fetchrow(
        "SELECT id, brand, brand_slug, slug, name, display_name, price_usd, price_basis FROM phone_catalog WHERE id = $1", phone_id,
    )
    return row_to_dict(row)


# ── AI copy cache ────────────────────────────────────────────────────────────

async def get_cached_ai_copy(conn: asyncpg.Connection, key: str, max_age_days: int) -> dict | None:
    return await conn.fetchval(
        "SELECT payload FROM ai_copy_cache WHERE cache_key = $1 AND created_at > now() - make_interval(days => $2)",
        key, max_age_days,
    )


async def put_cached_ai_copy(conn: asyncpg.Connection, key: str, payload: dict) -> None:
    await conn.execute(
        "INSERT INTO ai_copy_cache (cache_key, payload) VALUES ($1, $2) "
        "ON CONFLICT (cache_key) DO UPDATE SET payload = EXCLUDED.payload, created_at = now()",
        key, payload,
    )