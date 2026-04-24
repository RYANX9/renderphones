"""
Mobylite API — V1
Phone search, filtering, detail, compare, brand pages, trending, category rankings.
No auth. No accounts. No user data. Pure phone data.
"""

from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from typing import Optional
import os
import json
import asyncpg
import re
from datetime import datetime, timedelta

# ─── DB ──────────────────────────────────────────────────────────────────────

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://neondb_owner:npg_c6nFi5XeBjIY@ep-shiny-feather-ag2vjll4-pooler.c-2.eu-central-1.aws.neon.tech/neondb?sslmode=require",
)

pool: asyncpg.Pool = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pool
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
    yield
    await pool.close()


app = FastAPI(title="Mobylite API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── HELPERS ─────────────────────────────────────────────────────────────────


def row_to_dict(row) -> dict:
    if row is None:
        return None
    d = dict(row)
    for k, v in d.items():
        if isinstance(v, datetime):
            d[k] = v.isoformat()
    return d


def rows_to_list(rows) -> list:
    return [row_to_dict(r) for r in rows]


def parse_json_safe(val) -> dict | list | None:
    if val is None:
        return None
    if isinstance(val, (dict, list)):
        return val
    try:
        return json.loads(val)
    except Exception:
        return None


def slug_to_words(slug: str) -> str:
    """galaxy-s25-ultra → galaxy s25 ultra"""
    return slug.replace("-", " ")


def build_phone_list_select() -> str:
    """Columns needed for card display — lightweight, no full_specifications."""
    return """
        id,
        model_name,
        brand,
        price_usd,
        main_image_url,
        screen_size,
        battery_capacity,
        ram_options,
        storage_options,
        main_camera_mp,
        chipset,
        antutu_score,
        amazon_link,
        release_year,
        release_month,
        release_day,
        release_date_full,
        weight_g,
        fast_charging_w
    """


def build_phone_detail_select() -> str:
    """All columns including full_specifications for detail page."""
    return """
        id,
        model_name,
        brand,
        price_usd,
        main_image_url,
        screen_size,
        battery_capacity,
        ram_options,
        storage_options,
        main_camera_mp,
        chipset,
        antutu_score,
        amazon_link,
        release_year,
        release_month,
        release_day,
        release_date_full,
        price_original,
        currency,
        brand_link,
        weight_g,
        thickness_mm,
        screen_resolution,
        fast_charging_w,
        video_resolution,
        geekbench_multi,
        gpu_score,
        full_specifications,
        features
    """


def compute_value_score(phone: dict, peers: list[dict]) -> float | None:
    """
    Value score: how good are this phone's specs relative to peers in ±30% price range.
    Returns 0.0-10.0 or None if insufficient data.
    """
    if not phone.get("price_usd") or not peers:
        return None

    def spec_score(p: dict) -> float:
        s = 0.0
        if p.get("antutu_score"):
            s += min(p["antutu_score"] / 2_000_000, 1.0) * 3.0
        if p.get("main_camera_mp"):
            s += min(p["main_camera_mp"] / 200, 1.0) * 2.0
        if p.get("battery_capacity"):
            s += min(p["battery_capacity"] / 7000, 1.0) * 2.0
        if p.get("ram_options"):
            max_ram = max(p["ram_options"]) if p["ram_options"] else 0
            s += min(max_ram / 16, 1.0) * 1.5
        if p.get("fast_charging_w"):
            s += min(p["fast_charging_w"] / 100, 1.0) * 1.0
        if p.get("screen_size"):
            s += min(p["screen_size"] / 7.0, 1.0) * 0.5
        return s

    this_score = spec_score(phone)
    if not peers:
        return round(min(this_score / 10.0 * 10, 10), 1)

    peer_scores = [spec_score(p) for p in peers if p.get("price_usd")]
    if not peer_scores:
        return None

    max_peer = max(peer_scores) or 1
    normalized = (this_score / max_peer) * 10
    return round(min(normalized, 10), 1)


def chipset_tier(chipset: str | None) -> str:
    if not chipset:
        return "unknown"
    c = chipset.lower()
    flagship_patterns = [
        r"snapdragon 8 gen \d",
        r"snapdragon 8 elite",
        r"snapdragon 8s gen \d",
        r"dimensity 9\d{3}",
        r"exynos 2\d{3}",
        r"apple a1[4-9]",
        r"apple a\d+ pro",
        r"tensor g[3-9]",
        r"kirin 99",
    ]
    for pat in flagship_patterns:
        if re.search(pat, c):
            return "flagship"
    mid_patterns = [
        r"snapdragon 7",
        r"snapdragon 6",
        r"dimensity 8\d{2}",
        r"dimensity 7\d{2}",
        r"exynos 1\d{3}",
        r"kirin 8",
    ]
    for pat in mid_patterns:
        if re.search(pat, c):
            return "mid"
    return "entry"


# ─── PHONE LIST / SEARCH ─────────────────────────────────────────────────────


@app.get("/phones/search")
async def search_phones(
    q: Optional[str] = Query(None, description="Text search across model name and brand"),
    brand: Optional[str] = Query(None),
    min_price: Optional[float] = Query(None),
    max_price: Optional[float] = Query(None),
    min_ram: Optional[int] = Query(None),
    min_battery: Optional[int] = Query(None),
    min_camera_mp: Optional[int] = Query(None),
    min_screen_size: Optional[float] = Query(None),
    max_screen_size: Optional[float] = Query(None),
    min_year: Optional[int] = Query(None),
    max_weight: Optional[int] = Query(None, description="Max weight in grams"),
    min_charging_w: Optional[int] = Query(None),
    has_5g: Optional[bool] = Query(None),
    chipset_tier_filter: Optional[str] = Query(None, alias="chipset_tier", description="flagship|mid|entry"),
    sort_by: str = Query("release_year", description="release_year|price_usd|battery_capacity|main_camera_mp|antutu_score|weight_g"),
    sort_order: str = Query("desc", description="asc|desc"),
    page: int = Query(1, ge=1),
    page_size: int = Query(24, ge=1, le=100),
):
    conditions = ["1=1"]
    params = []
    i = 1

    if q and q.strip():
        conditions.append(f"(LOWER(model_name) LIKE ${i} OR LOWER(brand) LIKE ${i})")
        params.append(f"%{q.strip().lower()}%")
        i += 1

    if brand:
        conditions.append(f"LOWER(brand) = ${i}")
        params.append(brand.lower())
        i += 1

    if min_price is not None:
        conditions.append(f"price_usd >= ${i}")
        params.append(min_price)
        i += 1

    if max_price is not None:
        conditions.append(f"price_usd <= ${i}")
        params.append(max_price)
        i += 1

    if min_ram is not None:
        conditions.append(f"${i} = ANY(ram_options)")
        params.append(min_ram)
        i += 1

    if min_battery is not None:
        conditions.append(f"battery_capacity >= ${i}")
        params.append(min_battery)
        i += 1

    if min_camera_mp is not None:
        conditions.append(f"main_camera_mp >= ${i}")
        params.append(min_camera_mp)
        i += 1

    if min_screen_size is not None:
        conditions.append(f"screen_size >= ${i}")
        params.append(min_screen_size)
        i += 1

    if max_screen_size is not None:
        conditions.append(f"screen_size <= ${i}")
        params.append(max_screen_size)
        i += 1

    if min_year is not None:
        conditions.append(f"release_year >= ${i}")
        params.append(min_year)
        i += 1

    if max_weight is not None:
        conditions.append(f"weight_g <= ${i}")
        params.append(max_weight)
        i += 1

    if min_charging_w is not None:
        conditions.append(f"fast_charging_w >= ${i}")
        params.append(min_charging_w)
        i += 1

    # chipset tier — filter post-fetch is too slow; use regex in SQL
    if chipset_tier_filter:
        if chipset_tier_filter == "flagship":
            conditions.append(
                "(LOWER(chipset) ~ 'snapdragon 8 gen|snapdragon 8 elite|dimensity 9[0-9]{3}|exynos 2[0-9]{3}|apple a1[4-9]|tensor g[3-9]')"
            )
        elif chipset_tier_filter == "mid":
            conditions.append(
                "(LOWER(chipset) ~ 'snapdragon [67]|dimensity [78][0-9]{2}|exynos 1[0-9]{3}')"
            )
        elif chipset_tier_filter == "entry":
            conditions.append(
                "(chipset IS NOT NULL AND LOWER(chipset) NOT LIKE '%snapdragon 8%' AND LOWER(chipset) NOT LIKE '%dimensity 9%')"
            )

    valid_sorts = {
        "release_year": "release_year",
        "price_usd": "price_usd",
        "battery_capacity": "battery_capacity",
        "main_camera_mp": "main_camera_mp",
        "antutu_score": "antutu_score",
        "weight_g": "weight_g",
    }
    sort_col = valid_sorts.get(sort_by, "release_year")
    order = "DESC" if sort_order.lower() == "desc" else "ASC"

    where = " AND ".join(conditions)
    offset = (page - 1) * page_size

    count_sql = f"SELECT COUNT(*) FROM phones WHERE {where}"
    data_sql = f"""
        SELECT {build_phone_list_select()}
        FROM phones
        WHERE {where}
        ORDER BY {sort_col} {order} NULLS LAST, id DESC
        LIMIT {page_size} OFFSET {offset}
    """

    async with pool.acquire() as conn:
        total = await conn.fetchval(count_sql, *params)
        rows = await conn.fetch(data_sql, *params)

    phones = rows_to_list(rows)

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "results": phones,
    }


# ─── PHONE DETAIL ─────────────────────────────────────────────────────────────


@app.get("/phones/{phone_id}")
async def get_phone(phone_id: int):
    sql = f"SELECT {build_phone_detail_select()} FROM phones WHERE id = $1"
    async with pool.acquire() as conn:
        row = await conn.fetchrow(sql, phone_id)
    if not row:
        raise HTTPException(status_code=404, detail="Phone not found")

    phone = row_to_dict(row)
    phone["full_specifications"] = parse_json_safe(phone.get("full_specifications"))

    # Compute value score vs peers in ±30% price band
    if phone.get("price_usd"):
        lo = phone["price_usd"] * 0.7
        hi = phone["price_usd"] * 1.3
        peer_sql = f"""
            SELECT {build_phone_list_select()}
            FROM phones
            WHERE price_usd BETWEEN $1 AND $2 AND id != $3
            LIMIT 50
        """
        async with pool.acquire() as conn:
            peer_rows = await conn.fetch(peer_sql, lo, hi, phone_id)
        peers = rows_to_list(peer_rows)
        phone["value_score"] = compute_value_score(phone, peers)
    else:
        phone["value_score"] = None

    phone["chipset_tier"] = chipset_tier(phone.get("chipset"))

    return phone


@app.get("/phones/{phone_id}/similar")
async def get_similar_phones(phone_id: int, limit: int = Query(12, ge=1, le=24)):
    """Find similar phones: same brand or same price tier with similar specs."""
    async with pool.acquire() as conn:
        base = await conn.fetchrow(
            "SELECT brand, price_usd, battery_capacity, chipset FROM phones WHERE id = $1",
            phone_id,
        )
    if not base:
        raise HTTPException(status_code=404, detail="Phone not found")

    params = [phone_id]
    conditions = ["id != $1"]
    i = 2

    if base["price_usd"]:
        lo = base["price_usd"] * 0.7
        hi = base["price_usd"] * 1.3
        conditions.append(f"price_usd BETWEEN ${i} AND ${i+1}")
        params += [lo, hi]
        i += 2

    where = " AND ".join(conditions)

    # Score same-brand higher in ORDER BY
    brand_escaped = base["brand"].replace("'", "''")
    sql = f"""
        SELECT {build_phone_list_select()},
               CASE WHEN brand = '{brand_escaped}' THEN 1 ELSE 0 END AS brand_match
        FROM phones
        WHERE {where}
        ORDER BY brand_match DESC, release_year DESC NULLS LAST
        LIMIT {limit}
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)

    results = []
    for r in rows:
        d = row_to_dict(r)
        d.pop("brand_match", None)
        results.append(d)

    return {"phones": results}


# ─── TRENDING ─────────────────────────────────────────────────────────────────


@app.get("/phones/trending")
async def get_trending(limit: int = Query(10, ge=1, le=20)):
    """
    Returns the most recently released phones with AnTuTu scores as a proxy for
    trending. In production you'd track page views; for V1 this is the best signal.
    """
    sql = f"""
        SELECT {build_phone_list_select()}
        FROM phones
        WHERE release_year IS NOT NULL AND antutu_score IS NOT NULL
        ORDER BY release_year DESC, antutu_score DESC
        LIMIT {limit}
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql)
    return {"phones": rows_to_list(rows)}


@app.get("/phones/latest")
async def get_latest(limit: int = Query(20, ge=1, le=50)):
    """Latest released phones for home page hero."""
    sql = f"""
        SELECT {build_phone_list_select()}
        FROM phones
        WHERE release_year IS NOT NULL
        ORDER BY release_year DESC, release_month DESC NULLS LAST, id DESC
        LIMIT {limit}
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql)
    return {"phones": rows_to_list(rows)}


# ─── BRANDS ───────────────────────────────────────────────────────────────────


@app.get("/brands")
async def get_brands():
    """All brands with phone counts, sorted by count descending."""
    sql = """
        SELECT brand, COUNT(*) as phone_count
        FROM phones
        WHERE brand IS NOT NULL
        GROUP BY brand
        ORDER BY phone_count DESC
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql)
    return {"brands": [{"brand": r["brand"], "count": r["phone_count"]} for r in rows]}


@app.get("/brands/{brand_slug}")
async def get_brand(brand_slug: str):
    """Brand page data: stats + latest phone."""
    brand_name = slug_to_words(brand_slug)

    # Try exact match first, then ILIKE
    sql = """
        SELECT brand, COUNT(*) as total,
               MIN(price_usd) as min_price,
               MAX(price_usd) as max_price,
               ROUND(AVG(price_usd)::numeric, 0) as avg_price,
               ROUND(AVG(battery_capacity)::numeric, 0) as avg_battery,
               MAX(release_year) as latest_year
        FROM phones
        WHERE LOWER(brand) = LOWER($1)
        GROUP BY brand
    """
    async with pool.acquire() as conn:
        stats = await conn.fetchrow(sql, brand_name)

    if not stats:
        raise HTTPException(status_code=404, detail="Brand not found")

    # Latest phone
    latest_sql = f"""
        SELECT {build_phone_list_select()}
        FROM phones
        WHERE LOWER(brand) = LOWER($1)
        ORDER BY release_year DESC, release_month DESC NULLS LAST
        LIMIT 1
    """
    async with pool.acquire() as conn:
        latest = await conn.fetchrow(latest_sql, brand_name)

    return {
        "brand": stats["brand"],
        "total_phones": stats["total"],
        "price_range": {
            "min": float(stats["min_price"]) if stats["min_price"] else None,
            "max": float(stats["max_price"]) if stats["max_price"] else None,
            "avg": float(stats["avg_price"]) if stats["avg_price"] else None,
        },
        "avg_battery": int(stats["avg_battery"]) if stats["avg_battery"] else None,
        "latest_year": stats["latest_year"],
        "latest_phone": row_to_dict(latest) if latest else None,
    }


@app.get("/brands/{brand_slug}/phones")
async def get_brand_phones(
    brand_slug: str,
    sort_by: str = Query("release_year"),
    sort_order: str = Query("desc"),
    page: int = Query(1, ge=1),
    page_size: int = Query(24, ge=1, le=100),
):
    """All phones for a brand, paginated."""
    brand_name = slug_to_words(brand_slug)
    valid_sorts = {"release_year", "price_usd", "battery_capacity", "main_camera_mp", "antutu_score"}
    sort_col = sort_by if sort_by in valid_sorts else "release_year"
    order = "DESC" if sort_order.lower() == "desc" else "ASC"
    offset = (page - 1) * page_size

    async with pool.acquire() as conn:
        total = await conn.fetchval(
            "SELECT COUNT(*) FROM phones WHERE LOWER(brand) = LOWER($1)", brand_name
        )
        rows = await conn.fetch(
            f"""
            SELECT {build_phone_list_select()}
            FROM phones
            WHERE LOWER(brand) = LOWER($1)
            ORDER BY {sort_col} {order} NULLS LAST
            LIMIT {page_size} OFFSET {offset}
            """,
            brand_name,
        )

    return {"total": total, "page": page, "page_size": page_size, "results": rows_to_list(rows)}


# ─── COMPARE ──────────────────────────────────────────────────────────────────


@app.get("/phones/compare")
async def compare_phones(ids: str = Query(..., description="Comma-separated phone IDs, 2-4")):
    """Fetch multiple phones for compare page."""
    try:
        id_list = [int(x.strip()) for x in ids.split(",") if x.strip()]
    except ValueError:
        raise HTTPException(status_code=400, detail="ids must be comma-separated integers")

    if len(id_list) < 2:
        raise HTTPException(status_code=400, detail="At least 2 phone IDs required")
    if len(id_list) > 4:
        raise HTTPException(status_code=400, detail="Maximum 4 phones")

    sql = f"""
        SELECT {build_phone_detail_select()}
        FROM phones
        WHERE id = ANY($1::int[])
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, id_list)

    phones = []
    for r in rows:
        d = row_to_dict(r)
        d["full_specifications"] = parse_json_safe(d.get("full_specifications"))
        d["chipset_tier"] = chipset_tier(d.get("chipset"))
        phones.append(d)

    # Preserve order of requested IDs
    phone_map = {p["id"]: p for p in phones}
    ordered = [phone_map[i] for i in id_list if i in phone_map]

    return {"phones": ordered}


# ─── CATEGORY RANKINGS ────────────────────────────────────────────────────────


CATEGORY_CONFIG = {
    "camera-phones": {
        "title": "Best Camera Phones",
        "description": "Ranked by main sensor resolution, sensor size, optical zoom, and video capability.",
        "sql": f"""
            SELECT {{cols}},
                   (
                     COALESCE(main_camera_mp, 0) * 0.3 +
                     COALESCE(antutu_score, 0) / 200000.0 * 0.1 +
                     CASE WHEN fast_charging_w IS NOT NULL THEN 5 ELSE 0 END
                   ) AS category_score
            FROM phones
            WHERE main_camera_mp IS NOT NULL AND release_year >= 2022
            ORDER BY main_camera_mp DESC, antutu_score DESC NULLS LAST
            LIMIT {{limit}}
        """,
    },
    "battery-life": {
        "title": "Best Battery Life",
        "description": "Highest battery capacity phones, weighted by efficiency.",
        "sql": f"""
            SELECT {{cols}},
                   battery_capacity::float AS category_score
            FROM phones
            WHERE battery_capacity IS NOT NULL AND release_year >= 2022
            ORDER BY battery_capacity DESC
            LIMIT {{limit}}
        """,
    },
    "gaming-phones": {
        "title": "Best Gaming Phones",
        "description": "Top AnTuTu and Geekbench scores — the fastest chips available.",
        "sql": f"""
            SELECT {{cols}},
                   COALESCE(antutu_score, 0)::float AS category_score
            FROM phones
            WHERE antutu_score IS NOT NULL AND release_year >= 2023
            ORDER BY antutu_score DESC
            LIMIT {{limit}}
        """,
    },
    "under-300": {
        "title": "Best Phones Under $300",
        "description": "Best specs-per-dollar under $300.",
        "sql": f"""
            SELECT {{cols}},
                   (
                     COALESCE(battery_capacity, 0) / 500.0 +
                     COALESCE(main_camera_mp, 0) / 10.0 +
                     COALESCE(antutu_score, 0) / 100000.0
                   ) AS category_score
            FROM phones
            WHERE price_usd <= 300 AND price_usd > 0 AND release_year >= 2022
            ORDER BY category_score DESC
            LIMIT {{limit}}
        """,
    },
    "under-500": {
        "title": "Best Phones Under $500",
        "description": "Best value in the $150-$500 range.",
        "sql": f"""
            SELECT {{cols}},
                   (
                     COALESCE(battery_capacity, 0) / 500.0 +
                     COALESCE(main_camera_mp, 0) / 10.0 +
                     COALESCE(antutu_score, 0) / 100000.0
                   ) AS category_score
            FROM phones
            WHERE price_usd <= 500 AND price_usd > 0 AND release_year >= 2022
            ORDER BY category_score DESC
            LIMIT {{limit}}
        """,
    },
    "lightweight": {
        "title": "Lightest Phones",
        "description": "Phones under 175g with flagship-or-better specs.",
        "sql": f"""
            SELECT {{cols}},
                   (1000.0 - COALESCE(weight_g, 999)) AS category_score
            FROM phones
            WHERE weight_g IS NOT NULL AND weight_g <= 200 AND release_year >= 2022
            ORDER BY weight_g ASC
            LIMIT {{limit}}
        """,
    },
    "compact-phones": {
        "title": "Best Compact Phones",
        "description": "Screen size under 6.3 inches — phones that fit comfortably in one hand.",
        "sql": f"""
            SELECT {{cols}},
                   COALESCE(antutu_score, 0)::float AS category_score
            FROM phones
            WHERE screen_size <= 6.3 AND screen_size IS NOT NULL AND release_year >= 2022
            ORDER BY antutu_score DESC NULLS LAST, release_year DESC
            LIMIT {{limit}}
        """,
    },
    "fast-charging": {
        "title": "Fastest Charging Phones",
        "description": "Wired charging speed in watts — less waiting, more using.",
        "sql": f"""
            SELECT {{cols}},
                   COALESCE(fast_charging_w, 0)::float AS category_score
            FROM phones
            WHERE fast_charging_w IS NOT NULL AND fast_charging_w > 0 AND release_year >= 2022
            ORDER BY fast_charging_w DESC
            LIMIT {{limit}}
        """,
    },
}


@app.get("/categories")
async def list_categories():
    return {
        "categories": [
            {"slug": slug, "title": cfg["title"], "description": cfg["description"]}
            for slug, cfg in CATEGORY_CONFIG.items()
        ]
    }


@app.get("/categories/{category_slug}")
async def get_category(category_slug: str, limit: int = Query(10, ge=5, le=20)):
    cfg = CATEGORY_CONFIG.get(category_slug)
    if not cfg:
        raise HTTPException(status_code=404, detail=f"Category '{category_slug}' not found")

    cols = build_phone_list_select()
    sql = cfg["sql"].format(cols=cols, limit=limit)

    async with pool.acquire() as conn:
        rows = await conn.fetch(sql)

    phones = []
    for r in rows:
        d = row_to_dict(r)
        d["category_score"] = round(float(d.pop("category_score", 0) or 0), 2)
        phones.append(d)

    return {
        "slug": category_slug,
        "title": cfg["title"],
        "description": cfg["description"],
        "phones": phones,
    }


# ─── FILTERS / STATS ──────────────────────────────────────────────────────────


@app.get("/filters/stats")
async def get_filter_stats():
    """
    Returns ranges and counts needed to build filter UI.
    Called once on home page load to populate slider ranges.
    """
    sql = """
        SELECT
            COUNT(*) as total,
            COUNT(DISTINCT brand) as total_brands,
            MIN(price_usd) as min_price,
            MAX(price_usd) as max_price,
            MIN(battery_capacity) as min_battery,
            MAX(battery_capacity) as max_battery,
            MIN(screen_size) as min_screen,
            MAX(screen_size) as max_screen,
            MIN(weight_g) as min_weight,
            MAX(weight_g) as max_weight,
            MIN(fast_charging_w) as min_charging,
            MAX(fast_charging_w) as max_charging,
            MIN(release_year) as min_year,
            MAX(release_year) as max_year
        FROM phones
        WHERE price_usd > 0
    """

    brands_sql = """
        SELECT brand, COUNT(*) as count
        FROM phones
        WHERE brand IS NOT NULL
        GROUP BY brand
        ORDER BY count DESC
    """

    ram_sql = """
        SELECT DISTINCT unnest(ram_options) as ram
        FROM phones
        WHERE ram_options IS NOT NULL
        ORDER BY ram
    """

    async with pool.acquire() as conn:
        stats = await conn.fetchrow(sql)
        brands = await conn.fetch(brands_sql)
        rams = await conn.fetch(ram_sql)

    return {
        "total_phones": stats["total"],
        "total_brands": stats["total_brands"],
        "price_range": {
            "min": float(stats["min_price"]) if stats["min_price"] else 0,
            "max": float(stats["max_price"]) if stats["max_price"] else 5000,
        },
        "battery_range": {
            "min": int(stats["min_battery"]) if stats["min_battery"] else 1000,
            "max": int(stats["max_battery"]) if stats["max_battery"] else 10000,
        },
        "screen_range": {
            "min": float(stats["min_screen"]) if stats["min_screen"] else 4.0,
            "max": float(stats["max_screen"]) if stats["max_screen"] else 7.5,
        },
        "weight_range": {
            "min": int(stats["min_weight"]) if stats["min_weight"] else 100,
            "max": int(stats["max_weight"]) if stats["max_weight"] else 300,
        },
        "charging_range": {
            "min": int(stats["min_charging"]) if stats["min_charging"] else 5,
            "max": int(stats["max_charging"]) if stats["max_charging"] else 240,
        },
        "year_range": {
            "min": int(stats["min_year"]) if stats["min_year"] else 2018,
            "max": int(stats["max_year"]) if stats["max_year"] else 2025,
        },
        "brands": [{"brand": r["brand"], "count": r["count"]} for r in brands],
        "ram_options": [r["ram"] for r in rams if r["ram"]],
    }


# ─── HELP ME CHOOSE ───────────────────────────────────────────────────────────


@app.get("/recommend")
async def recommend(
    min_price: Optional[float] = Query(None),
    max_price: Optional[float] = Query(None),
    priorities: str = Query(..., description="Comma-separated: camera,battery,performance,compact,lightweight,display,fast_charging,value"),
    limit: int = Query(5, ge=3, le=10),
):
    """
    Returns top phones matching budget + priorities.
    Scoring: each priority contributes weighted spec scores, normalized to 0-10.
    """
    priority_list = [p.strip().lower() for p in priorities.split(",") if p.strip()]

    conditions = ["1=1"]
    params = []
    i = 1

    if min_price is not None:
        conditions.append(f"price_usd >= ${i}")
        params.append(min_price)
        i += 1

    if max_price is not None:
        conditions.append(f"price_usd <= ${i}")
        params.append(max_price)
        i += 1

    where = " AND ".join(conditions)

    sql = f"""
        SELECT {build_phone_list_select()}
        FROM phones
        WHERE {where} AND release_year >= 2022
        ORDER BY release_year DESC
        LIMIT 200
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)

    phones = rows_to_list(rows)
    if not phones:
        return {"phones": []}

    # Normalizers
    def norm(val, lo, hi):
        if val is None or hi == lo:
            return 0.0
        return max(0.0, min(1.0, (val - lo) / (hi - lo)))

    cam_max = max((p["main_camera_mp"] or 0) for p in phones) or 1
    bat_max = max((p["battery_capacity"] or 0) for p in phones) or 1
    ant_max = max((p["antutu_score"] or 0) for p in phones) or 1
    chg_max = max((p["fast_charging_w"] or 0) for p in phones) or 1
    screen_max = max((p["screen_size"] or 0) for p in phones) or 1
    screen_min = min((p["screen_size"] or 100) for p in phones) or 1
    weight_max = max((p["weight_g"] or 0) for p in phones) or 1

    def score_phone(p: dict, priorities: list[str]) -> float:
        s = 0.0
        n = len(priorities) or 1
        for pr in priorities:
            if pr == "camera":
                s += norm(p["main_camera_mp"] or 0, 0, cam_max)
            elif pr == "battery":
                s += norm(p["battery_capacity"] or 0, 0, bat_max)
            elif pr == "performance":
                s += norm(p["antutu_score"] or 0, 0, ant_max)
            elif pr == "compact":
                s += 1.0 - norm(p["screen_size"] or screen_max, screen_min, screen_max)
            elif pr == "lightweight":
                s += 1.0 - norm(p["weight_g"] or weight_max, 0, weight_max)
            elif pr == "display":
                # screen size + inferred AMOLED bonus
                s += norm(p["screen_size"] or 0, screen_min, screen_max) * 0.7
                s += 0.3  # assume display quality is uniform; chipset proxy
            elif pr == "fast_charging":
                s += norm(p["fast_charging_w"] or 0, 0, chg_max)
            elif pr == "value":
                # specs / price
                spec = (
                    (p["main_camera_mp"] or 0) / 200 +
                    (p["battery_capacity"] or 0) / 7000 +
                    (p["antutu_score"] or 0) / 2_000_000
                )
                price_norm = p["price_usd"] / 2000 if p["price_usd"] else 1
                s += spec / max(price_norm, 0.01) / 3
        return (s / n) * 10

    scored = []
    for p in phones:
        match_score = round(score_phone(p, priority_list), 1)
        p["match_score"] = match_score
        scored.append(p)

    scored.sort(key=lambda x: x["match_score"], reverse=True)
    top = scored[:limit]

    return {"phones": top, "priorities": priority_list}


# ─── HEALTH ───────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    try:
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return {"status": "ok", "db": "connected"}
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "error", "detail": str(e)})
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
