from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from cache import cached
from config import settings
from database import (
    get_pool, row_to_dict, rows_to_list,
    PHONE_LIST_SELECT, RELEASE_TS_EXPR,
)
from utils.query import build_search_where, resolve_sort, parse_json_safe
from utils.scoring import chipset_tier, compute_value_score

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/phones", tags=["phones"])


# ── static routes — must come before /{phone_id} ─────────────────────────────

@router.get("/search")
async def search_phones(
    q:                  Optional[str]   = Query(None),
    brand:              Optional[str]   = Query(None),
    min_price:          Optional[float] = Query(None),
    max_price:          Optional[float] = Query(None),
    min_ram:            Optional[int]   = Query(None),
    min_battery:        Optional[int]   = Query(None),
    min_camera_mp:      Optional[int]   = Query(None),
    min_screen_size:    Optional[float] = Query(None),
    max_screen_size:    Optional[float] = Query(None),
    min_year:           Optional[int]   = Query(None),
    max_weight:         Optional[int]   = Query(None),
    min_charging_w:     Optional[int]   = Query(None),
    chipset_tier_param: Optional[str]   = Query(None, alias="chipset_tier"),
    sort_by:            str             = Query("release_year"),
    sort_order:         str             = Query("desc"),
    page:               int             = Query(1,  ge=1),
    page_size:          int             = Query(24, ge=1, le=100),
):
    where, params = build_search_where(
        q=q,
        brand=brand,
        min_price=min_price,
        max_price=max_price,
        min_ram=min_ram,
        min_battery=min_battery,
        min_camera_mp=min_camera_mp,
        min_screen_size=min_screen_size,
        max_screen_size=max_screen_size,
        min_year=min_year,
        max_weight=max_weight,
        min_charging_w=min_charging_w,
        chipset_tier_filter=chipset_tier_param,
    )
    sort_expr, order = resolve_sort(sort_by, sort_order)
    offset = (page - 1) * page_size

    async with get_pool().acquire() as conn:
        total = await conn.fetchval(
            f"SELECT COUNT(*) FROM phones WHERE {where}",
            *params,
        )
        rows = await conn.fetch(
            f"""
            SELECT {PHONE_LIST_SELECT}
            FROM   phones
            WHERE  {where}
            ORDER  BY {sort_expr} {order} NULLS LAST, id DESC
            LIMIT  {page_size} OFFSET {offset}
            """,
            *params,
        )

    return {"total": total, "page": page, "page_size": page_size, "results": rows_to_list(rows)}


@router.get("/trending")
async def get_trending(limit: int = Query(10, ge=1, le=20)):
    async def _fetch():
        async with get_pool().acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT {PHONE_LIST_SELECT}
                FROM   phones
                WHERE  release_year IS NOT NULL
                  AND  antutu_score IS NOT NULL
                ORDER  BY {RELEASE_TS_EXPR} DESC, antutu_score DESC
                LIMIT  $1
                """,
                limit,
            )
        return {"phones": rows_to_list(rows)}

    return await cached(f"trending:{limit}", settings.cache_ttl_trending, _fetch)


@router.get("/latest")
async def get_latest(limit: int = Query(20, ge=1, le=50)):
    async def _fetch():
        async with get_pool().acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT {PHONE_LIST_SELECT}
                FROM   phones
                WHERE  release_year IS NOT NULL
                ORDER  BY {RELEASE_TS_EXPR} DESC, id DESC
                LIMIT  $1
                """,
                limit,
            )
        return {"phones": rows_to_list(rows)}

    return await cached(f"latest:{limit}", settings.cache_ttl_stable, _fetch)


@router.get("/compare")
async def compare_phones(
    ids:   Optional[str] = Query(None),
    slugs: Optional[str] = Query(None),
):
    if not ids and not slugs:
        raise HTTPException(status_code=400, detail="Provide ids or slugs.")

    if slugs:
        slug_list = [s.strip() for s in slugs.split(",") if s.strip()]
        if len(slug_list) > 4:
            raise HTTPException(status_code=400, detail="Maximum 4 phones.")
        async with get_pool().acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM phones WHERE slug = ANY($1::text[])",
                slug_list,
            )
        key_field  = "slug"
        order_keys: list = slug_list
    else:
        try:
            id_list = [int(x.strip()) for x in ids.split(",") if x.strip()]
        except ValueError:
            raise HTTPException(status_code=400, detail="ids must be comma-separated integers.")
        if len(id_list) < 2:
            raise HTTPException(status_code=400, detail="At least 2 phone IDs required.")
        if len(id_list) > 4:
            raise HTTPException(status_code=400, detail="Maximum 4 phones.")
        async with get_pool().acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM phones WHERE id = ANY($1::int[])",
                id_list,
            )
        key_field  = "id"
        order_keys = id_list

    phones = []
    for r in rows:
        d = row_to_dict(r)
        d["full_specifications"] = parse_json_safe(d.get("full_specifications"))
        d["chipset_tier"] = chipset_tier(d.get("chipset"))
        phones.append(d)

    phone_map = {p[key_field]: p for p in phones}
    return {"phones": [phone_map[k] for k in order_keys if k in phone_map]}


@router.get("/recommend")
async def recommend(
    min_price:  Optional[float] = Query(None),
    max_price:  Optional[float] = Query(None),
    priorities: str             = Query(...),
    limit:      int             = Query(5, ge=3, le=10),
):
    priority_list = [p.strip().lower() for p in priorities.split(",") if p.strip()]
    if not priority_list:
        raise HTTPException(status_code=400, detail="At least one priority required.")

    conditions = ["release_year >= 2022", "price_usd > 0"]
    params: list = []
    i = 1

    if min_price is not None:
        conditions.append(f"price_usd >= ${i}"); params.append(min_price); i += 1
    if max_price is not None:
        conditions.append(f"price_usd <= ${i}"); params.append(max_price); i += 1

    where = " AND ".join(conditions)

    async with get_pool().acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT {PHONE_LIST_SELECT}
            FROM   phones
            WHERE  {where}
            ORDER  BY release_year DESC, id DESC
            LIMIT  200
            """,
            *params,
        )

    phones = rows_to_list(rows)
    if not phones:
        return {"phones": [], "priorities": priority_list}

    def _max(key: str) -> float:
        return max((p[key] or 0) for p in phones) or 1.0

    def _norm(val: float | None, lo: float, hi: float) -> float:
        if hi == lo:
            return 0.0
        return max(0.0, min(1.0, ((val or 0) - lo) / (hi - lo)))

    cam_max = _max("main_camera_mp")
    bat_max = _max("battery_capacity")
    ant_max = _max("antutu_score")
    chg_max = _max("fast_charging_w")
    scr_max = _max("screen_size")
    scr_min = min((p["screen_size"] or 100) for p in phones) or 1.0
    wgt_max = _max("weight_g")

    def _score(p: dict) -> float:
        s = 0.0
        for pr in priority_list:
            if   pr == "camera":        s += _norm(p["main_camera_mp"],  0, cam_max)
            elif pr == "battery":       s += _norm(p["battery_capacity"], 0, bat_max)
            elif pr == "performance":   s += _norm(p["antutu_score"],     0, ant_max)
            elif pr == "compact":       s += 1.0 - _norm(p["screen_size"] or scr_max, scr_min, scr_max)
            elif pr == "lightweight":   s += 1.0 - _norm(p["weight_g"]   or wgt_max, 0, wgt_max)
            elif pr == "display":       s += _norm(p["screen_size"] or 0, scr_min, scr_max) * 0.7 + 0.3
            elif pr == "fast_charging": s += _norm(p["fast_charging_w"],  0, chg_max)
            elif pr == "value":
                spec = (
                    (p["main_camera_mp"]  or 0) / 200
                    + (p["battery_capacity"] or 0) / 7000
                    + (p["antutu_score"]     or 0) / 2_000_000
                )
                price_n = p["price_usd"] / 2000 if p["price_usd"] else 1.0
                s += spec / max(price_n, 0.01) / 3
        return round((s / len(priority_list)) * 10, 1)

    for p in phones:
        p["match_score"] = _score(p)

    phones.sort(key=lambda x: x["match_score"], reverse=True)
    return {"phones": phones[:limit], "priorities": priority_list}


# ── parameterized routes ──────────────────────────────────────────────────────

@router.get("/{phone_id}")
async def get_phone(phone_id: int):
    async def _fetch():
        async with get_pool().acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM phones WHERE id = $1",
                phone_id,
            )
            if not row:
                return None

            phone = row_to_dict(row)
            phone["full_specifications"] = parse_json_safe(phone.get("full_specifications"))

            if phone.get("price_usd"):
                lo = phone["price_usd"] * 0.70
                hi = phone["price_usd"] * 1.30
                peer_rows = await conn.fetch(
                    """
                    SELECT id, model_name, brand, price_usd, antutu_score,
                           main_camera_mp, battery_capacity, ram_options,
                           fast_charging_w, screen_size
                    FROM   phones
                    WHERE  price_usd BETWEEN $1 AND $2
                      AND  id != $3
                    ORDER  BY id
                    LIMIT  50
                    """,
                    lo, hi, phone_id,
                )
                phone["value_score"] = compute_value_score(phone, rows_to_list(peer_rows))
            else:
                phone["value_score"] = None

            phone["chipset_tier"] = chipset_tier(phone.get("chipset"))
            return phone

    result = await cached(f"phone:{phone_id}", settings.cache_ttl_phone_detail, _fetch)
    if result is None:
        raise HTTPException(status_code=404, detail="Phone not found.")
    return result


@router.get("/{phone_id}/similar")
async def get_similar(phone_id: int, limit: int = Query(12, ge=1, le=24)):
    async def _fetch():
        async with get_pool().acquire() as conn:
            base = await conn.fetchrow(
                """
                SELECT id, brand, price_usd, screen_size,
                       battery_capacity, antutu_score, release_year
                FROM   phones
                WHERE  id = $1
                """,
                phone_id,
            )
            if not base:
                return None

            price_usd        = float(base["price_usd"])        if base["price_usd"]        is not None else None
            screen_size      = float(base["screen_size"])      if base["screen_size"]      is not None else None
            antutu_score     = float(base["antutu_score"])     if base["antutu_score"]     is not None else None
            battery_capacity = float(base["battery_capacity"]) if base["battery_capacity"] is not None else None
            brand            = base["brand"]

            conditions = ["p.id != $1", "p.release_year >= 2020"]
            params: list = [phone_id]
            idx = 2

            if price_usd is not None:
                conditions.append(f"p.price_usd BETWEEN ${idx} AND ${idx + 1}")
                params += [price_usd * 0.60, price_usd * 1.40]
                idx += 2

            if screen_size is not None:
                conditions.append(f"p.screen_size BETWEEN ${idx} AND ${idx + 1}")
                params += [screen_size - 0.7, screen_size + 0.7]
                idx += 2

            where = " AND ".join(conditions)
            params += [brand, antutu_score or 0.0, battery_capacity or 0.0]
            brand_i = idx
            ant_i   = idx + 1
            bat_i   = idx + 2

            rows = await conn.fetch(
                f"""
                SELECT
                    p.id, p.model_name, p.brand, p.price_usd, p.main_image_url,
                    p.release_year, p.release_month, p.release_day,
                    p.main_camera_mp, p.battery_capacity, p.screen_size,
                    p.weight_g, p.chipset, p.ram_options, p.storage_options,
                    p.fast_charging_w, p.antutu_score,
                    EXTRACT(EPOCH FROM MAKE_DATE(
                        COALESCE(p.release_year, 1970),
                        COALESCE(p.release_month, 1),
                        COALESCE(p.release_day, 1)
                    ))::bigint AS release_ts
                FROM phones p
                WHERE {where}
                ORDER BY (
                    CASE WHEN p.brand = ${brand_i} THEN 3 ELSE 0 END
                    + CASE
                        WHEN ${ant_i}::float > 0 AND p.antutu_score IS NOT NULL
                             AND ABS(p.antutu_score::float - ${ant_i}::float)
                                 / NULLIF(${ant_i}::float, 0) < 0.20
                        THEN 2 ELSE 0
                      END
                    + CASE
                        WHEN ${bat_i}::float > 0 AND p.battery_capacity IS NOT NULL
                             AND ABS(p.battery_capacity::float - ${bat_i}::float)
                                 / NULLIF(${bat_i}::float, 0) < 0.20
                        THEN 1 ELSE 0
                      END
                ) DESC,
                p.release_year DESC NULLS LAST
                LIMIT {limit}
                """,
                *params,
            )

        return {"phones": rows_to_list(rows)}

    result = await cached(f"similar:{phone_id}:{limit}", settings.cache_ttl_stable, _fetch)
    if result is None:
        raise HTTPException(status_code=404, detail="Phone not found.")
    return result


@router.get("/{phone_id}/stats")
async def get_stats(_phone_id: int):
    return {
        "success": True,
        "stats": {
            "average_rating": 0,
            "total_reviews": 0,
            "total_favorites": 0,
            "total_owners": 0,
            "rating_distribution": {"5": 0, "4": 0, "3": 0, "2": 0, "1": 0},
            "verified_owners_percentage": 0,
        },
    }


@router.get("/{phone_id}/also-compared")
async def also_compared(_phone_id: int):
    return {"phones": []}
