from __future__ import annotations

import logging
from typing import Optional

import anyio
from fastapi import APIRouter, HTTPException, Query

from ..core.cache import cached
from ..core.config import settings
from ..core.database import get_pool
from ..core.query import FilterParams
from ..core.shaping import attach_computed_fields, pop_smart_score
from ..services import phone_repo
from ..services.compare_copy import generate_compare_verdict
from ..services.recommend_copy import generate_match_copy
from ..services.recommend_service import TIER_BOUNDS, recommend as recommend_svc

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/phones", tags=["phones"])


@router.get("/search")
async def search_phones(
    q: Optional[str] = Query(None),
    brand: Optional[str] = Query(None),
    brands: Optional[str] = Query(None, description="Comma-separated brand list"),
    chipset: Optional[str] = Query(None),
    min_price: Optional[float] = Query(None),
    max_price: Optional[float] = Query(None),
    min_ram: Optional[int] = Query(None),
    min_storage: Optional[int] = Query(None),
    min_battery: Optional[int] = Query(None),
    min_camera_mp: Optional[int] = Query(None),
    min_screen_size: Optional[float] = Query(None),
    max_screen_size: Optional[float] = Query(None),
    min_year: Optional[int] = Query(None),
    max_year: Optional[int] = Query(None),
    max_weight: Optional[int] = Query(None),
    min_charging_w: Optional[int] = Query(None),
    min_refresh_rate: Optional[int] = Query(None),
    min_antutu: Optional[int] = Query(None),
    chipset_tier: Optional[str] = Query(None),
    has_nfc: Optional[bool] = Query(None),
    has_ois: Optional[bool] = Query(None),
    has_wireless_charging: Optional[bool] = Query(None),
    has_headphone_jack: Optional[bool] = Query(None),
    is_foldable: Optional[bool] = Query(None),
    is_premium_gaming: Optional[bool] = Query(None),
    camera_setup_type: Optional[str] = Query(None),
    water_resistant: Optional[bool] = Query(None),
    sort_by: str = Query("relevance"),
    sort_order: str = Query("desc"),
    page: int = Query(1, ge=1),
    page_size: int = Query(24, ge=1, le=100),
):
    filters = FilterParams(
        q=q, brand=brand,
        brands=[b.strip() for b in brands.split(",") if b.strip()] if brands else None,
        chipset=chipset,
        min_price=min_price, max_price=max_price,
        min_ram=min_ram, min_storage=min_storage,
        min_battery=min_battery, min_camera_mp=min_camera_mp,
        min_screen_size=min_screen_size, max_screen_size=max_screen_size,
        min_year=min_year, max_year=max_year, max_weight=max_weight,
        min_charging_w=min_charging_w, min_refresh_rate=min_refresh_rate,
        min_antutu=min_antutu, chipset_tier=chipset_tier,
        has_nfc=has_nfc, has_ois=has_ois, has_wireless_charging=has_wireless_charging,
        has_headphone_jack=has_headphone_jack, is_foldable=is_foldable,
        is_premium_gaming=is_premium_gaming, camera_setup_type=camera_setup_type,
        water_resistant=water_resistant,
    )

    async with get_pool().acquire() as conn:
        total, phones = await phone_repo.search(
            conn, filters=filters, sort_by=sort_by, sort_order=sort_order,
            page=page, page_size=page_size,
        )

    return {"total": total, "page": page, "page_size": page_size, "results": phones}


@router.get("/latest")
async def latest_phones(limit: int = Query(20, ge=1, le=100)):
    async def _fetch():
        async with get_pool().acquire() as conn:
            return {"phones": await phone_repo.latest(conn, limit)}

    return await cached(f"phones:latest:{limit}", settings.cache_ttl_stable, _fetch)


@router.get("/trending")
async def trending_phones(limit: int = Query(10, ge=1, le=50)):
    async def _fetch():
        async with get_pool().acquire() as conn:
            return {"phones": await phone_repo.trending(conn, limit)}

    return await cached(f"phones:trending:{limit}", settings.cache_ttl_trending, _fetch)


@router.get("/compare")
async def compare_phones(
    ids: Optional[str] = Query(None),
    slugs: Optional[str] = Query(None),
):
    if not ids and not slugs:
        raise HTTPException(status_code=400, detail="Provide `ids` or `slugs`.")

    async with get_pool().acquire() as conn:
        if ids:
            id_list = [int(i) for i in ids.split(",") if i.strip().isdigit()]
            if not id_list:
                raise HTTPException(status_code=400, detail="No valid ids provided.")
            phones = await phone_repo.compare_by_ids(conn, id_list)
        else:
            slug_list = [s.strip() for s in slugs.split(",") if s.strip()]
            if not slug_list:
                raise HTTPException(status_code=400, detail="No valid slugs provided.")
            phones = await phone_repo.compare_by_slugs(conn, slug_list)

    if not phones:
        raise HTTPException(status_code=404, detail="No phones found for the given ids/slugs.")

    # Blocking HTTP call to Gemini offloaded to a worker thread so it
    # doesn't stall the event loop for the round trip duration.
    verdict = None
    if len(phones) >= 2:
        verdict = await anyio.to_thread.run_sync(generate_compare_verdict, phones)

    return {"phones": phones, "verdict": verdict}


@router.get("/recommend")
async def recommend_phones(
    priorities: str = Query(..., description="Comma-separated priority ids"),
    tier: Optional[str] = Query(None, description="s|a|b|c|d — overrides min_price/max_price when set"),
    min_price: Optional[float] = Query(None),
    max_price: Optional[float] = Query(None),
    limit: int = Query(5, ge=1, le=20),
):
    if tier and tier in TIER_BOUNDS:
        min_price, max_price = TIER_BOUNDS[tier]

    priority_list = [p.strip() for p in priorities.split(",") if p.strip()]
    if not priority_list:
        raise HTTPException(status_code=400, detail="No valid priorities provided.")

    async with get_pool().acquire() as conn:
        result = await recommend_svc(
            conn, priorities=priority_list, min_price=min_price, max_price=max_price, limit=limit,
        )

    if result["phones"]:
        if result["requested_price_range"]["min"] and result["requested_price_range"]["max"]:
            budget_label = f"${result['requested_price_range']['min']:.0f}-${result['requested_price_range']['max']:.0f}"
        elif result["requested_price_range"]["max"]:
            budget_label = f"under ${result['requested_price_range']['max']:.0f}"
        elif result["requested_price_range"]["min"]:
            budget_label = f"${result['requested_price_range']['min']:.0f}+"
        else:
            budget_label = "any budget"

        match_copy = await anyio.to_thread.run_sync(
            generate_match_copy, result["phones"], result["priorities"], budget_label,
        )
        if match_copy:
            for p in result["phones"]:
                copy = match_copy.get(p["id"])
                if copy:
                    p["match_line"] = copy["match_line"]
                    p["tradeoff_line"] = copy["tradeoff_line"]

    return result


@router.get("/{phone_id}")
async def get_phone(phone_id: str):
    async with get_pool().acquire() as conn:
        phone = await phone_repo.get_by_id_or_slug(conn, phone_id)
        if phone is None:
            raise HTTPException(status_code=404, detail=f"Phone '{phone_id}' not found.")

        phone["variants"] = await phone_repo.fetch_variants(conn, phone["id"])
        phone["images"] = await phone_repo.fetch_images(conn, phone["id"])
        phone["features"] = await phone_repo.fetch_features(conn, phone["id"])

        price = await phone_repo.latest_price_point(conn, phone["id"])
        phone_repo.apply_latest_price(phone, price)

        peers = (
            await phone_repo.fetch_value_peers(conn, phone)
            if phone.get("smart_value_score") is None
            else []
        )

    attach_computed_fields([phone], peers=peers or [phone])
    phone["smart_score"] = pop_smart_score(phone)
    return phone


@router.get("/{phone_id}/variants")
async def get_phone_variants(phone_id: int):
    async with get_pool().acquire() as conn:
        exists = await conn.fetchval("SELECT 1 FROM phones WHERE id = $1", phone_id)
        if not exists:
            raise HTTPException(status_code=404, detail=f"Phone {phone_id} not found.")
        variants = await phone_repo.fetch_variants(conn, phone_id)
    return {"phone_id": phone_id, "variants": variants}


@router.get("/{phone_id}/similar")
async def similar_phones_route(phone_id: int, limit: int = Query(12, ge=1, le=30)):
    async with get_pool().acquire() as conn:
        exists = await conn.fetchval("SELECT 1 FROM phones WHERE id = $1", phone_id)
        if not exists:
            raise HTTPException(status_code=404, detail=f"Phone {phone_id} not found.")
        phones = await phone_repo.similar_phones(conn, phone_id, limit)
    return {"phones": phones}


@router.get("/{phone_id}/price-history")
async def price_history(
    phone_id: int,
    condition: str = Query("new", description="'new', 'used', or 'all'."),
    scope: str = Query("global", description="'global', 'local', or 'all'."),
):
    async with get_pool().acquire() as conn:
        exists = await conn.fetchval("SELECT 1 FROM phones WHERE id = $1", phone_id)
        if not exists:
            raise HTTPException(status_code=404, detail=f"Phone {phone_id} not found.")

        if condition == "all":
            history_rows = await conn.fetch(
                """
                SELECT snapshot_date, condition, min_price_usd, max_price_usd,
                       avg_price_usd, listing_count
                FROM price_history
                WHERE phone_id = $1
                ORDER BY snapshot_date ASC, condition ASC
                """,
                phone_id,
            )
        else:
            history_rows = await conn.fetch(
                """
                SELECT snapshot_date, condition, min_price_usd, max_price_usd,
                       avg_price_usd, listing_count
                FROM price_history
                WHERE phone_id = $1 AND condition = $2
                ORDER BY snapshot_date ASC
                """,
                phone_id, condition,
            )

        if scope == "all":
            point_rows = await conn.fetch(
                """
                SELECT snapshot_date, scope, price_usd
                FROM price_points
                WHERE phone_id = $1
                ORDER BY snapshot_date ASC, scope ASC
                """,
                phone_id,
            )
        else:
            point_rows = await conn.fetch(
                """
                SELECT snapshot_date, scope, price_usd
                FROM price_points
                WHERE phone_id = $1 AND scope = $2
                ORDER BY snapshot_date ASC
                """,
                phone_id, scope,
            )

    from ..core.database import rows_to_list
    return {
        "phone_id": phone_id,
        "points": rows_to_list(history_rows),
        "price_points": rows_to_list(point_rows),
    }
