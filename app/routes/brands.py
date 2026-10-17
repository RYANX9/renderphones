from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request

from ..core.cache import cached
from ..core.config import settings
from ..core.database import acquire_conn
from ..core.query import FilterParams
from ..core.seo import brand_indexable
from ..core.slugs import brand_slug_of
from ..services import next_steps, phone_repo
from .phones import _csv, _region, qkey

router = APIRouter(prefix="/brands", tags=["brands"])


@router.get("")
async def list_brands():
    async def fetch():
        async with acquire_conn() as conn:
            return {"brands": await phone_repo.brand_list(conn)}
    return await cached("brands:all", settings.cache_ttl_stable, fetch)


@router.get("/{brand_slug}")
async def get_brand(brand_slug: str):
    slug = brand_slug_of(brand_slug)

    async def fetch():
        async with acquire_conn() as conn:
            brand = await phone_repo.brand_detail(conn, slug)
            if brand is None:
                return None
            all_brands = await phone_repo.brand_list(conn)
            brand["next_steps"] = await next_steps.for_brand(conn, brand, all_brands)
            brand["indexable"] = brand_indexable(brand["total_phones"])
            brand["similar_brands"] = [
                {"brand": b["brand"], "brand_slug": b["brand_slug"], "path": b["path"], "phone_count": b["phone_count"]}
                for b in sorted(
                    (b for b in all_brands if b["brand_slug"] != slug and b.get("avg_price") and brand["price_range"]["avg"]),
                    key=lambda b: abs(b["avg_price"] - brand["price_range"]["avg"]),
                )[:3]
            ]
            return brand

    result = await cached(f"brand:v2:{slug}", settings.cache_ttl_stable, fetch)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Brand '{brand_slug}' not found.")
    return result


@router.get("/{brand_slug}/phones")
async def get_brand_phones(
    request: Request, brand_slug: str,
    sort_by: str = Query("release_year"), sort_order: str = Query("desc"),
    page: int = Query(1, ge=1, le=500), page_size: int = Query(24, ge=1, le=100),
    min_price: Optional[float] = Query(None), max_price: Optional[float] = Query(None),
    min_ram: Optional[int] = Query(None), min_battery: Optional[int] = Query(None),
    min_camera_mp: Optional[int] = Query(None), min_screen_size: Optional[float] = Query(None),
    max_screen_size: Optional[float] = Query(None), min_year: Optional[int] = Query(None),
    max_weight: Optional[int] = Query(None), min_charging_w: Optional[int] = Query(None),
    tier: Optional[str] = Query(None), chipset_tier: Optional[str] = Query(None),
    features: Optional[str] = Query(None), region: Optional[str] = Query(None),
):
    slug = brand_slug_of(brand_slug)
    filters = FilterParams(
        brand=slug, min_price=min_price, max_price=max_price, min_ram=min_ram, min_battery=min_battery,
        min_camera_mp=min_camera_mp, min_screen_size=min_screen_size, max_screen_size=max_screen_size,
        min_year=min_year, max_weight=max_weight, min_charging_w=min_charging_w, tier=tier or chipset_tier,
        features=_csv(features), region=_region(region),
    )

    async def fetch():
        async with acquire_conn() as conn:
            total, results = await phone_repo.search(conn, filters=filters, sort_by=sort_by, sort_order=sort_order,
                                                     page=page, page_size=page_size)
        return {"brand_slug": slug, "total": total, "page": page, "page_size": page_size,
                "total_pages": max(1, -(-total // page_size)), "results": results}

    return await cached(f"brandphones:{slug}:{qkey(request)}", 300, fetch)
