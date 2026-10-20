from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..core.database import acquire_conn
from ..core.category_config import CATEGORY_CONFIG, price_band_slug
from ..core.slugs import category_path
from ..core.tradein import estimate_trade_in
from ..services import phone_repo
from ..services.recommend_service import TIER_BOUNDS

router = APIRouter(prefix="/tradein", tags=["tradein"])


def _tier_id_for_price(price: float) -> str:
    for tier_id, (lo, hi) in TIER_BOUNDS.items():
        if price >= lo and (hi is None or price <= hi):
            return tier_id
    return "d"


def _budget_category_for(price: float) -> str:
    return price_band_slug(price) or "under-700"


class TradeInRequest(BaseModel):
    phone_id: int
    variant_id: Optional[int] = None
    screen_condition: str
    body_condition: str
    battery_health: int = Field(ge=1, le=10)
    battery_non_original: bool = False
    screen_non_original: bool = False
    broken_components: list[str] = []


@router.post("/estimate")
async def estimate(payload: TradeInRequest):
    async with acquire_conn() as conn:
        phone = await phone_repo.tradein_phone(conn, payload.phone_id)
        variants = await phone_repo.fetch_variants(conn, payload.phone_id) if payload.variant_id else []
    if phone is None:
        raise HTTPException(status_code=404, detail=f"Phone {payload.phone_id} not found.")

    # Same price rule as everywhere else: the chosen variant's lowest price, else the phone price.
    chosen = next((v for v in variants if v["id"] == payload.variant_id and v.get("price_usd")), None)
    baseline_price = chosen["price_usd"] if chosen else phone["price_usd"]
    if not baseline_price:
        raise HTTPException(status_code=422, detail="No tracked price available for this phone.")

    tier_id = _tier_id_for_price(baseline_price)
    result = estimate_trade_in(
        baseline_price=baseline_price,
        brand=phone["brand"],
        model_name=phone["name"],
        price_tier_id=tier_id,
        screen_condition=payload.screen_condition,
        body_condition=payload.body_condition,
        battery_health=payload.battery_health,
        battery_non_original=payload.battery_non_original,
        broken_components=payload.broken_components,
        screen_non_original=payload.screen_non_original,
    )
    return {
        "phone_id": phone["id"], "brand": phone["brand"], "name": phone["name"], "display_name": phone["display_name"],
        "baseline_price": baseline_price,
        "baseline_variant": ({"id": chosen["id"], "ram_gb": chosen["ram_gb"], "storage_gb": chosen["storage_gb"]} if chosen else None),
        "price_basis": phone["price_basis"], "price_tier_id": tier_id, **result,
        "next_steps": [
            {"slot": "tradein.pick", "kind": "pick", "label": "Find your next phone", "href": f"/pick?budget={int(baseline_price)}"},
            {"slot": "tradein.best", "kind": "category", "label": CATEGORY_CONFIG[_budget_category_for(baseline_price)]["title"],
             "href": category_path(_budget_category_for(baseline_price))},
            {"slot": "tradein.phone", "kind": "phone", "label": f"View the {phone['display_name']}", "href": f"/brand/{phone['brand_slug']}/{phone['slug']}"},
            {"slot": "tradein.hub", "kind": "hub", "label": "All best-of lists", "href": "/best"},
        ],
    }
