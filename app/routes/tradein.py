from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..core.database import get_pool
from ..core.tradein import estimate_trade_in
from ..services import phone_repo
from ..services.recommend_service import TIER_BOUNDS

router = APIRouter(prefix="/tradein", tags=["tradein"])


def _tier_id_for_price(price: float) -> str:
    for tier_id, (lo, hi) in TIER_BOUNDS.items():
        if price >= lo and (hi is None or price <= hi):
            return tier_id
    return "d"


class TradeInRequest(BaseModel):
    phone_id: int
    screen_condition: str
    body_condition: str
    battery_health: int = Field(ge=1, le=10)
    battery_non_original: bool = False
    broken_components: list[str] = []


@router.post("/estimate")
async def estimate(payload: TradeInRequest):
    async with get_pool().acquire() as conn:
        phone = await phone_repo.get_by_id_or_slug(conn, str(payload.phone_id))
        if phone is None:
            raise HTTPException(status_code=404, detail=f"Phone {payload.phone_id} not found.")
        price = await phone_repo.latest_price_point(conn, phone["id"])
        phone_repo.apply_latest_price(phone, price)

    baseline_price = phone.get("price_usd")
    if not baseline_price:
        raise HTTPException(status_code=422, detail="No tracked price available for this phone.")

    tier_id = _tier_id_for_price(baseline_price)

    result = estimate_trade_in(
        baseline_price=baseline_price,
        brand=phone["brand"],
        model_name=phone["model_name"],
        price_tier_id=tier_id,
        screen_condition=payload.screen_condition,
        body_condition=payload.body_condition,
        battery_health=payload.battery_health,
        battery_non_original=payload.battery_non_original,
        broken_components=payload.broken_components,
    )

    return {
        "phone_id": phone["id"],
        "brand": phone["brand"],
        "model_name": phone["model_name"],
        "baseline_price": baseline_price,
        "price_tier_id": tier_id,
        **result,
    }
