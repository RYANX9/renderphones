from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel


class ChipsetTier(BaseModel):
    id: str
    label: str


class SmartScore(BaseModel):
    overall_score: Optional[float] = None
    camera_score: Optional[float] = None
    performance_score: Optional[float] = None
    battery_score: Optional[float] = None
    display_score: Optional[float] = None
    build_score: Optional[float] = None
    value_score: Optional[float] = None
    tier: Optional[str] = None
    reasoning: Optional[str] = None
    strengths: Optional[list[str]] = None
    weaknesses: Optional[list[str]] = None
    model_version: Optional[str] = None
    scored_at: Optional[str] = None


class PhoneListItem(BaseModel):
    model_config = {"extra": "allow"}

    id: int
    model_name: str
    brand: str
    slug: str
    main_image_url: Optional[str] = None
    release_year: Optional[int] = None
    price_usd: Optional[float] = None
    availability_status: Optional[str] = None
    screen_size: Optional[float] = None
    battery_capacity: Optional[int] = None
    fast_charging_w: Optional[int] = None
    main_camera_mp: Optional[int] = None
    chipset: Optional[str] = None
    antutu_score: Optional[int] = None
    ram_options: Optional[list[int]] = None
    storage_options: Optional[list[int]] = None
    chipset_tier: Optional[ChipsetTier] = None
    value_score: Optional[float] = None
    popularity: Optional[float] = None
    smart_score: Optional[SmartScore] = None


class PhoneVariant(BaseModel):
    id: int
    ram_gb: Optional[int] = None
    storage_gb: Optional[int] = None
    price: Optional[float] = None
    url: Optional[str] = None


class PhoneImage(BaseModel):
    id: int
    image_url: str
    sort_order: Optional[int] = None


class PhoneDetail(PhoneListItem):
    weight_g: Optional[float] = None
    thickness_mm: Optional[float] = None
    screen_resolution: Optional[str] = None
    full_specifications: Optional[dict[str, Any]] = None
    variants: list[PhoneVariant] = []
    images: list[PhoneImage] = []
    features: list[str] = []


class SearchResponse(BaseModel):
    total: int
    page: int
    page_size: int
    results: list[dict]


class CompareVerdict(BaseModel):
    verdict: str
    picks: list[dict]


class CompareResponse(BaseModel):
    phones: list[dict]
    verdict: Optional[CompareVerdict] = None


class RecommendResponse(BaseModel):
    phones: list[dict]
    priorities: list[str]
    hard_filters: list[str]
    requested_price_range: dict
    effective_price_range: dict
    budget_widened: bool
    insufficient_matches: bool
