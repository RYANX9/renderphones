from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel


class PhoneBase(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    slug: Optional[str] = None
    model_name: str
    brand: str
    price_usd: Optional[float] = None
    main_image_url: Optional[str] = None
    screen_size: Optional[float] = None
    battery_capacity: Optional[int] = None
    ram_options: Optional[list[int]] = None
    storage_options: Optional[list[int]] = None
    main_camera_mp: Optional[int] = None
    chipset: Optional[str] = None
    antutu_score: Optional[int] = None
    amazon_link: Optional[str] = None
    release_year: Optional[int] = None
    release_month: Optional[int] = None
    release_day: Optional[int] = None
    release_ts: Optional[int] = None
    value_score: Optional[float] = None
    chipset_tier: Optional[str] = None


class SmartScore(BaseModel):
    overall_score: Optional[float] = None
    camera_score: Optional[float] = None
    performance_score: Optional[float] = None
    battery_score: Optional[float] = None
    display_score: Optional[float] = None
    build_score: Optional[float] = None
    value_score: Optional[float] = None
    strengths: Optional[list[str]] = None
    weaknesses: Optional[list[str]] = None
    reasoning: Optional[str] = None
    tier: Optional[str] = None
    model_version: Optional[str] = None
    scored_at: Optional[str] = None


class PhoneDetail(PhoneBase):
    weight_g: Optional[float] = None
    thickness_mm: Optional[float] = None
    screen_resolution: Optional[str] = None
    fast_charging_w: Optional[int] = None
    full_specifications: Optional[dict[str, Any]] = None
    features: Optional[list[str]] = None
    price_updated_at: Optional[str] = None
    price_scope: Optional[str] = None
    smart_score: Optional[SmartScore] = None


class PhoneSearchResponse(BaseModel):
    total: int
    page: int
    page_size: int
    results: list[PhoneBase]


class TrendingResponse(BaseModel):
    phones: list[PhoneBase]


class CompareResponse(BaseModel):
    phones: list[PhoneDetail]


class RecommendResponse(BaseModel):
    phones: list[PhoneBase]
    priorities: list[str]


class CategoryPhone(PhoneBase):
    category_score: float


class CategoryResponse(BaseModel):
    slug: str
    title: str
    description: str
    phones: list[CategoryPhone]


class CategoryMeta(BaseModel):
    slug: str
    title: str
    description: str


class CategoriesResponse(BaseModel):
    categories: list[CategoryMeta]


class FilterStats(BaseModel):
    total_phones: int
    total_brands: int
    price_range: dict[str, float]
    battery_range: dict[str, int]
    screen_range: dict[str, float]
    weight_range: dict[str, int]
    charging_range: dict[str, int]
    year_range: dict[str, int]
    brands: list[dict[str, Any]]
    ram_options: list[int]
    release_years: list[int]
