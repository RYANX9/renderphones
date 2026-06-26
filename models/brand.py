from __future__ import annotations

from typing import Optional
from pydantic import BaseModel

from .phone import PhoneBase


class PriceRange(BaseModel):
    min: Optional[float] = None
    max: Optional[float] = None
    avg: Optional[float] = None


class BrandStats(BaseModel):
    brand: str
    total_phones: int
    price_range: PriceRange
    avg_battery: Optional[int] = None
    latest_year: Optional[int] = None
    latest_phone: Optional[PhoneBase] = None


class BrandListItem(BaseModel):
    brand: str
    count: int


class BrandsResponse(BaseModel):
    brands: list[BrandListItem]


class BrandPhonesResponse(BaseModel):
    total: int
    page: int
    page_size: int
    results: list[PhoneBase]
