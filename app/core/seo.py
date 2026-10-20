"""Server-side SEO decisions, so the frontend never has to guess them.

- indexable flags (what Google may index)
- AggregateOffer built ONLY from real retailer offers (never from placeholder
  or catalog prices, which would be inaccurate markup)
"""
from __future__ import annotations

from typing import Any

from .config import settings


def phone_seo(phone: dict[str, Any], offers: list[dict[str, Any]]) -> dict[str, Any]:
    live = [
        o for o in offers
        if o.get("usd") and o.get("is_available", True) and not o.get("is_placeholder")
        and o.get("price_source") == "retailer"
    ]
    aggregate = None
    if live:
        prices = [float(o["usd"]) for o in live]
        aggregate = {
            "@type": "AggregateOffer", "priceCurrency": "USD",
            "lowPrice": min(prices), "highPrice": max(prices), "offerCount": len(live),
        }
    return {
        "indexable": True,
        "aggregate_offer": aggregate,
        # The frontend must not emit Review / AggregateRating from our own scores.
        "allow_rating_markup": False,
    }


def category_indexable(ranked_count: int) -> bool:
    return ranked_count >= settings.category_min_phones_indexable


def brand_indexable(phone_count: int) -> bool:
    return phone_count >= 3
