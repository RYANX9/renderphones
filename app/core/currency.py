"""Display-currency helpers.

Platform rule: every price is USD everywhere. The ONLY exception is a visitor
who is in the same country as the retailer behind the price: that visitor sees
the local currency. The API therefore always returns `usd` and, when the offer
has a region whose currency is not USD, a `local` block the frontend swaps in
if (and only if) viewer_region == offer.region.

fx_rates.usd_per_unit means "1 unit of currency = N USD", so
local_amount = usd / usd_per_unit.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ISO-3166 alpha-2 -> ISO-4217. USD regions are intentionally absent: they
# never get a `local` block.
REGION_CURRENCY: dict[str, str] = {
    "IN": "INR", "PK": "PKR", "BD": "BDT", "LK": "LKR", "NP": "NPR",
    "GB": "GBP",
    "DE": "EUR", "FR": "EUR", "IT": "EUR", "ES": "EUR", "NL": "EUR", "BE": "EUR",
    "AT": "EUR", "PT": "EUR", "IE": "EUR", "FI": "EUR", "GR": "EUR",
    "SE": "SEK", "NO": "NOK", "DK": "DKK", "CH": "CHF", "PL": "PLN", "CZ": "CZK",
    "TR": "TRY", "RU": "RUB", "UA": "UAH",
    "CA": "CAD", "MX": "MXN", "BR": "BRL", "AR": "ARS", "CL": "CLP", "CO": "COP",
    "AU": "AUD", "NZ": "NZD",
    "JP": "JPY", "KR": "KRW", "CN": "CNY", "HK": "HKD", "TW": "TWD",
    "SG": "SGD", "MY": "MYR", "TH": "THB", "VN": "VND", "ID": "IDR", "PH": "PHP",
    "AE": "AED", "SA": "SAR", "EG": "EGP", "NG": "NGN", "ZA": "ZAR", "KE": "KES",
    "DZ": "DZD", "MA": "MAD",
}

_WHOLE_UNIT = frozenset({"JPY", "KRW", "VND", "IDR", "CLP", "COP", "PKR", "INR", "NGN", "KES", "HUF"})

_RATES: dict[str, float] = {"USD": 1.0}


async def load_fx(conn) -> int:
    """Refresh the in-memory rate table from fx_rates. Keeps the old table if the
    query fails or returns nothing, so a bad refresh never drops local prices."""
    rows = await conn.fetch("SELECT currency, usd_per_unit FROM fx_rates")
    fresh = {r["currency"].upper(): float(r["usd_per_unit"]) for r in rows if r["usd_per_unit"]}
    if fresh:
        fresh.setdefault("USD", 1.0)
        _RATES.clear()
        _RATES.update(fresh)
    return len(fresh)


def currency_for_region(region: str | None) -> str | None:
    if not region:
        return None
    return REGION_CURRENCY.get(region.strip().upper())


def to_local(usd: Any, region: str | None) -> dict[str, Any] | None:
    """{'currency','amount','rate'} for a USD price in the region's currency, or
    None when the region is USD/unknown or there is no rate for it."""
    if usd is None:
        return None
    currency = currency_for_region(region)
    if not currency or currency == "USD":
        return None
    rate = _RATES.get(currency)
    if not rate:
        return None
    amount = float(usd) / rate
    digits = 0 if (currency in _WHOLE_UNIT or amount >= 1000) else 2
    return {"currency": currency, "amount": round(amount, digits), "rate": rate}
