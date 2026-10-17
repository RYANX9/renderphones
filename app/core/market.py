"""
market.py — region-aware offer filtering and ordering.

Every retailer belongs to one market scope (retailers.scope in the database).
Scope membership is loaded from the `retailers` table at startup; this file
holds only the scope RULES (which regions may see which scope), not retailer
names. An unknown retailer fails open to 'global' so a missing row never hides
a phone from everyone.
"""
from __future__ import annotations

from typing import Any

NORTH_AMERICA: frozenset[str] = frozenset({"US", "CA"})
EURO_ZONE: frozenset[str] = frozenset({
    "GB", "DE", "FR", "IT", "ES", "NL", "SE", "NO",
    "DK", "FI", "BE", "AT", "CH", "PT", "IE", "PL",
})
OCEANIA: frozenset[str] = frozenset({"AU", "NZ"})
CHINA_MARKET_REGIONS: frozenset[str] = frozenset({
    "CN", "HK", "MO", "TW", "JP", "KR", "TH", "VN", "MY", "SG", "ID", "PH",
})
INDIA_PAKISTAN_REGIONS: frozenset[str] = frozenset({"IN", "PK"})
CHINA_VERSION_EXCLUDED: frozenset[str] = NORTH_AMERICA | EURO_ZONE | OCEANIA

# Zones used for display priority only, not eligibility.
REGION_ZONES: dict[str, str] = {r: "euro" for r in EURO_ZONE}
REGION_ZONES.update({r: "north_america" for r in NORTH_AMERICA})
REGION_ZONES.update({r: "oceania" for r in OCEANIA})
REGION_ZONES.update({r: "east_asia" for r in ("JP", "KR")})

SCOPE_ALLOWED: dict[str, frozenset[str]] = {
    "china_market": CHINA_MARKET_REGIONS,
    "india_pakistan": INDIA_PAKISTAN_REGIONS,
}
SCOPE_EXCLUDED: dict[str, frozenset[str]] = {
    "europe": NORTH_AMERICA,
    "china_version": CHINA_VERSION_EXCLUDED,
}
ALL_SCOPES: tuple[str, ...] = ("global", "europe", "china_market", "china_version", "india_pakistan")


def region_allowed_for_scope(scope: str, region: str) -> bool:
    region = region.upper()
    if scope in SCOPE_ALLOWED:
        return region in SCOPE_ALLOWED[scope]
    if scope in SCOPE_EXCLUDED:
        return region not in SCOPE_EXCLUDED[scope]
    return True


def allowed_scopes(region: str | None) -> list[str] | None:
    if not region:
        return None
    return [s for s in ALL_SCOPES if region_allowed_for_scope(s, region)]


# retailer_key -> {"scope": str, "region": str|None, "name": str, "domain": str|None}
RETAILERS: dict[str, dict[str, Any]] = {}


async def load_retailers(conn) -> int:
    rows = await conn.fetch("SELECT key, name, scope, region, domain FROM retailers")
    if rows:
        RETAILERS.clear()
        RETAILERS.update({
            r["key"].strip().lower(): {
                "scope": r["scope"], "region": r["region"], "name": r["name"], "domain": r["domain"],
            }
            for r in rows
        })
    return len(rows)


def retailer_scope(retailer_key: str) -> str:
    info = RETAILERS.get((retailer_key or "").strip().lower())
    return info["scope"] if info else "global"


def retailer_info(retailer_key: str) -> dict[str, Any]:
    return RETAILERS.get((retailer_key or "").strip().lower(), {})


def phone_allowed_for_region(offers: list[dict[str, Any]], user_region: str | None) -> bool:
    """A phone with no offers is visible everywhere; otherwise at least one
    offer's retailer scope must permit the region."""
    if not offers or not user_region:
        return True
    region = user_region.upper()
    return any(region_allowed_for_scope(retailer_scope(o.get("retailer_key", "")), region) for o in offers)


def _offer_priority(offer: dict[str, Any], user_region: str) -> tuple[int, float]:
    """0 exact region, 1 same zone, 2 global, 3 europe, 4 china_version, 9 other."""
    scope = retailer_scope(offer.get("retailer_key", ""))
    offer_region = (offer.get("region") or "").upper()
    usd = float(offer.get("usd") or 0)

    if offer_region == user_region.upper():
        return (0, usd)
    uz, oz = REGION_ZONES.get(user_region.upper()), REGION_ZONES.get(offer_region)
    if uz and oz and uz == oz:
        return (1, usd)
    return ({"global": 2, "europe": 3, "china_version": 4}.get(scope, 9), usd)


def resolve_offers_for_region(
    offers: list[dict[str, Any]],
    user_region: str | None,
    available_only: bool = True,
    include_ineligible: bool = False,
) -> dict[str, Any]:
    """Order (and optionally filter) offers for a viewer's region.

    include_ineligible=False (default): offers whose retailer scope does not
    permit the region are dropped. Used by listing/compare code.
    include_ineligible=True: nothing is dropped. Ineligible offers are sorted
    AFTER every eligible one, so offers[0] is always a region-appropriate offer
    and the rest can live in a "more places to buy" dropdown.
    """
    rows = offers or []
    if available_only:
        rows = [r for r in rows if r.get("is_available", True)]

    regions_available = sorted({r["region"].upper() for r in rows if r.get("region")})

    if not user_region or not rows:
        rows = sorted(rows, key=lambda o: float(o.get("usd") or 1e12))
        return {"offers": rows, "regions_available": regions_available, "is_region_exclusive": bool(offers) and not rows}

    ur = user_region.upper()
    eligible = [r for r in rows if region_allowed_for_scope(retailer_scope(r.get("retailer_key", "")), ur)]
    if include_ineligible:
        eligible_ids = {id(r) for r in eligible}
        ordered = sorted(
            rows,
            key=lambda o: (0 if id(o) in eligible_ids else 1, *_offer_priority(o, ur)),
        )
    else:
        ordered = sorted(eligible, key=lambda o: _offer_priority(o, ur))
    return {
        "offers": ordered,
        "regions_available": regions_available,
        "is_region_exclusive": bool(offers) and not eligible,
    }


def cap_per_brand(phones: list[dict[str, Any]], max_per_brand: int) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    result: list[dict[str, Any]] = []
    for p in phones:
        brand = (p.get("brand_slug") or p.get("brand") or "").strip().lower()
        counts[brand] = counts.get(brand, 0) + 1
        if counts[brand] <= max_per_brand:
            result.append(p)
    return result