"""
market.py — Single source of truth for retailer market-scope classification
and region-aware filtering.

Model: every retailer belongs to exactly one market scope. Each scope has
its own region rule — an explicit allow-list (small, targeted scopes) or
an explicit exclude-list (broad scopes with a few carve-outs). No other
file should hardcode retailer names, region codes, or scope membership.
"""

from __future__ import annotations
import re
from typing import Any

# ---------------------------------------------------------------------------
# Region groupings (building blocks, not scopes themselves)
# ---------------------------------------------------------------------------

NORTH_AMERICA: frozenset[str] = frozenset({"US", "CA"})

EURO_ZONE: frozenset[str] = frozenset({
    "GB", "DE", "FR", "IT", "ES", "NL", "SE", "NO",
    "DK", "FI", "BE", "AT", "CH", "PT", "IE", "PL",
})

OCEANIA: frozenset[str] = frozenset({"AU", "NZ"})

# East + Southeast Asia — the china_market allow-list. Includes CN itself
# (domestic retailers are obviously valid for users physically in China).
CHINA_MARKET_REGIONS: frozenset[str] = frozenset({
    "CN", "HK", "MO", "TW",             # Greater China
    "JP", "KR",                          # East Asia
    "TH", "VN", "MY", "SG", "ID", "PH",  # Southeast Asia
})

INDIA_PAKISTAN_REGIONS: frozenset[str] = frozenset({"IN", "PK"})

# china_version (cross-border resellers) is blocked in NA + Europe + Oceania,
# open everywhere else — including China itself, Africa, LatAm, rest of Asia.
CHINA_VERSION_EXCLUDED: frozenset[str] = NORTH_AMERICA | EURO_ZONE | OCEANIA

# Zones used only for offer *display priority*, not eligibility.
REGION_ZONES: dict[str, str] = {r: "euro" for r in EURO_ZONE}
REGION_ZONES.update({r: "north_america" for r in NORTH_AMERICA})
REGION_ZONES.update({r: "oceania" for r in OCEANIA})
REGION_ZONES.update({r: "east_asia" for r in ("JP", "KR")})

# ---------------------------------------------------------------------------
# Market scopes
# ---------------------------------------------------------------------------
# "global"        — no restriction, visible everywhere.
# "europe"        — visible everywhere except North America.
# "china_market"  — visible only within CHINA_MARKET_REGIONS.
# "china_version" — visible everywhere except CHINA_VERSION_EXCLUDED.
# "india_pakistan"— visible only within INDIA_PAKISTAN_REGIONS.

SCOPE_ALLOWED: dict[str, frozenset[str]] = {
    "china_market": CHINA_MARKET_REGIONS,
    "india_pakistan": INDIA_PAKISTAN_REGIONS,
}

SCOPE_EXCLUDED: dict[str, frozenset[str]] = {
    "europe": NORTH_AMERICA,
    "china_version": CHINA_VERSION_EXCLUDED,
}
# "global" appears in neither dict — treated as fully open.


def region_allowed_for_scope(scope: str, region: str) -> bool:
    region = region.upper()
    if scope in SCOPE_ALLOWED:
        return region in SCOPE_ALLOWED[scope]
    if scope in SCOPE_EXCLUDED:
        return region not in SCOPE_EXCLUDED[scope]
    return True  # global, or an unrecognized scope — fail open, not closed


# ---------------------------------------------------------------------------
# Retailer -> scope classification
# ---------------------------------------------------------------------------

RETAILER_SCOPE: dict[str, str] = {
    # global
    "amazon": "global",
    "amazon_us": "global",
    "amazon_global": "global",
    "bestbuy": "global",
    "walmart": "global",
    "bhphotovideo": "global",
    "newegg": "global",
    "samsung_us": "global",
    "apple_us": "global",
    "google_store": "global",

    # europe
    "amazon_uk": "europe",
    "amazon_de": "europe",
    "amazon_fr": "europe",
    "amazon_it": "europe",
    "amazon_es": "europe",
    "samsung_de": "europe",
    "samsung_uk": "europe",
    "mediamarkt": "europe",
    "currys": "europe",
    "fnac": "europe",

    # china_market (domestic-only, will not ship outside CHINA_MARKET_REGIONS)
    "jd": "china_market",
    "jd.com": "china_market",
    "tmall": "china_market",
    "taobao": "china_market",
    "suning": "china_market",
    "pinduoduo": "china_market",
    "mi_cn": "china_market",
    "honor_cn": "china_market",
    "oppo_cn": "china_market",
    "vivo_cn": "china_market",

    # china_version (cross-border resellers — your AliExpress default lives here)
    "aliexpress": "china_version",
    "aliexpress_global": "china_version",
    "ebay_global": "china_version",
    "gearbest": "china_version",
    "banggood": "china_version",
    "hekka": "china_version",
    "tomtop": "china_version",

    # india_pakistan
    "amazon_in": "india_pakistan",
    "flipkart": "india_pakistan",
    "daraz_pk": "india_pakistan",
    "mi_in": "india_pakistan",
    "samsung_in": "india_pakistan",
}


def _normalize_retailer(retailer: str) -> str:
    return retailer.strip().lower()


def retailer_scope(retailer: str) -> str:
    """Unknown retailers default to 'global' — fail open so an unclassified
    retailer never accidentally hides a phone from every region."""
    return RETAILER_SCOPE.get(_normalize_retailer(retailer), "global")


# ---------------------------------------------------------------------------
# Phone-level eligibility
# ---------------------------------------------------------------------------

def phone_allowed_for_region(
    retailer_links_rows: list[dict[str, Any]],
    user_region: str | None,
) -> bool:
    """
    A phone is visible if it has no retailer_links rows at all (unscoped,
    legacy global fallback), or if at least one row's retailer scope
    permits the user's region. Blocking is a whole-phone decision made
    from the union of all its offers, not a single-row check.
    """
    if not retailer_links_rows or not user_region:
        return True

    region = user_region.upper()
    for row in retailer_links_rows:
        scope = retailer_scope(row.get("retailer", ""))
        if region_allowed_for_scope(scope, region):
            return True
    return False


def disallowed_retailers_for_region(region: str | None) -> list[str]:
    """
    Every retailer key whose scope excludes `region`. Used to build the
    SQL EXISTS/NOT-EXISTS clause in query.py — a phone survives the SQL
    filter if it has no retailer_links rows, or if at least one row's
    retailer is NOT in this returned list.
    """
    if not region:
        return []
    region = region.upper()
    return sorted(
        retailer for retailer, scope in RETAILER_SCOPE.items()
        if not region_allowed_for_scope(scope, region)
    )


# ---------------------------------------------------------------------------
# Offer resolution and sorting (per-phone, post-fetch — unchanged shape,
# scope-aware tiering)
# ---------------------------------------------------------------------------

def _offer_priority(offer: dict[str, Any], user_region: str) -> tuple[int, float]:
    """
    Tier 0 — exact region match
    Tier 1 — same display zone
    Tier 2 — global scope retailer
    Tier 3 — europe scope retailer (visible outside NA, ranks below global)
    Tier 4 — china_version cross-border reseller
    Tier 9 — everything else (china_market/india_pakistan shown out-of-scope
             only reachable if it's the single offer a phone has)
    """
    retailer = _normalize_retailer(offer.get("retailer", ""))
    scope = retailer_scope(retailer)
    offer_region = (offer.get("region") or "").upper()
    price = float(offer.get("price") or 0)

    if offer_region == user_region.upper():
        return (0, price)

    user_zone = REGION_ZONES.get(user_region.upper())
    offer_zone = REGION_ZONES.get(offer_region)
    if user_zone and offer_zone and user_zone == offer_zone:
        return (1, price)

    if scope == "global":
        return (2, price)
    if scope == "europe":
        return (3, price)
    if scope == "china_version":
        return (4, price)

    return (9, price)


def resolve_offers_for_region(
    retailer_links_rows: list[dict[str, Any]],
    user_region: str | None,
    available_only: bool = True,
) -> dict[str, Any]:
    rows = retailer_links_rows or []
    if available_only:
        rows = [r for r in rows if r.get("is_available", True)]

    regions_available: list[str] = sorted(
        {r["region"].upper() for r in rows if r.get("region")}
    )

    if not user_region or not rows:
        return {
            "offers": rows,
            "regions_available": regions_available,
            "is_region_exclusive": bool(retailer_links_rows),
        }

    ur = user_region.upper()
    visible = [
        row for row in rows
        if region_allowed_for_scope(retailer_scope(row.get("retailer", "")), ur)
    ]
    visible.sort(key=lambda o: _offer_priority(o, ur))

    is_region_exclusive = bool(retailer_links_rows) and not visible

    return {
        "offers": visible,
        "regions_available": regions_available,
        "is_region_exclusive": is_region_exclusive,
    }


# ---------------------------------------------------------------------------
# Brand cap (unchanged from before)
# ---------------------------------------------------------------------------

def cap_per_brand(phones: list[dict[str, Any]], max_per_brand: int) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    result: list[dict[str, Any]] = []
    for p in phones:
        brand = (p.get("brand") or "").strip().lower()
        counts[brand] = counts.get(brand, 0) + 1
        if counts[brand] <= max_per_brand:
            result.append(p)
    return result


# ---------------------------------------------------------------------------
# Near-duplicate detection for /pick deduplication (unchanged from before)
# ---------------------------------------------------------------------------

_EDITION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\(\s*china\s*(edition|version|variant)?\s*\)", re.IGNORECASE),
    re.compile(r"\(\s*global\s*(edition|version|variant)?\s*\)", re.IGNORECASE),
    re.compile(r"\(\s*international\s*(edition|version|variant)?\s*\)", re.IGNORECASE),
    re.compile(r"\(\s*cn\s*\)", re.IGNORECASE),
    re.compile(r"\(\s*india\s*(edition|version|variant)?\s*\)", re.IGNORECASE),
    re.compile(r"\(\s*japan\s*(edition|version|variant)?\s*\)", re.IGNORECASE),
    re.compile(r"\bcn\s+edition\b", re.IGNORECASE),
    re.compile(r"\bglobal\s+version\b", re.IGNORECASE),
    re.compile(r"\bchina\s+version\b", re.IGNORECASE),
    re.compile(r"\bchina\s+edition\b", re.IGNORECASE),
]


def canonical_name(model_name: str) -> str:
    name = model_name.strip()
    for pattern in _EDITION_PATTERNS:
        name = pattern.sub("", name)
    return re.sub(r"\s{2,}", " ", name).strip().lower()


def deduplicate_candidates(
    phones: list[dict[str, Any]],
    user_region: str | None,
) -> list[dict[str, Any]]:
    seen: dict[tuple[str, str], int] = {}
    result: list[dict[str, Any]] = []

    for phone in phones:
        brand = (phone.get("brand") or "").strip().lower()
        canon = canonical_name(phone.get("model_name") or "")
        key = (brand, canon)

        if key not in seen:
            seen[key] = len(result)
            result.append(phone)
        else:
            incumbent_idx = seen[key]
            incumbent = result[incumbent_idx]

            newcomer_offers = resolve_offers_for_region(
                phone.get("_retailer_links", []), user_region
            )["offers"]
            incumbent_offers = resolve_offers_for_region(
                incumbent.get("_retailer_links", []), user_region
            )["offers"]

            if newcomer_offers and not incumbent_offers:
                result[incumbent_idx] = phone

    return result