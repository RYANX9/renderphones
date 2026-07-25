from __future__ import annotations
import re

# Single source of truth for chipset tier detection. Raw pattern strings
# (lowercase, no verbose mode) so the same source compiles as a Python
# regex here AND gets passed as-is into Postgres `~` in query.py — one
# definition, two consumers, impossible to drift apart.
TIER_REGEX_SOURCE: dict[str, str] = {
    "flagship": (
        r"snapdragon\s+8[+s]?\s*(elite|gen\s*[1-9])"
        r"|\bsm8[1-9]\d{2}\b"
        r"|dimensity\s+9\d{3}"
        r"|\bmt699\d\b"
        r"|exynos\s+2\d{3}"
        r"|apple\s+a(1[4-9]|[2-9]\d)(\s+(bionic|pro|max))?"
        r"|tensor\s+g[3-9]"
        r"|kirin\s+9\d{3}"
    ),
    "upper_mid": (
        r"snapdragon\s+7[+s]?\s*gen"
        r"|snapdragon\s+6\s+gen"
        r"|\bsm7[1-6]\d{2}\b"
        r"|dimensity\s+[78]\d{2,3}"
        r"|\bmt6[78]\d{2}\b"
        r"|exynos\s+1\d{3}"
        r"|kirin\s+8\d{2}"
        r"|tensor\s+g[12]"
        r"|apple\s+a1[0-3](\s+(bionic|pro))?"
    ),
}

_FLAGSHIP = re.compile(TIER_REGEX_SOURCE["flagship"], re.IGNORECASE)
_UPPER_MID = re.compile(TIER_REGEX_SOURCE["upper_mid"], re.IGNORECASE)

_TIER_LABELS = {
    "ultra_flagship": "Ultra Flagship",
    "flagship": "Flagship",
    "upper_mid_range": "Upper Mid-Range",
    "mid_range": "Mid-Range",
    "budget": "Budget",
    "upper_mid": "Upper Mid-Range",
    "entry": "Budget",
}


def chipset_tier_fallback(chipset: str | None) -> str:
    """Deterministic tier from chipset string only. Used when the AI
    smart-score tier is absent (unscored phone)."""
    if not chipset:
        return "unknown"
    if _FLAGSHIP.search(chipset):
        return "flagship"
    if _UPPER_MID.search(chipset):
        return "upper_mid"
    return "entry"


def resolve_tier(smart_tier: str | None, chipset: str | None) -> dict | None:
    raw = smart_tier or chipset_tier_fallback(chipset)
    if not raw or raw == "unknown":
        return None
    return {"id": raw, "label": _TIER_LABELS.get(raw, raw.replace("_", " ").title())}



def average_component_scores(p: dict, *, min_components: int = 1) -> float | None:
    """The one number: average of every AI sub-score this phone actually
    has, value_score included as a sixth dimension rather than a separate
    override. Every input is a fixed column on this phone's own
    phone_smart_scores row, so the result is identical on every page that
    renders this phone — homepage grid, brand page, detail page, compare.

    min_components guards against a phone with only one sub-score filled
    in producing a number that reads as "the average" but is really just
    that one field wearing a different label.
    """
    fields = (
        "smart_camera_score",
        "smart_performance_score",
        "smart_battery_score",
        "smart_display_score",
        "smart_build_score",
        "smart_value_score",
    )
    vals = [float(p[f]) for f in fields if p.get(f) is not None]
    if len(vals) < min_components:
        return None
    return round(sum(vals) / len(vals), 1)
    

# ─── spec composite (used when no price/peers exist, or as a value_score
#     denominator when smart_value_score is absent) ───────────────────────────

def spec_composite(p: dict) -> float:
    """0-10ish hardware-only composite, no price involved."""
    s = 0.0
    if p.get("antutu_score"):
        s += min(p["antutu_score"] / 2_000_000, 1.0) * 3.0
    if p.get("main_camera_mp"):
        s += min(p["main_camera_mp"] / 200, 1.0) * 2.0
    if p.get("battery_capacity"):
        s += min(p["battery_capacity"] / 7_000, 1.0) * 2.0
    if p.get("ram_options"):
        s += min(max(p["ram_options"]) / 16, 1.0) * 1.5
    if p.get("fast_charging_w"):
        s += min(p["fast_charging_w"] / 100, 1.0) * 1.0
    if p.get("screen_size"):
        s += min(float(p["screen_size"]) / 7.0, 1.0) * 0.5
    return s


def compute_value_score(phone: dict, peers: list[dict]) -> float | None:
    """Specs-per-dollar 0-10, normalised against a real peer group's best
    spec_composite. A peer group of size one always normalises to 10.0,
    so callers must supply a genuine comparison set."""
    if not phone.get("price_usd") or not peers:
        return None
    this = spec_composite(phone)
    peer_scores = [spec_composite(p) for p in peers if p.get("price_usd")]
    if not peer_scores:
        return None
    peak = max(peer_scores) or 1.0
    return round(min(this / peak * 10, 10.0), 1)


def normalize_popularity(raw, max_seen: float = 100.0) -> float | None:
    if raw is None:
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    return round(max(0.0, min(val, max_seen)), 1)


# ─── similarity scoring for /similar (multi-factor, not just price+brand) ────

# Weights for the "similar phones" ranking. Tuned so price proximity
# dominates (buyers compare within budget first) but hardware/brand
# affinity meaningfully separates otherwise-tied candidates.
_SIMILARITY_WEIGHTS = {
    "price": 0.40,
    "brand": 0.15,
    "tier": 0.15,
    "camera": 0.15,
    "battery": 0.10,
    "screen": 0.05,
}


def similarity_score(base: dict, candidate: dict) -> float:
    """0-1 similarity score between the anchor phone and a candidate,
    used to rank /phones/{id}/similar beyond a crude price-band filter."""
    score = 0.0

    base_price = base.get("price_usd")
    cand_price = candidate.get("price_usd")
    if base_price and cand_price:
        diff_ratio = abs(base_price - cand_price) / base_price
        score += _SIMILARITY_WEIGHTS["price"] * max(0.0, 1.0 - min(diff_ratio / 0.5, 1.0))

    if base.get("brand") and base["brand"] == candidate.get("brand"):
        score += _SIMILARITY_WEIGHTS["brand"]

    base_tier = resolve_tier(base.get("smart_tier"), base.get("chipset"))
    cand_tier = resolve_tier(candidate.get("smart_tier"), candidate.get("chipset"))
    if base_tier and cand_tier and base_tier["id"] == cand_tier["id"]:
        score += _SIMILARITY_WEIGHTS["tier"]

    for field, key, scale in (
        ("camera", "main_camera_mp", 108),
        ("battery", "battery_capacity", 2000),
        ("screen", "screen_size", 1.0),
    ):
        bv, cv = base.get(key), candidate.get(key)
        if bv and cv:
            closeness = max(0.0, 1.0 - min(abs(float(bv) - float(cv)) / scale, 1.0))
            score += _SIMILARITY_WEIGHTS[field] * closeness

    return round(score, 4)


# ─── /recommend priority scoring, expressed as SQL for DB-side ranking ───────
# All expressions here reference real phone_specs columns that exist in the
# current schema (no more guessed columns like p.has_ois on the phones table
# — everything below is s.<col> against phone_specs, or sc.<col> against
# phone_smart_scores).

PRIORITY_SQL_EXPR: dict[str, str] = {
    "camera": (
        "COALESCE(sc.camera_score, "
        "LEAST(COALESCE(s.main_camera_mp, 0) / 20.0, 10) "
        "+ CASE WHEN s.has_ois THEN 1.0 ELSE 0 END "
        "+ CASE WHEN s.optical_zoom IS NOT NULL THEN 1.0 ELSE 0 END)"
    ),
    "battery": "COALESCE(sc.battery_score, LEAST(COALESCE(s.battery_capacity, 0) / 600.0, 10))",
    "performance": "COALESCE(sc.performance_score, LEAST(COALESCE(s.antutu_score, 0) / 150000.0, 10))",
    "gaming": (
        "CASE WHEN s.is_premium_gaming THEN 10 "
        "ELSE COALESCE(sc.performance_score, LEAST(COALESCE(s.antutu_score, 0) / 150000.0, 10)) END"
    ),
    "compact": "GREATEST(0, 10 - (COALESCE(s.screen_size, 6.5) - 5.5) * 4)",
    "lightweight": "GREATEST(0, 10 - (COALESCE(s.weight_g, 200) - 140) / 8.0)",
    "display": "COALESCE(sc.display_score, LEAST(COALESCE(s.refresh_rate_hz, 60) / 12.0, 10))",
    "smooth_display": "LEAST(COALESCE(s.refresh_rate_hz, 60) / 12.0, 10)",
    "fast_charging": "LEAST(COALESCE(s.fast_charging_w, 0) / 10.0, 10)",
    "wireless_charging": (
        "CASE WHEN s.has_wireless_charging "
        "THEN LEAST(COALESCE(s.wireless_charging_w, 10) / 15.0, 10) ELSE 0 END"
    ),
    "durability": (
        "CASE "
        "WHEN s.water_resistance ILIKE '%IP68%' THEN 10 "
        "WHEN s.water_resistance ILIKE '%IP6%' OR s.water_resistance ILIKE '%IP5%' THEN 7 "
        "ELSE 3 END"
    ),
    "value": "COALESCE(sc.value_score, 6.0)",
}

# Priorities that gate results (WHERE clause) rather than blend additively.
# A binary trait shouldn't average against continuous scores — a non-
# foldable phone that scores well elsewhere must never outrank a foldable
# one when "foldable" was explicitly requested.
HARD_FILTER_PRIORITIES: dict[str, str] = {
    "foldable": "s.is_foldable IS TRUE",
    "headphone_jack": "s.has_headphone_jack IS TRUE",
    "nfc": "s.has_nfc IS TRUE",
}

PRIORITY_LABELS: dict[str, str] = {
    "camera": "Camera Quality",
    "battery": "Battery Life",
    "performance": "Performance",
    "gaming": "Gaming Performance",
    "compact": "Compact Size",
    "lightweight": "Lightweight",
    "display": "Display Quality",
    "smooth_display": "High Refresh Rate",
    "fast_charging": "Fast Charging",
    "wireless_charging": "Wireless Charging",
    "foldable": "Foldable Design",
    "durability": "Water/Dust Resistance",
    "value": "Best Value",
    "headphone_jack": "Headphone Jack",
    "nfc": "NFC",
}
