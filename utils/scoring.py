from __future__ import annotations

import re

_FLAGSHIP = re.compile(
    r"""
    # Snapdragon 8 / 8+ / 8s — Elite or Gen N
    snapdragon \s+ 8 [+s]? \s* (?: elite | gen \s* \d+ )
    # Qualcomm model numbers: SM8150 (SD855) → SM8750 (SD8 Elite)
    | \b sm8[1-9]\d{2} \b
    # MediaTek Dimensity 9xxx
    | dimensity \s+ 9\d{3}
    # MediaTek Dimensity 9xxx model numbers (MT6989=9300, MT6991=9400)
    | \b mt699\d \b
    # Samsung Exynos 2xxx
    | exynos \s+ 2\d{3}
    # Apple A14 and above (bionic / pro / max suffix optional)
    | apple \s+ a (?: 1[4-9] | [2-9]\d ) (?: \s+ (?: bionic | pro | max ) )?
    # Google Tensor G3+
    | tensor \s+ g [3-9]
    # Huawei / HiSilicon Kirin 9xxx
    | kirin \s+ 9\d{3}
    """,
    re.VERBOSE | re.IGNORECASE,
)

_MID = re.compile(
    r"""
    # Snapdragon 7 / 7+ / 7s Gen N (requires "Gen" — excludes 730, 720, etc.)
    snapdragon \s+ 7 [+s]? \s* gen
    # Snapdragon 6 Gen N only (excludes 695, 690, 680 — those are entry)
    | snapdragon \s+ 6 \s+ gen
    # Qualcomm SM7xxx model numbers
    | \b sm7[1-6]\d{2} \b
    # MediaTek Dimensity 7xxx / 8xxx
    | dimensity \s+ [78]\d{2,3}
    # MediaTek 7/8xx model numbers (MT67xx, MT68xx)
    | \b mt6[78]\d{2} \b
    # Samsung Exynos 1xxx
    | exynos \s+ 1\d{3}
    # Huawei Kirin 8xx
    | kirin \s+ 8\d{2}
    # Google Tensor G1 / G2
    | tensor \s+ g [12]
    # Apple A10–A13
    | apple \s+ a1[0-3] (?: \s+ (?: bionic | pro ) )?
    """,
    re.VERBOSE | re.IGNORECASE,
)


def chipset_tier(chipset: str | None) -> str:
    if not chipset:
        return "unknown"
    if _FLAGSHIP.search(chipset):
        return "flagship"
    if _MID.search(chipset):
        return "mid"
    return "entry"


def spec_score(p: dict) -> float:
    """Raw 0-10ish composite from hardware specs only, no price involved.
    Used as the value-score fallback when a phone has no price_usd or no
    priced peers, and as the base for spec-driven priority matching."""
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
    """Specs-per-dollar, 0-10, normalised against the peer group's best
    spec_score. Returns None if the phone or every peer lacks a price."""
    if not phone.get("price_usd") or not peers:
        return None

    this = spec_score(phone)
    peer_scores = [spec_score(p) for p in peers if p.get("price_usd")]
    if not peer_scores:
        return None
    peak = max(peer_scores) or 1.0
    return round(min(this / peak * 10, 10.0), 1)


def normalize_popularity(raw: float | int | None, max_seen: float = 100.0) -> float | None:
    """`phones.popularity` is stored on whatever scale the scraper produced.
    Clamp to 0-100 so the API always returns a comparable value."""
    if raw is None:
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    return round(max(0.0, min(val, max_seen)), 1)


def attach_computed_fields(phones: list[dict]) -> list[dict]:
    """Mutates and returns `phones` in place: fills chipset_tier for every
    row, and value_score using (in priority order) the AI smart-score
    value_score, then a specs-per-dollar composite scored against the
    other phones in this same result set as peers.

    Expects each dict to optionally carry `smart_value_score` (popped here)
    from a `phone_smart_scores` join — callers building the SELECT should
    alias it exactly that way.
    """
    peers = phones
    for p in phones:
        p["chipset_tier"] = chipset_tier(p.get("chipset"))
        p["popularity"] = normalize_popularity(p.get("popularity"))

        smart_value = p.pop("smart_value_score", None)
        if smart_value is not None:
            p["value_score"] = round(float(smart_value), 1)
        elif p.get("value_score") is None:
            p["value_score"] = compute_value_score(p, peers)
    return phones


# Expressions used by /phones/recommend to rank phones per selected
# priority. Each yields roughly a 0-10 scale so priorities combine evenly.
# Prefer the AI sub-score (s.*) when present; fall back to a spec-derived
# proxy computed in SQL so unscored phones can still rank.
PRIORITY_SQL_EXPR: dict[str, str] = {
    "camera":        "COALESCE(s.camera_score, LEAST(COALESCE(p.main_camera_mp, 0) / 20.0, 10))",
    "battery":       "COALESCE(s.battery_score, LEAST(COALESCE(p.battery_capacity, 0) / 600.0, 10))",
    "performance":   "COALESCE(s.performance_score, LEAST(COALESCE(p.antutu_score, 0) / 150000.0, 10))",
    "compact":       "GREATEST(0, 10 - (COALESCE(p.screen_size, 6.5) - 5.5) * 4)",
    "lightweight":   "GREATEST(0, 10 - (COALESCE(p.weight_g, 200) - 140) / 8.0)",
    "display":       "COALESCE(s.display_score, 6.0)",
    "fast_charging": "LEAST(COALESCE(p.fast_charging_w, 0) / 10.0, 10)",
    "value":         "COALESCE(s.value_score, 6.0)",
}
