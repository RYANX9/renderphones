from __future__ import annotations

PRIORITY_LABELS: dict[str, str] = {
    "camera": "Camera Quality", "battery": "Battery Life", "performance": "Performance",
    "gaming": "Gaming Performance", "compact": "Compact Size", "lightweight": "Lightweight",
    "display": "Display Quality", "smooth_display": "High Refresh Rate", "fast_charging": "Fast Charging",
    "wireless_charging": "Wireless Charging", "foldable": "Foldable Design", "durability": "Water/Dust Resistance",
    "value": "Best Value", "headphone_jack": "Headphone Jack", "nfc": "NFC",
}

HARD_FILTER_PRIORITIES: dict[str, str] = {
    "foldable": "p.is_foldable IS TRUE",
    "headphone_jack": "p.has_headphone_jack IS TRUE",
    "nfc": "p.has_nfc IS TRUE",
}

PRIORITY_SQL_EXPR: dict[str, str] = {
    "camera": "COALESCE(p.camera_score, LEAST(COALESCE(p.main_camera_mp, 0) / 20.0, 10) + CASE WHEN p.has_ois THEN 1.0 ELSE 0 END + CASE WHEN p.has_optical_zoom THEN 1.0 ELSE 0 END)",
    "battery": "COALESCE(p.battery_score, LEAST(COALESCE(p.battery_capacity, 0) / 600.0, 10))",
    "performance": "COALESCE(p.performance_score, LEAST(COALESCE(p.antutu_score, 0) / 150000.0, 10))",
    "gaming": "CASE WHEN p.is_premium_gaming THEN 10 ELSE COALESCE(p.performance_score, LEAST(COALESCE(p.antutu_score, 0) / 150000.0, 10)) END",
    "compact": "GREATEST(0, 10 - (COALESCE(p.screen_size, 6.5) - 5.5) * 4)",
    "lightweight": "GREATEST(0, 10 - (COALESCE(p.weight_g, 200) - 140) / 8.0)",
    "display": "COALESCE(p.display_score, LEAST(COALESCE(p.refresh_rate_hz, 60) / 12.0, 10))",
    "smooth_display": "LEAST(COALESCE(p.refresh_rate_hz, 60) / 12.0, 10)",
    "fast_charging": "LEAST(COALESCE(p.fast_charging_w, 0) / 10.0, 10)",
    "wireless_charging": "CASE WHEN p.has_wireless_charging THEN LEAST(COALESCE(p.wireless_charging_w, 10) / 15.0, 10) ELSE 0 END",
    "durability": "COALESCE(p.durability_score, 3)",
    "value": "COALESCE(p.value_score, 6.0)",
}

TIER_FILTER_ALIASES: dict[str, list[str]] = {
    "ultra_flagship": ["ultra_flagship"],
    "flagship": ["flagship"],
    "upper_mid_range": ["upper_mid_range"],
    "upper_mid": ["upper_mid_range"],
    "mid": ["upper_mid_range"],
    "mid_range": ["mid_range"],
    "budget": ["budget"],
    "entry": ["mid_range", "budget"],
}

_SIMILARITY_WEIGHTS = {
    "price": 0.40, "brand": 0.15, "tier": 0.15, "camera": 0.15, "battery": 0.10, "screen": 0.05,
}


def similarity_score(base: dict, candidate: dict) -> float:
    score = 0.0
    base_price = base.get("price_usd")
    cand_price = candidate.get("price_usd")
    if base_price and cand_price:
        diff_ratio = abs(base_price - cand_price) / base_price
        score += _SIMILARITY_WEIGHTS["price"] * max(0.0, 1.0 - min(diff_ratio / 0.5, 1.0))
    if base.get("brand") and base["brand"] == candidate.get("brand"):
        score += _SIMILARITY_WEIGHTS["brand"]
    base_tier = base.get("tier")
    cand_tier = candidate.get("tier")
    if base_tier is not None and cand_tier is not None and base_tier == cand_tier:
        score += _SIMILARITY_WEIGHTS["tier"]
    for field, key, scale in (("camera", "main_camera_mp", 108), ("battery", "battery_capacity", 2000), ("screen", "screen_size", 1.0)):
        bv, cv = base.get(key), candidate.get(key)
        if bv and cv:
            closeness = max(0.0, 1.0 - min(abs(float(bv) - float(cv)) / scale, 1.0))
            score += _SIMILARITY_WEIGHTS[field] * closeness
    return round(score, 4)
