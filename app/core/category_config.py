from __future__ import annotations

CATEGORY_BRAND_CAP = 2
CATEGORY_POOL = 100

# Ranking eligibility: current calendar year + previous calendar year.
# With the current date 2026-10-06 this means 2025-01-01 through today.
RELEASE_YEARS_BACK = 1
INCLUDE_UPCOMING = True

# Freshness preference applied on the final 0-10 score.
# Current-year released phones receive this bonus; previous-year and upcoming
# phones receive no freshness bonus. This is deliberately modest so camera
# quality still matters more than age.
CURRENT_YEAR_FRESHNESS_BONUS = 0.15
MIN_COMPLETENESS = 0.0
CATEGORY_STORE_LIMIT: int | None = None
LINEUP_MARGIN = 0.02
LINEUP_WINDOW_DAYS = 400
REQUIRED_MODEL_VERSION: str | None = None

WEIGHT_CEILING_G = 180.0
WEIGHT_VALID_RANGE = (80.0, 400.0)
WEIGHT_BAND_G = 5.0
LIGHTWEIGHT_MIN_OVERALL = 7.0
LIGHTWEIGHT_MIN_SCREEN = 5.5

COMPACT_SCREEN_RANGE = (5.0, 6.3)
FAST_CHARGING_MIN_W = 45.0
WATER_MIN_IP = (6, 7)

# Placeholders until fitted against published endurance tests; outputs are labelled estimates.
CELL_VOLTAGE = 3.87
P0_W = 0.5
K_W_PER_CM2 = 0.015
CHARGE_EFFICIENCY = 0.80
CHARGE_TAPER_CAP = 0.80
CHARGE_WINDOW_H = 0.25

# Fixed (low, high) anchors, clipped to 0..1. Never derived from the pool.
ANCHORS = {
    "main_light": (5.0, 30.0),
    "zoom_x": (1.0, 10.0),
    "mp": (8.0, 50.0),
    "fps_4k": (24.0, 120.0),
    "battery_hours": (6.0, 25.0),
    "charge_hours_gained": (2.0, 10.0),
    "weight_g": (120.0, WEIGHT_CEILING_G),
    "antutu": (1_000_000.0, 3_500_000.0),
    "ram_gb": (6.0, 24.0),
    "refresh_hz": (60.0, 165.0),
    "compact_screen": COMPACT_SCREEN_RANGE,
    "compact_weight": (140.0, 220.0),
}

CAMERA_WEIGHTS = {"main": 0.40, "telephoto": 0.25, "video": 0.15, "ultrawide": 0.12, "selfie": 0.08}
GAMING_WEIGHTS = {"performance": 0.60, "ram": 0.10, "refresh": 0.10, "premium": 0.05}

CANDIDATE_SQL = """
SELECT
    p.id, p.brand, p.name, p.display_name, p.status, p.release_ts, p.is_regional_edition,
    p.price_usd, p.price_basis,
    p.overall_score, p.camera_score, p.performance_score, p.battery_score,
    p.display_score, p.build_score, p.value_score,
    p.screen_size, p.screen_resolution, p.refresh_rate_hz, p.battery_capacity,
    p.fast_charging_w, p.weight_g, p.chipset, p.antutu_score, p.ram_max_gb,
    p.is_premium_gaming, p.is_foldable, p.water_resistance,
    p.main_sensor_area_mm2, p.main_aperture, p.has_ois,
    p.has_telephoto, p.optical_zoom_x, p.telephoto_mp, p.is_periscope,
    p.has_ultrawide, p.ultrawide_mp, p.video_max_4k_fps, p.has_8k,
    p.has_dolby_vision, p.selfie_mp,
    COALESCE(
        fs.full_specifications->'specifications'->'Body'->>'Weight',
        fs.full_specifications->'Body'->>'Weight'
    ) AS weight_text
FROM phone_catalog p
LEFT JOIN phone_full_specifications fs ON fs.phone_id = p.id
LEFT JOIN phone_analysis a ON a.phone_id = p.id
WHERE ($1::text IS NULL OR a.model_version = $1)
  AND (
      (LOWER(COALESCE(p.status, '')) = 'upcoming' AND TRUE)
      OR (
          p.release_ts IS NOT NULL
          AND to_timestamp(p.release_ts) >= date_trunc('year', CURRENT_DATE) - interval '1 year'
          AND to_timestamp(p.release_ts) <= NOW()
      )
  )
"""

CATEGORY_CONFIG: dict[str, dict] = {
    "camera-phones": {
        "title": "Best Camera Phones",
        "description": (
            "Computed from listed camera hardware, not image tests: main sensor light-gathering "
            "(sensor area over aperture squared) and OIS, telephoto, video, ultrawide and selfie. "
            "Megapixels of the main camera are not used. 50% comes from our AI camera score when available; missing AI scores are estimated consistently from the category pool."
        ),
        "scorer": "camera",
        "lineup_guard": True,
        "ai_weight": 0.50,
        "ai_field": "camera_score",
    },
    "battery-life": {
        "title": "Best Battery Life",
        "description": (
            "Estimated screen-on hours from battery energy (Wh) and display area. "
            "An estimate from specifications, not a measured endurance test."
        ),
        "scorer": "battery",
        "ai_weight": 0.40,
        "ai_field": "battery_score",
    },
    "gaming-phones": {
        "title": "Best Gaming Phones",
        "description": (
            "Chipset performance (AnTuTu where listed, otherwise the median of phones with the "
            "same chipset, otherwise our AI performance score), plus RAM, refresh rate and "
            "gaming-oriented hardware. Sustained performance and thermals are not measured."
        ),
        "scorer": "gaming",
        "lineup_guard": True,
        "ai_weight": 0.40,
        "ai_field": "performance_score",
    },
    "fast-charging": {
        "title": "Fastest Charging Phones",
        "description": (
            f"Estimated runtime recovered in {int(CHARGE_WINDOW_H * 60)} minutes of wired charging, "
            f"from charging power, battery energy and display area. Minimum {int(FAST_CHARGING_MIN_W)}W. "
            "An estimate; charging curves taper and are not measured."
        ),
        "scorer": "fast_charging",
        "ai_weight": 0.40,
        "ai_field": "overall_score",
    },
    "lightweight": {
        "title": "Lightest Smartphones",
        "description": (
            f"Phones up to {int(WEIGHT_CEILING_G)}g by lowest listed weight, grouped in "
            f"{int(WEIGHT_BAND_G)}g bands and ordered by overall score within a band."
        ),
        "scorer": "lightweight",
        "order": "weight_band",
        "ai_weight": 0.40,
        "ai_field": "overall_score",
    },
    "compact-phones": {
        "title": "Best Compact Phones",
        "description": "Screens from 5.0 to 6.3 inches, ranked 75% on quality and 25% on footprint.",
        "scorer": "compact",
        "ai_weight": 0.40,
        "ai_field": "overall_score",
    },
    "foldables": {
        "title": "Best Foldable Phones",
        "description": "Foldable form factors ranked by overall score.",
        "scorer": "foldable",
        "lineup_guard": True,
        "ai_weight": 0.40,
        "ai_field": "overall_score",
    },
    "water-resistant": {
        "title": "Best Water/Dust Resistant Phones",
        "description": "IP67 or better, ranked by overall score.",
        "scorer": "water",
        "lineup_guard": True,
        "ai_weight": 0.40,
        "ai_field": "build_score",
    },
    "under-300": {
        "title": "Best Phones Under $300",
        "description": (
            "Phones priced at $300 or less, ranked on performance, battery, display and build. "
            "Prices flagged as estimated are tracked manually."
        ),
        "scorer": "budget",
        "price_min": 0.0,
        "price_limit": 300.0,
        "ai_weight": 0.40,
        "ai_field": "value_score",
    },
    "under-500": {
        "title": "Best Phones Under $500",
        "description": (
            "Phones priced above $300 and up to $500, ranked on performance, battery, display and build. "
            "Prices flagged as estimated are tracked manually."
        ),
        "scorer": "budget",
        "price_min": 300.0,
        "price_limit": 500.0,
        "ai_weight": 0.40,
        "ai_field": "value_score",
    },
    "under-700": {
        "title": "Best Phones Under $700",
        "description": (
            "Phones priced above $500 and up to $700, ranked on performance, battery, display and build. "
            "Prices flagged as estimated are tracked manually."
        ),
        "scorer": "budget",
        "price_min": 500.0,
        "price_limit": 700.0,
        "ai_weight": 0.40,
        "ai_field": "value_score",
    },
}

SLUG_ALIASES: dict[str, str] = {
    "camera": "camera-phones",
    "battery": "battery-life",
    "gaming": "gaming-phones",
    "lightweight": "lightweight",
    "compact": "compact-phones",
    "charging": "fast-charging",
    "foldable": "foldables",
    "waterproof": "water-resistant",
}


# Price bands are (price_min, price_limit]: a phone at exactly $300 is in under-300,
# a phone at $301 is in under-500. A phone above the top band appears in no price list.
PRICE_BAND_SLUGS = ("under-300", "under-500", "under-700")


def price_band_slug(price: float | None) -> str | None:
    if price is None or price <= 0:
        return None
    for slug in PRICE_BAND_SLUGS:
        spec = CATEGORY_CONFIG[slug]
        if spec["price_min"] < price <= spec["price_limit"]:
            return slug
    return None


# Grouping for the /best hub and the home page tiles.
CATEGORY_GROUPS: list[dict] = [
    {"id": "by_use", "title": "By what matters to you",
     "slugs": ["camera-phones", "battery-life", "gaming-phones", "fast-charging", "lightweight", "compact-phones", "foldables", "water-resistant"]},
    {"id": "by_price", "title": "By budget", "slugs": ["under-300", "under-500", "under-700"]},
]

# /pick priority id -> the category list that matches it (used for "see the full list" exits)
PRIORITY_CATEGORY: dict[str, str] = {
    "camera": "camera-phones", "battery": "battery-life", "gaming": "gaming-phones", "performance": "gaming-phones",
    "compact": "compact-phones", "lightweight": "lightweight", "fast_charging": "fast-charging",
    "foldable": "foldables", "durability": "water-resistant",
}
