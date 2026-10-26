"""SELECT lists and sort maps over the `phone_catalog` view (alias `p`).

Everything that reads phones goes through phone_catalog, which already carries
the effective price (see migrations/001_v2_api.sql: phone_prices), so no query
in the API computes a price on its own.
"""
from __future__ import annotations

# Phones the public site lists. Upcoming phones get pages but never appear in
# "best" lists or search unless asked for.
LISTABLE = "p.status = 'available'"

# Matches phones announced but not yet on sale. Statuses vary in the data, so
# this is a pattern plus a future release date. Check
#   SELECT status, count(*) FROM phones GROUP BY 1;
# and tighten if needed.
UPCOMING = (
    "(p.status <> 'available' AND p.status <> 'discontinued' AND "
    "(p.release_ts > EXTRACT(EPOCH FROM now()) OR "
    "p.status ILIKE ANY (ARRAY['%upcoming%', '%coming%', '%rumor%', '%rumour%', '%announced%'])))"
)

UPCOMING_STAGE = (
    "(CASE WHEN p.status ILIKE '%rumo%' THEN 'rumored' "
    "WHEN p.release_ts BETWEEN EXTRACT(EPOCH FROM now()) "
    "AND EXTRACT(EPOCH FROM now() + interval '60 days') THEN 'launching_soon' "
    "ELSE 'announced' END)"
)

UPCOMING_STAGE_RANK = (
    f"(CASE {UPCOMING_STAGE} WHEN 'launching_soon' THEN 0 WHEN 'announced' THEN 1 ELSE 2 END)"
)

NOT_REGIONAL = "NOT COALESCE(p.is_regional_edition, FALSE)"

MINI_SELECT = """p.id, p.brand, p.brand_slug, p.name, p.slug, p.full_slug, p.display_name,
p.main_image_url, p.price_usd, p.price_basis, p.price_region, p.tier, p.overall_score"""

SCORE_SELECT = """p.camera_score, p.performance_score, p.battery_score, p.display_score,
p.build_score, p.durability_score"""

# Lean list card: only what card surfaces render. Scores and spec columns are
# opt-in through the tiers below.
CARD_SELECT = """p.id, p.brand, p.brand_slug, p.name, p.slug, p.full_slug, p.display_name,
p.status, p.main_image_url, p.release_year, p.release_ts, p.tier,
p.price_usd, p.price_basis, p.price_region,
p.overall_score, p.value_score,
p.battery_capacity, p.screen_size, p.fast_charging_w, p.main_camera_mp, p.ram_max_gb"""

_SPEC_SELECT = """p.chipset, p.weight_g, p.ram_options, p.antutu_score, p.refresh_rate_hz, p.has_ois,
p.has_wireless_charging, p.wireless_charging_w, p.water_resistance, p.thickness_mm,
p.screen_resolution, p.display_type, p.peak_brightness_nits, p.geekbench_single, p.gpu_score,
p.storage_options, p.build_material, p.optical_zoom_x, p.selfie_mp, p.camera_setup_type, p.price_url"""

# Phone page header, buy box and insights. search_doc and offer_scopes stay internal.
DETAIL_SELECT = (
    CARD_SELECT + ",\n" + SCORE_SELECT + ",\n" + _SPEC_SELECT + """,
p.release_date_full, p.camera_summary, p.has_telephoto, p.has_headphone_jack,
p.is_premium_gaming, p.is_water_resistant, p.is_foldable"""
)

# Compare: the columns read by SPEC_SECTIONS / VERDICTS plus the score axes. No analysis.
COMPARE_SELECT = CARD_SELECT + ",\n" + SCORE_SELECT + ",\n" + _SPEC_SELECT

# Category lists: score axes feed best_for / reason, then the route strips them.
CATEGORY_SELECT = (
    CARD_SELECT + ",\n" + SCORE_SELECT +
    ",\np.has_ois, p.has_telephoto, p.water_resistance, p.antutu_score, p.refresh_rate_hz, p.weight_g"
)

# Recommend: the extra columns are prompt context for the AI copy.
RECOMMEND_SELECT = CARD_SELECT + ",\np.chipset, p.weight_g, p.is_foldable, p.has_wireless_charging"

SORT_COL_MAP: dict[str, str] = {
    "release_ts": "p.release_ts",
    "release_year": "p.release_ts",
    "price_usd": "p.price_usd",
    "battery_capacity": "p.battery_capacity",
    "main_camera_mp": "p.main_camera_mp",
    "antutu_score": "p.antutu_score",
    "weight_g": "p.weight_g",
    "screen_size": "p.screen_size",
    "fast_charging_w": "p.fast_charging_w",
    "refresh_rate_hz": "p.refresh_rate_hz",
    "popularity": "p.trend_score",
    "trending": "p.trend_score",
    "overall_score": "p.overall_score",
    "value_score": "p.value_score",
    "camera_score": "p.camera_score",
    "battery_score": "p.battery_score",
    "performance_score": "p.performance_score",
    "display_score": "p.display_score",
    "durability_score": "p.durability_score",
}
