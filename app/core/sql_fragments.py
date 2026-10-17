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

CARD_SELECT = """p.id, p.brand, p.brand_slug, p.name, p.slug, p.full_slug, p.display_name,
p.status, p.main_image_url, p.release_year, p.release_ts, p.tier,
p.price_usd, p.price_basis, p.price_region,
p.overall_score, p.camera_score, p.performance_score, p.battery_score, p.display_score,
p.build_score, p.value_score, p.durability_score,
p.battery_capacity, p.chipset, p.screen_size, p.weight_g, p.fast_charging_w,
p.antutu_score, p.main_camera_mp, p.ram_max_gb, p.ram_options, p.refresh_rate_hz,
p.has_nfc, p.has_ois, p.has_telephoto, p.has_wireless_charging, p.is_foldable,
p.is_water_resistant, p.has_headphone_jack, p.is_premium_gaming, p.water_resistance,
p.views_7d"""

# Everything a phone page or compare table needs. search_doc and offer_scopes
# are internal and never returned.
DETAIL_SELECT = CARD_SELECT + """,
p.release_month, p.release_day, p.release_date_full, p.source_url, p.price_updated_at,
p.catalog_price_usd, p.price_retailer, p.price_url, p.scored_at, p.created_at, p.updated_at,
p.thickness_mm, p.screen_resolution, p.display_type, p.peak_brightness_nits,
p.measured_brightness_nits, p.battery_material, p.wireless_charging_w, p.has_reverse_wireless,
p.antutu_version, p.geekbench_single, p.geekbench_multi, p.gpu_score,
p.storage_options, p.storage_max_gb, p.build_material, p.design_form, p.sim_layout,
p.network_generation, p.main_sensor_format, p.main_sensor_area_mm2, p.main_aperture,
p.optical_zoom_x, p.has_optical_zoom, p.telephoto_mp, p.is_periscope, p.has_ultrawide,
p.ultrawide_mp, p.video_max_4k_fps, p.has_8k, p.has_dolby_vision, p.has_log_video,
p.selfie_mp, p.camera_setup_type, p.camera_summary, p.video_resolution,
p.compares_30d"""

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
