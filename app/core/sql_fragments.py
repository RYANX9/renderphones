"""
Single source of truth for the phones/phone_specs/phone_smart_scores join.
Every route that lists or looks up phones uses PHONE_JOIN + PHONE_SELECT so
the row shape returned to the frontend never drifts between endpoints.
"""

from __future__ import annotations

# p = phones, s = phone_specs, sc = phone_smart_scores
PHONE_JOIN = """
FROM phones p
JOIN phone_specs s ON s.phone_id = p.id
LEFT JOIN phone_smart_scores sc ON sc.phone_id = p.id
"""

RELEASE_TS_EXPR = (
    "EXTRACT(EPOCH FROM MAKE_DATE("
    "COALESCE(p.release_year, 1970),"
    "COALESCE(p.release_month, 1),"
    "COALESCE(p.release_day, 1)"
    "))::bigint"
)

# Full row for detail/compare. full_specifications is included here only;
# list/search endpoints use PHONE_LIST_SELECT (below) which omits it since
# it can be tens of KB per row and is never rendered in a grid/list view.
PHONE_DETAIL_SELECT = f"""
p.id, p.model_name, p.brand, p.slug, p.source_url, p.main_image_url,
p.release_year, p.release_month, p.release_day, p.release_date_full,
p.price_usd, p.price_original, p.currency, p.price_updated_at,
p.availability_status, p.amazon_link, p.brand_link, p.popularity, p.fans,
{RELEASE_TS_EXPR} AS release_ts,

s.weight_g, s.thickness_mm, s.screen_size, s.screen_resolution,
s.display_type, s.refresh_rate_hz, s.peak_brightness_nits, s.measured_brightness_nits,
s.battery_capacity, s.battery_material, s.fast_charging_w,
s.has_wireless_charging, s.wireless_charging_w, s.has_reverse_wireless,
s.main_camera_mp, s.camera_setup_type, s.optical_zoom, s.has_ois,
s.camera_summary, s.chipset, s.antutu_score, s.geekbench_single,
s.geekbench_multi, s.gpu_score, s.is_premium_gaming,
s.ram_options, s.storage_options, s.water_resistance, s.build_material,
s.design_form, s.is_foldable, s.sim_layout, s.network_generation,
s.has_nfc, s.has_headphone_jack, s.full_specifications,

sc.overall_score AS smart_overall_score,
sc.camera_score AS smart_camera_score,
sc.performance_score AS smart_performance_score,
sc.battery_score AS smart_battery_score,
sc.display_score AS smart_display_score,
sc.build_score AS smart_build_score,
sc.value_score AS smart_value_score,
sc.tier AS smart_tier,
sc.reasoning AS smart_reasoning,
sc.strengths AS smart_strengths,
sc.weaknesses AS smart_weaknesses,
sc.model_version AS smart_model_version,
sc.scored_at AS smart_scored_at
"""

# Lighter projection for list/search/similar/compare-grid endpoints.
# Excludes full_specifications (large JSONB blob, unused in list UIs).
PHONE_LIST_SELECT = f"""
p.id, p.model_name, p.brand, p.slug, p.main_image_url,
p.release_year, p.release_month, p.release_day,
p.price_usd, p.price_original, p.currency, p.availability_status,
p.amazon_link, p.popularity,
{RELEASE_TS_EXPR} AS release_ts,

s.screen_size, s.screen_resolution, s.display_type, s.refresh_rate_hz,
s.battery_capacity, s.fast_charging_w, s.has_wireless_charging,
s.main_camera_mp, s.camera_setup_type, s.optical_zoom, s.has_ois,
s.chipset, s.antutu_score, s.gpu_score, s.is_premium_gaming,
s.ram_options, s.storage_options, s.water_resistance, s.is_foldable,
s.weight_g, s.has_nfc, s.has_headphone_jack,

sc.overall_score AS smart_overall_score,
sc.camera_score AS smart_camera_score,
sc.performance_score AS smart_performance_score,
sc.battery_score AS smart_battery_score,
sc.display_score AS smart_display_score,
sc.build_score AS smart_build_score,
sc.value_score AS smart_value_score,
sc.tier AS smart_tier,
sc.reasoning AS smart_reasoning,
sc.strengths AS smart_strengths,
sc.weaknesses AS smart_weaknesses,
sc.model_version AS smart_model_version,
sc.scored_at AS smart_scored_at
"""

# Columns a caller may sort by, mapped to their real SQL expression.
# COALESCE(..., 0)-style defaults keep NULLS from silently reordering
# a "best X first" sort to the top.
SORT_COL_MAP: dict[str, str] = {
    "release_ts": RELEASE_TS_EXPR,
    "release_year": RELEASE_TS_EXPR,
    "price_usd": "p.price_usd",
    "battery_capacity": "s.battery_capacity",
    "main_camera_mp": "s.main_camera_mp",
    "antutu_score": "s.antutu_score",
    "gpu_score": "s.gpu_score",
    "weight_g": "s.weight_g",
    "popularity": "p.popularity",
    "screen_size": "s.screen_size",
    "fast_charging_w": "s.fast_charging_w",
    "refresh_rate_hz": "s.refresh_rate_hz",
    "overall_score": "COALESCE(sc.overall_score, 0)",
    "value_score": "COALESCE(sc.value_score, 0)",
    "relevance": None,  # handled specially by search_where when q is present
}
