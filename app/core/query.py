from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .market import allowed_scopes
from .scoring import TIER_FILTER_ALIASES
from .slugs import brand_slug_of
from .sql_fragments import LISTABLE, NOT_REGIONAL, SORT_COL_MAP, UPCOMING

# Feature tag keys currently defined in the feature_tags table. Loaded at
# startup; when empty any key the client sends is accepted (it is always passed
# as a bound parameter, never interpolated).
FEATURE_KEYS: set[str] = set()

_TOKEN_RE = re.compile(r"[a-z0-9+]+")
MAX_TOKENS = 6


async def load_feature_keys(conn) -> int:
    rows = await conn.fetch("SELECT key FROM feature_tags")
    FEATURE_KEYS.clear()
    FEATURE_KEYS.update(r["key"] for r in rows)
    return len(rows)


def like_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def tokenize(q: str) -> list[str]:
    return _TOKEN_RE.findall(q.lower())[:MAX_TOKENS]


@dataclass
class FilterParams:
    q: str | None = None
    brand: str | None = None
    brands: list[str] | None = None
    chipset: str | None = None
    min_price: float | None = None
    max_price: float | None = None
    min_ram: int | None = None
    min_storage: int | None = None
    min_battery: int | None = None
    min_camera_mp: int | None = None
    min_screen_size: float | None = None
    max_screen_size: float | None = None
    min_year: int | None = None
    max_year: int | None = None
    max_weight: int | None = None
    min_charging_w: int | None = None
    min_refresh_rate: int | None = None
    min_antutu: int | None = None
    tier: str | None = None
    has_nfc: bool | None = None
    has_ois: bool | None = None
    has_wireless_charging: bool | None = None
    has_headphone_jack: bool | None = None
    is_foldable: bool | None = None
    is_premium_gaming: bool | None = None
    camera_setup_type: str | None = None
    water_resistant: bool | None = None
    features: list[str] | None = None
    region: str | None = None
    exclude_ids: list[int] = field(default_factory=list)
    include_upcoming: bool = False
    only_upcoming: bool = False
    include_regional: bool = False


def build_filter_where(f: FilterParams) -> tuple[str, list[Any], str | None]:
    """Returns (where_sql, params, relevance_sql). Every value is a bound
    parameter; only column names and operators from fixed tables are
    interpolated."""
    conditions: list[str] = []
    params: list[Any] = []
    relevance_expr: str | None = None

    def add(value: Any) -> str:
        params.append(value)
        return f"${len(params)}"

    # listing scope
    if f.only_upcoming:
        conditions.append(UPCOMING)
    elif f.include_upcoming:
        conditions.append(f"({LISTABLE} OR {UPCOMING})")
    else:
        conditions.append(LISTABLE)
    if not f.include_regional:
        conditions.append(NOT_REGIONAL)

    # text search: every token must appear in search_doc (brand + name +
    # display name + chipset), so "samsung s24" and "iphone 12" just work with
    # no alias table. Typo tolerance comes from trigram similarity on the name.
    if f.q and f.q.strip():
        q = " ".join(f.q.strip().lower().split())
        tokens = tokenize(q)
        if tokens:
            token_sql = " AND ".join(f"p.search_doc LIKE {add('%' + like_escape(t) + '%')}" for t in tokens)
            raw, esc = add(q), add(like_escape(q))
            if len(q) >= 4:
                conditions.append(f"(({token_sql}) OR lower(p.display_name) % {raw} OR lower(p.name) % {raw})")
            else:
                conditions.append(f"({token_sql})")
            relevance_expr = (
                f"(CASE WHEN lower(p.name) = {raw} OR lower(p.display_name) = {raw} THEN 100 "
                f"WHEN lower(p.name) LIKE {esc} || '%' THEN 80 "
                f"WHEN lower(p.display_name) LIKE {esc} || '%' THEN 70 "
                f"WHEN lower(p.display_name) LIKE '%' || {esc} || '%' THEN 55 ELSE 0 END "
                f"+ similarity(lower(p.display_name), {raw}) * 20)"
            )

    if f.brand:
        conditions.append(f"p.brand_slug = {add(brand_slug_of(f.brand))}")
    if f.brands:
        conditions.append(f"p.brand_slug = ANY({add([brand_slug_of(b) for b in f.brands])}::text[])")
    if f.chipset:
        conditions.append(f"lower(p.chipset) LIKE {add('%' + like_escape(f.chipset.lower()) + '%')}")

    ranges = [
        (f.min_price, "p.price_usd", ">="), (f.max_price, "p.price_usd", "<="),
        (f.min_ram, "p.ram_max_gb", ">="), (f.min_storage, "p.storage_max_gb", ">="),
        (f.min_battery, "p.battery_capacity", ">="), (f.min_camera_mp, "p.main_camera_mp", ">="),
        (f.min_screen_size, "p.screen_size", ">="), (f.max_screen_size, "p.screen_size", "<="),
        (f.max_weight, "p.weight_g", "<="), (f.min_charging_w, "p.fast_charging_w", ">="),
        (f.min_refresh_rate, "p.refresh_rate_hz", ">="), (f.min_antutu, "p.antutu_score", ">="),
        (f.min_year, "p.release_year", ">="), (f.max_year, "p.release_year", "<="),
    ]
    for value, col, op in ranges:
        if value is not None:
            conditions.append(f"{col} {op} {add(value)}")

    for value, col in [
        (f.has_nfc, "p.has_nfc"), (f.has_ois, "p.has_ois"),
        (f.has_wireless_charging, "p.has_wireless_charging"),
        (f.has_headphone_jack, "p.has_headphone_jack"), (f.is_foldable, "p.is_foldable"),
        (f.is_premium_gaming, "p.is_premium_gaming"),
    ]:
        if value is not None:
            conditions.append(f"{col} IS {'TRUE' if value else 'FALSE'}")

    if f.water_resistant is True:
        conditions.append("p.is_water_resistant IS TRUE")
    elif f.water_resistant is False:
        conditions.append("p.is_water_resistant IS NOT TRUE")

    if f.camera_setup_type:
        conditions.append(f"p.camera_setup_type = {add(f.camera_setup_type.lower())}")

    if f.tier and f.tier in TIER_FILTER_ALIASES:
        conditions.append(f"p.tier = ANY({add(TIER_FILTER_ALIASES[f.tier])}::text[])")

    # features come from phone_features.feature_name matched against the
    # feature_tags patterns (feature_key on the row may be null), one EXISTS
    # per requested key = all requested features required.
    if f.features:
        keys = [k for k in dict.fromkeys(f.features) if not FEATURE_KEYS or k in FEATURE_KEYS]
        for key in keys:
            conditions.append(
                "EXISTS (SELECT 1 FROM phone_features pf JOIN feature_tags ft ON pf.feature_name ILIKE ft.pattern "
                f"WHERE pf.phone_id = p.id AND ft.key = {add(key)})"
            )

    scopes = allowed_scopes(f.region)
    if scopes is not None:
        conditions.append(f"(cardinality(p.offer_scopes) = 0 OR p.offer_scopes && {add(scopes)}::text[])")

    if f.exclude_ids:
        conditions.append(f"p.id <> ALL({add(f.exclude_ids)}::int[])")

    return " AND ".join(conditions), params, relevance_expr


def resolve_sort(sort_by: str, sort_order: str, has_query: bool, relevance_expr: str | None = None) -> tuple[str, str]:
    order = "DESC" if sort_order.lower() == "desc" else "ASC"
    if sort_by == "relevance":
        if has_query and relevance_expr:
            return relevance_expr, "DESC"
        sort_by = "release_ts"
    return SORT_COL_MAP.get(sort_by) or SORT_COL_MAP["release_ts"], order
