"""
Composable filter/search WHERE builder for phones + phone_specs.

Design: every filter appends a `$n` positional condition and its param;
callers get back (where_sql, params, extra_select) so search endpoints can
also expose a relevance score without duplicating this logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .sql_fragments import SORT_COL_MAP

# Brand/line aliasing so "s24" finds "Galaxy S24", "iphone 16" finds
# "Apple iPhone 16", etc. Expanded from the old utils/query.py table with
# a few more common shorthands users actually type.
_SEARCH_EXPANSIONS: dict[str, str] = {
    "samsung s": "samsung galaxy s",
    "samsung a": "samsung galaxy a",
    "samsung z": "samsung galaxy z",
    "galaxy s": "samsung galaxy s",
    "redmi ": "xiaomi redmi ",
    "poco ": "xiaomi poco ",
    "moto ": "motorola moto ",
    "nothing ": "nothing phone ",
    "asus rog": "asus rog phone",
    "sony xperia": "sony xperia",
    "iphone": "apple iphone",
    "pixel": "google pixel",
}


def expand_query_alias(q: str) -> str:
    lower = q.lower().strip()
    for short, full in _SEARCH_EXPANSIONS.items():
        if lower.startswith(short) and not lower.startswith(full):
            return full + lower[len(short):]
    return q


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

    chipset_tier: str | None = None  # flagship | upper_mid | mid | entry

    has_nfc: bool | None = None
    has_ois: bool | None = None
    has_wireless_charging: bool | None = None
    has_headphone_jack: bool | None = None
    is_foldable: bool | None = None
    is_premium_gaming: bool | None = None

    camera_setup_type: str | None = None  # single/dual/triple/quad
    water_resistant: bool | None = None   # any IP rating present

    exclude_ids: list[int] = field(default_factory=list)


# Regex-based chipset tiering. Kept and tightened from the previous
# implementation, now sourced against the real `s.chipset` TEXT column.
_TIER_PATTERNS: dict[str, str] = {
    "flagship": (
        r"snapdragon 8 elite|snapdragon 8s elite"
        r"|snapdragon 8[+s]? gen [1-9]"
        r"|dimensity 9[0-9]{3}"
        r"|exynos 2[0-9]{3}"
        r"|apple a1[4-9]|apple a[2-9][0-9]"
        r"|tensor g[3-9]"
        r"|kirin 9[0-9]{3}"
    ),
    "upper_mid": (
        r"snapdragon 7[+s]? gen"
        r"|snapdragon 6 gen"
        r"|dimensity [78][0-9]{2,3}"
        r"|exynos 1[0-9]{3}"
        r"|kirin 8[0-9]{2}"
        r"|tensor g[12]"
        r"|apple a1[0-3]"
    ),
}


def build_filter_where(f: FilterParams) -> tuple[str, list[Any]]:
    """Returns (WHERE clause body, params). Caller prefixes with 'WHERE'."""
    conditions: list[str] = ["1=1"]
    params: list[Any] = []
    i = 1

    if f.q and f.q.strip():
        expanded = expand_query_alias(f.q.strip()).lower()
        conditions.append(
            f"(LOWER(p.model_name) LIKE ${i}"
            f" OR LOWER(p.brand) LIKE ${i}"
            f" OR LOWER(s.chipset) LIKE ${i}"
            # trigram similarity catches typos ("iphon 16" -> "iphone 16")
            # without requiring an exact substring match
            f" OR similarity(LOWER(p.model_name), ${i + 1}) > 0.25)"
        )
        params.append(f"%{expanded}%")
        params.append(expanded)
        i += 2

    if f.brand:
        conditions.append(f"LOWER(p.brand) = ${i}")
        params.append(f.brand.lower())
        i += 1

    if f.brands:
        conditions.append(f"LOWER(p.brand) = ANY(${i})")
        params.append([b.lower() for b in f.brands])
        i += 1

    if f.chipset:
        conditions.append(f"LOWER(s.chipset) LIKE ${i}")
        params.append(f"%{f.chipset.lower()}%")
        i += 1

    _RANGE: list[tuple[Any, str, str]] = [
        (f.min_price, "p.price_usd", ">="),
        (f.max_price, "p.price_usd", "<="),
        (f.min_battery, "s.battery_capacity", ">="),
        (f.min_camera_mp, "s.main_camera_mp", ">="),
        (f.min_screen_size, "s.screen_size", ">="),
        (f.max_screen_size, "s.screen_size", "<="),
        (f.min_year, "p.release_year", ">="),
        (f.max_year, "p.release_year", "<="),
        (f.max_weight, "s.weight_g", "<="),
        (f.min_charging_w, "s.fast_charging_w", ">="),
        (f.min_refresh_rate, "s.refresh_rate_hz", ">="),
        (f.min_antutu, "s.antutu_score", ">="),
    ]
    for value, col, op in _RANGE:
        if value is not None:
            conditions.append(f"{col} {op} ${i}")
            params.append(value)
            i += 1

    if f.min_ram is not None:
        conditions.append(f"EXISTS (SELECT 1 FROM unnest(s.ram_options) r WHERE r >= ${i})")
        params.append(f.min_ram)
        i += 1

    if f.min_storage is not None:
        conditions.append(f"EXISTS (SELECT 1 FROM unnest(s.storage_options) st WHERE st >= ${i})")
        params.append(f.min_storage)
        i += 1

    _BOOL: list[tuple[bool | None, str]] = [
        (f.has_nfc, "s.has_nfc"),
        (f.has_ois, "s.has_ois"),
        (f.has_wireless_charging, "s.has_wireless_charging"),
        (f.has_headphone_jack, "s.has_headphone_jack"),
        (f.is_foldable, "s.is_foldable"),
        (f.is_premium_gaming, "s.is_premium_gaming"),
    ]
    for value, col in _BOOL:
        if value is not None:
            conditions.append(f"{col} IS {'TRUE' if value else 'FALSE'}")

    if f.water_resistant is True:
        conditions.append("s.water_resistance IS NOT NULL AND s.water_resistance ILIKE '%IP%'")
    elif f.water_resistant is False:
        conditions.append("(s.water_resistance IS NULL OR s.water_resistance NOT ILIKE '%IP%')")

    if f.camera_setup_type:
        conditions.append(f"LOWER(s.camera_setup_type) = ${i}")
        params.append(f.camera_setup_type.lower())
        i += 1

    if f.chipset_tier and f.chipset_tier in _TIER_PATTERNS:
        conditions.append(f"LOWER(s.chipset) ~ ${i}")
        params.append(_TIER_PATTERNS[f.chipset_tier])
        i += 1
    elif f.chipset_tier == "entry":
        # entry = not matching flagship or upper_mid, and has a chipset at all
        conditions.append(
            f"s.chipset IS NOT NULL AND LOWER(s.chipset) !~ ${i} AND LOWER(s.chipset) !~ ${i + 1}"
        )
        params.append(_TIER_PATTERNS["flagship"])
        params.append(_TIER_PATTERNS["upper_mid"])
        i += 2

    if f.exclude_ids:
        conditions.append(f"p.id != ALL(${i})")
        params.append(f.exclude_ids)
        i += 1

    return " AND ".join(conditions), params


def resolve_sort(sort_by: str, sort_order: str, has_query: bool) -> tuple[str, str]:
    """Returns (order-by expression, direction). Falls back to release
    date for unknown sort keys instead of erroring."""
    if sort_by == "relevance" and not has_query:
        sort_by = "release_ts"
    expr = SORT_COL_MAP.get(sort_by) or SORT_COL_MAP["release_ts"]
    order = "DESC" if sort_order.lower() == "desc" else "ASC"
    return expr, order
