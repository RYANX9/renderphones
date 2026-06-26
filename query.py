from __future__ import annotations

import json
from typing import Any

from database import SORT_COL_MAP, RELEASE_TS_EXPR


_SEARCH_EXPANSIONS = {
    "samsung s":  "samsung galaxy s",
    "samsung a":  "samsung galaxy a",
    "samsung z":  "samsung galaxy z",
    "redmi ":     "xiaomi redmi ",
    "poco ":      "xiaomi poco ",
    "moto ":      "motorola moto ",
    "nothing ":   "nothing phone ",
    "asus rog":   "asus rog phone",
    "sony ":      "sony xperia ",
}


def expand_search_query(q: str) -> str:
    lower = q.lower().strip()
    for short, full in _SEARCH_EXPANSIONS.items():
        if lower.startswith(short) and not lower.startswith(full):
            return full + lower[len(short):]
    return q


def slug_to_words(slug: str) -> str:
    return slug.replace("-", " ")


def parse_json_safe(val: Any) -> dict | list | None:
    if val is None:
        return None
    if isinstance(val, (dict, list)):
        return val
    try:
        return json.loads(val)
    except Exception:
        return None


def resolve_sort(sort_by: str, sort_order: str) -> tuple[str, str]:
    expr = SORT_COL_MAP.get(sort_by, RELEASE_TS_EXPR)
    order = "DESC" if sort_order.lower() == "desc" else "ASC"
    return expr, order


def build_search_where(
    q: str | None = None,
    brand: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    min_ram: int | None = None,
    min_battery: int | None = None,
    min_camera_mp: int | None = None,
    min_screen_size: float | None = None,
    max_screen_size: float | None = None,
    min_year: int | None = None,
    max_weight: int | None = None,
    min_charging_w: int | None = None,
    chipset_tier_filter: str | None = None,
) -> tuple[str, list[Any]]:
    """Return (WHERE clause string, positional params list)."""
    conditions: list[str] = ["1=1"]
    params: list[Any] = []
    i = 1

    if q and q.strip():
        expanded = expand_search_query(q.strip()).lower()
        conditions.append(
            f"(LOWER(model_name) LIKE ${i}"
            f" OR LOWER(brand) LIKE ${i}"
            f" OR LOWER(chipset) LIKE ${i})"
        )
        params.append(f"%{expanded}%")
        i += 1

    if brand:
        conditions.append(f"LOWER(brand) = ${i}")
        params.append(brand.lower())
        i += 1

    _SIMPLE: list[tuple[Any, str, str]] = [
        (min_price,       "price_usd",       ">="),
        (max_price,       "price_usd",       "<="),
        (min_battery,     "battery_capacity",">="),
        (min_camera_mp,   "main_camera_mp",  ">="),
        (min_screen_size, "screen_size",     ">="),
        (max_screen_size, "screen_size",     "<="),
        (min_year,        "release_year",    ">="),
        (max_weight,      "weight_g",        "<="),
        (min_charging_w,  "fast_charging_w", ">="),
    ]
    for value, col, op in _SIMPLE:
        if value is not None:
            conditions.append(f"{col} {op} ${i}")
            params.append(value)
            i += 1

    if min_ram is not None:
        conditions.append(f"${i} = ANY(ram_options)")
        params.append(min_ram)
        i += 1

    if chipset_tier_filter:
        clause = _chipset_tier_sql(chipset_tier_filter)
        if clause:
            conditions.append(clause)

    return " AND ".join(conditions), params


def _chipset_tier_sql(tier: str) -> str:
    patterns: dict[str, str] = {
        "flagship": (
            "LOWER(chipset) ~ "
            "'snapdragon 8 elite|snapdragon 8s elite"
            "|snapdragon 8 gen [1-9]|snapdragon 8s gen [1-9]"
            "|dimensity 9[0-9]{3}|exynos 2[0-9]{3}"
            "|apple a1[4-9]|apple a[2-9][0-9]"
            "|tensor g[3-9]|kirin 9[0-9]{3}'"
        ),
        "mid": (
            "LOWER(chipset) ~ "
            "'snapdragon [67]|dimensity [78][0-9]{2,3}"
            "|exynos 1[0-9]{3}|kirin 8[0-9]{2}|tensor g[12]'"
        ),
        "entry": (
            "chipset IS NOT NULL"
            " AND LOWER(chipset) NOT LIKE '%snapdragon 8%'"
            " AND LOWER(chipset) NOT LIKE '%dimensity 9%'"
        ),
    }
    return patterns.get(tier, "")
