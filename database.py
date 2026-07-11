from __future__ import annotations

import decimal
import json
import logging
from datetime import datetime
from typing import Any

import asyncpg

logger = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None


async def create_pool(
    dsn: str,
    *,
    min_size: int,
    max_size: int,
    command_timeout: float,
) -> asyncpg.Pool:
    global _pool
    _pool = await asyncpg.create_pool(
        dsn,
        min_size=min_size,
        max_size=max_size,
        command_timeout=command_timeout,
        init=_init_conn,
    )
    logger.info("DB pool ready  min=%d  max=%d", min_size, max_size)
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        logger.info("DB pool closed")


def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("DB pool not initialised")
    return _pool


async def _init_conn(conn: asyncpg.Connection) -> None:
    for pg_type in ("json", "jsonb"):
        await conn.set_type_codec(
            pg_type,
            encoder=json.dumps,
            decoder=json.loads,
            schema="pg_catalog",
        )


def _serialize(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, datetime):
        return v.isoformat()
    if isinstance(v, decimal.Decimal):
        return float(v)
    if isinstance(v, (int, float, str)):
        return v
    if isinstance(v, dict):
        return {k: _serialize(vv) for k, vv in v.items()}
    if isinstance(v, (list, tuple)):
        return [_serialize(i) for i in v]
    return str(v)


def row_to_dict(row: asyncpg.Record | None) -> dict | None:
    if row is None:
        return None
    return {k: _serialize(v) for k, v in dict(row).items()}


def rows_to_list(rows: list[asyncpg.Record]) -> list[dict]:
    return [row_to_dict(r) for r in rows]  # type: ignore[misc]


PHONE_LIST_SELECT = (
    "*, "
    "EXTRACT(EPOCH FROM MAKE_DATE("
    "  COALESCE(release_year, 1970),"
    "  COALESCE(release_month, 1),"
    "  COALESCE(release_day,  1)"
    "))::bigint AS release_ts"
)

RELEASE_TS_EXPR = (
    "EXTRACT(EPOCH FROM MAKE_DATE("
    "  COALESCE(release_year, 1970),"
    "  COALESCE(release_month, 1),"
    "  COALESCE(release_day,  1)"
    "))::bigint"
)

# Used by routes/phones.py: every list/detail query there is
# `FROM phones p LEFT JOIN phone_smart_scores s ON s.phone_id = p.id`,
# so this pulls p.* plus release_ts plus the AI sub-scores, prefixed
# `smart_*` so they never collide with a real phones column and can be
# popped off cleanly by utils.scoring.attach_computed_fields /
# routes/phones.py's own smart-score extraction.
PHONE_SCORED_SELECT = (
    "p.*, "
    "EXTRACT(EPOCH FROM MAKE_DATE("
    "  COALESCE(p.release_year, 1970),"
    "  COALESCE(p.release_month, 1),"
    "  COALESCE(p.release_day,  1)"
    "))::bigint AS release_ts, "
    "s.overall_score AS smart_overall_score, "
    "s.value_score AS smart_value_score, "
    "s.camera_score AS smart_camera_score, "
    "s.performance_score AS smart_performance_score, "
    "s.battery_score AS smart_battery_score, "
    "s.display_score AS smart_display_score, "
    "s.build_score AS smart_build_score, "
    "s.strengths AS smart_strengths, "
    "s.weaknesses AS smart_weaknesses, "
    "s.reasoning AS smart_reasoning, "
    "s.model_version AS smart_model_version, "
    "s.scored_at AS smart_scored_at, "
    "s.tier AS smart_tier"
)

PHONE_SCORED_FROM = "FROM phones p LEFT JOIN phone_smart_scores s ON s.phone_id = p.id"

SORT_COL_MAP: dict[str, str] = {
    "release_year": RELEASE_TS_EXPR,
    "release_ts":   RELEASE_TS_EXPR,
    "price_usd":        "price_usd",
    "battery_capacity": "battery_capacity",
    "main_camera_mp":   "main_camera_mp",
    "antutu_score":     "antutu_score",
    "weight_g":         "weight_g",
    "popularity":       "popularity",
}

RELEASE_TS_EXPR_P = (
    "EXTRACT(EPOCH FROM MAKE_DATE("
    "  COALESCE(p.release_year, 1970),"
    "  COALESCE(p.release_month, 1),"
    "  COALESCE(p.release_day,  1)"
    "))::bigint"
)

# Sort columns valid only for the `p`/`s`-aliased queries in routes/phones.py
# (they reference the phone_smart_scores join). Kept separate from
# SORT_COL_MAP so brands.py / other unaliased queries can't be handed a
# `s.` reference and blow up.
SCORED_SORT_COL_MAP: dict[str, str] = {
    "release_year":      RELEASE_TS_EXPR_P,
    "release_ts":        RELEASE_TS_EXPR_P,
    "price_usd":         "p.price_usd",
    "battery_capacity":  "p.battery_capacity",
    "main_camera_mp":    "p.main_camera_mp",
    "antutu_score":      "p.antutu_score",
    "weight_g":          "p.weight_g",
    "popularity":        "p.popularity",
    "overall_score":     "COALESCE(s.overall_score, 0)",
    "value_score":       "COALESCE(s.value_score, 0)",
}
