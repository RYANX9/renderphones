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

SORT_COL_MAP: dict[str, str] = {
    "release_year": RELEASE_TS_EXPR,
    "release_ts":   RELEASE_TS_EXPR,
    "price_usd":        "price_usd",
    "battery_capacity": "battery_capacity",
    "main_camera_mp":   "main_camera_mp",
    "antutu_score":     "antutu_score",
    "weight_g":         "weight_g",
}
