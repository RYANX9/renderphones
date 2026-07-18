from __future__ import annotations

import decimal
import logging
from datetime import date, datetime
from typing import Any

import asyncpg

logger = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None


async def create_pool(dsn: str, *, min_size: int, max_size: int, command_timeout: float) -> asyncpg.Pool:
    global _pool
    _pool = await asyncpg.create_pool(
        dsn,
        min_size=min_size,
        max_size=max_size,
        command_timeout=command_timeout,
        init=_init_conn,
    )
    logger.info("DB pool ready min=%d max=%d", min_size, max_size)
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("DB pool not initialised")
    return _pool


async def _init_conn(conn: asyncpg.Connection) -> None:
    import json
    for pg_type in ("json", "jsonb"):
        await conn.set_type_codec(
            pg_type, encoder=json.dumps, decoder=json.loads, schema="pg_catalog",
        )


def _is_char_split_corruption(s: str) -> bool:
    """Detects the scraper bug where a string got exploded to
    ','.join(list(original_string)) somewhere upstream, e.g.
    'Glass front' -> 'G, l, a, s, s,  , f, r, o, n, t'.
    Signature: comma+space separated single characters making up
    a large fraction of the string length.
    """
    if not s or len(s) < 20:
        return False
    parts = s.split(", ")
    if len(parts) < 8:
        return False
    single_char_parts = sum(1 for p in parts if len(p) <= 1)
    return single_char_parts / len(parts) > 0.6


def repair_char_split(s: str | None) -> str | None:
    """Collapses a char-split-corrupted string back to normal text.
    Safe no-op on already-clean strings."""
    if s is None:
        return None
    if not _is_char_split_corruption(s):
        return s
    return "".join(p for p in s.split(", "))


def _serialize(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (datetime, date)):
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


# Columns known to carry the char-split scraper corruption. Repaired
# transparently at read time; the underlying data is still broken and
# should be fixed at the import/scrape stage.
_CHAR_SPLIT_REPAIR_COLUMNS = frozenset({"build_material"})


def row_to_dict(row: asyncpg.Record | None) -> dict | None:
    if row is None:
        return None
    out: dict[str, Any] = {}
    for k, v in dict(row).items():
        val = _serialize(v)
        if k in _CHAR_SPLIT_REPAIR_COLUMNS and isinstance(val, str):
            val = repair_char_split(val)
        out[k] = val
    return out


def rows_to_list(rows: list[asyncpg.Record]) -> list[dict]:
    return [row_to_dict(r) for r in rows]  # type: ignore[misc]
