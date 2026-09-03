# app/core/database.py
from __future__ import annotations

import asyncio
import decimal
import logging
from contextlib import asynccontextmanager
from datetime import date, datetime
from typing import Any, AsyncIterator

import asyncpg

logger = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None

# Transient acquire failures (stale pooled connection, dropped handshake)
# get retried before we give up and let the route's own error handling
# take over. Not for query-level errors — only for acquiring the
# connection itself.
_ACQUIRE_RETRIES = 2
_ACQUIRE_RETRY_DELAY_S = 0.25


async def create_pool(dsn: str, *, min_size: int, max_size: int, command_timeout: float) -> asyncpg.Pool:
    global _pool
    _pool = await asyncpg.create_pool(
        dsn,
        min_size=min_size,
        max_size=max_size,
        command_timeout=command_timeout,
        init=_init_conn,
        # Recycle connections before a remote provider's own idle timeout
        # has a chance to kill them out from under the pool — this is
        # what usually causes "connection was closed in the middle of
        # operation" when acquiring a fresh one.
        max_inactive_connection_lifetime=180.0,
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


@asynccontextmanager
async def acquire_conn() -> AsyncIterator[asyncpg.Connection]:
    """Use instead of `get_pool().acquire()` directly in routes. Retries
    a small, bounded number of times on transient connection-acquire
    failures only (stale pooled connection, dropped handshake) — never
    retries query execution, so it can't double-run a write."""
    pool = get_pool()
    last_exc: Exception | None = None

    for attempt in range(_ACQUIRE_RETRIES + 1):
        try:
            async with pool.acquire() as conn:
                yield conn
                return
        except (asyncpg.exceptions.ConnectionDoesNotExistError,
                asyncpg.exceptions.ConnectionFailureError,
                asyncpg.exceptions.InterfaceError) as exc:
            last_exc = exc
            if attempt < _ACQUIRE_RETRIES:
                logger.warning(
                    "DB acquire failed (attempt %d/%d), retrying: %s",
                    attempt + 1, _ACQUIRE_RETRIES + 1, exc,
                )
                await asyncio.sleep(_ACQUIRE_RETRY_DELAY_S * (attempt + 1))
                continue
            raise

    if last_exc:
        raise last_exc


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