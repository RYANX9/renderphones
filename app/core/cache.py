from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from typing import Any, Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class TTLCache:
    """Asyncio-safe LRU + TTL cache backed by an OrderedDict."""

    __slots__ = ("_store", "_max_size", "_lock", "_hits", "_misses")

    def __init__(self, max_size: int = 2_048) -> None:
        self._store: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._max_size = max_size
        self._lock = asyncio.Lock()
        self._hits = self._misses = 0

    async def get(self, key: str) -> Any:
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            value, exp = entry
            if time.monotonic() > exp:
                del self._store[key]
                self._misses += 1
                return None
            self._store.move_to_end(key)
            self._hits += 1
            return value

    async def set(self, key: str, value: Any, ttl: int) -> None:
        async with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            elif len(self._store) >= self._max_size:
                self._evict()
            self._store[key] = (value, time.monotonic() + ttl)

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._store.pop(key, None)

    async def delete_prefix(self, prefix: str) -> int:
        async with self._lock:
            keys = [k for k in self._store if k.startswith(prefix)]
            for k in keys:
                del self._store[k]
            return len(keys)

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()

    def _evict(self) -> None:
        now = time.monotonic()
        expired = [k for k, (_, exp) in self._store.items() if now > exp]
        for k in expired:
            del self._store[k]
        while len(self._store) >= self._max_size:
            self._store.popitem(last=False)

    @property
    def stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        return {
            "size": len(self._store),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total else 0.0,
        }


cache: TTLCache = TTLCache()


async def cached(key: str, ttl: int, fn: Callable[[], Awaitable[T]]) -> T:
    hit = await cache.get(key)
    if hit is not None:
        return hit  # type: ignore[return-value]
    result = await fn()
    await cache.set(key, result, ttl)
    return result
