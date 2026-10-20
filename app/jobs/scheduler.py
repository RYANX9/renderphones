"""In-process scheduler (one asyncio task per job). Fine for a single Render
instance. If you ever run several instances, disable it (SCHEDULER_ENABLED=false)
and call the /admin/jobs/* endpoints from a cron instead."""
from __future__ import annotations

import asyncio
import logging

from ..core.cache import cache
from ..core.config import settings
from ..core.reference import load_reference_data
from ..core.database import acquire_conn
from .refresh import refresh_all
from .rollups import run_all_rollups

logger = logging.getLogger(__name__)
_tasks: list[asyncio.Task] = []


async def _loop(name: str, interval_s: float, fn, first_delay_s: float, clear_cache: bool) -> None:
    await asyncio.sleep(first_delay_s)
    while True:
        try:
            async with acquire_conn() as conn:
                result = await fn(conn)
            if clear_cache:
                await cache.clear()
            logger.info("job %s ok: %s", name, result)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("job %s failed", name)
        await asyncio.sleep(interval_s)


async def _reload(conn):
    return await load_reference_data(conn)


def start() -> None:
    if not settings.scheduler_enabled or _tasks:
        return
    loop = asyncio.get_running_loop()
    specs = [
        ("rollups", settings.rollup_interval_minutes * 60, run_all_rollups, 90, True),
        ("category_refresh", settings.category_refresh_interval_hours * 3600, refresh_all, 300, True),
        ("reference_reload", settings.fx_refresh_interval_minutes * 60, _reload, 600, False),
    ]
    for name, interval, fn, delay, clear in specs:
        _tasks.append(loop.create_task(_loop(name, interval, fn, delay, clear), name=f"job-{name}"))


async def stop() -> None:
    for t in _tasks:
        t.cancel()
    await asyncio.gather(*_tasks, return_exceptions=True)
    _tasks.clear()
