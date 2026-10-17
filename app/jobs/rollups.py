"""Nightly/hourly rollups. Idempotent: safe to run at any time, any number of
times. Request handlers never write the rollup tables."""
from __future__ import annotations

import logging

from ..core.config import settings

logger = logging.getLogger(__name__)


async def rollup_phone_views(conn) -> int:
    """Rebuild the last 3 days of phone_view_daily from events (older days are final)."""
    async with conn.transaction():
        await conn.execute("DELETE FROM phone_view_daily WHERE day >= (now() AT TIME ZONE 'utc')::date - 2")
        res = await conn.execute(
            """INSERT INTO phone_view_daily (phone_id, day, views, uniques)
SELECT e.phone_id, (e.created_at AT TIME ZONE 'utc')::date, count(*), count(DISTINCT e.session_key)
FROM events e JOIN phones p ON p.id = e.phone_id
WHERE e.type = 'page_view' AND e.page_type = 'phone' AND e.phone_id IS NOT NULL
  AND (e.created_at AT TIME ZONE 'utc')::date >= (now() AT TIME ZONE 'utc')::date - 2
GROUP BY 1, 2"""
        )
    return int(res.split()[-1])


async def rollup_compare_pairs(conn) -> int:
    async with conn.transaction():
        await conn.execute("DELETE FROM compare_pair_stats")
        res = await conn.execute(
            """INSERT INTO compare_pair_stats (phone_a, phone_b, uniq_7d, uniq_30d, updated_at)
SELECT phone_a, phone_b,
       count(DISTINCT session_key) FILTER (WHERE created_at > now() - interval '7 days'),
       count(DISTINCT session_key),
       now()
FROM compare_events WHERE created_at > now() - interval '30 days'
GROUP BY phone_a, phone_b"""
        )
    return int(res.split()[-1])


async def rollup_compare_sets(conn) -> int:
    async with conn.transaction():
        await conn.execute("DELETE FROM compare_set_stats")
        res = await conn.execute(
            """INSERT INTO compare_set_stats (set_key, set_size, phone_ids, uniq_7d, uniq_30d, updated_at)
SELECT set_key, max(set_size), string_to_array(set_key, '-')::int[],
       count(DISTINCT session_key) FILTER (WHERE created_at > now() - interval '7 days'),
       count(DISTINCT session_key),
       now()
FROM compare_events
WHERE set_key IS NOT NULL AND created_at > now() - interval '30 days'
GROUP BY set_key"""
        )
    return int(res.split()[-1])


async def rollup_popularity(conn) -> int:
    """trend_score = 0.6*overall + log-ish views + compares, so a new phone with
    real traffic climbs, and a phone with no traffic still sorts by quality."""
    async with conn.transaction():
        await conn.execute("DELETE FROM phone_popularity")
        res = await conn.execute(
            """INSERT INTO phone_popularity (phone_id, views_7d, compares_30d, trend_score, updated_at)
SELECT p.id,
       COALESCE(v.views, 0), COALESCE(c.n, 0),
       round((0.6 * COALESCE(p.overall_score, 0)
              + 1.2 * ln(1 + COALESCE(v.uniques, 0))
              + 1.8 * ln(1 + COALESCE(c.n, 0)))::numeric, 3),
       now()
FROM phones p
LEFT JOIN (SELECT phone_id, sum(views)::int AS views, sum(uniques)::int AS uniques FROM phone_view_daily
           WHERE day >= (now() AT TIME ZONE 'utc')::date - 6 GROUP BY phone_id) v ON v.phone_id = p.id
LEFT JOIN (SELECT pid, count(DISTINCT session_key)::int AS n FROM (
             SELECT phone_a AS pid, session_key FROM compare_events WHERE created_at > now() - interval '30 days'
             UNION ALL
             SELECT phone_b, session_key FROM compare_events WHERE created_at > now() - interval '30 days') x
           GROUP BY pid) c ON c.pid = p.id"""
        )
    return int(res.split()[-1])


async def purge_old_events(conn) -> int:
    res = await conn.execute(
        "DELETE FROM events WHERE created_at < now() - make_interval(days => $1)", settings.events_retention_days,
    )
    await conn.execute(
        "DELETE FROM compare_events WHERE created_at < now() - make_interval(days => $1)", settings.events_retention_days,
    )
    return int(res.split()[-1])


async def run_all_rollups(conn) -> dict:
    out = {
        "phone_view_daily": await rollup_phone_views(conn),
        "compare_pair_stats": await rollup_compare_pairs(conn),
        "compare_set_stats": await rollup_compare_sets(conn),
        "phone_popularity": await rollup_popularity(conn),
    }
    out["events_purged"] = await purge_old_events(conn)
    return out
