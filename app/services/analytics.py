"""First-party analytics: no IPs, no user ids, no raw client ids stored.

session_key = sha256(salt | UTC day | client token | user agent family), so a
visitor is countable within one day and unlinkable across days. The client
token is a random id the frontend keeps in memory/sessionStorage; if it is
missing we fall back to the IP, which is hashed and never stored.
"""
from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import asyncpg
from fastapi import Request

from ..core.config import settings

logger = logging.getLogger(__name__)

EVENT_TYPES = frozenset({
    "page_view", "next_step_click", "block_click", "compare_add", "pick_step",
    "pick_complete", "offer_click", "search", "filter",
})
PAGE_TYPES = frozenset({
    "home", "phone", "compare", "category", "brand", "brands", "pick",
    "best_hub", "tradein", "methodology", "other",
})
COMPARE_ENTRIES = frozenset({"search", "category", "phone_page", "home", "pick", "tray", "direct", "next_step"})

_SEARCH_HOSTS = ("google.", "bing.", "duckduckgo.", "yahoo.", "yandex.", "baidu.", "ecosia.", "brave.")
_SOCIAL_HOSTS = ("facebook.", "instagram.", "twitter.", "x.com", "t.co", "reddit.", "youtube.", "tiktok.",
                 "linkedin.", "pinterest.", "telegram.", "whatsapp.", "t.me")
_MOBILE_RE = re.compile(r"android|iphone|ipod|mobile", re.I)
_TABLET_RE = re.compile(r"ipad|tablet", re.I)


def _clip(value: Any, n: int) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s[:n] if s else None


def classify_source(referrer: str | None, own_host: str | None = None) -> str:
    if not referrer:
        return "direct"
    host = (urlparse(referrer).hostname or "").lower()
    if not host:
        return "direct"
    if own_host and host.endswith(own_host.lower()):
        return "internal"
    if any(h in host for h in _SEARCH_HOSTS):
        return "search"
    if any(h in host for h in _SOCIAL_HOSTS):
        return "social"
    return "referral"


def classify_device(user_agent: str | None) -> str:
    ua = user_agent or ""
    if _TABLET_RE.search(ua):
        return "tablet"
    return "mobile" if _MOBILE_RE.search(ua) else "desktop"


def session_key(request: Request, client_id: str | None) -> str:
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    fwd = request.headers.get("X-Forwarded-For")
    ip = fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "")
    token = client_id or ip
    ua = (request.headers.get("user-agent") or "")[:80]
    raw = f"{settings.analytics_salt}|{day}|{token}|{ua}".encode()
    return hashlib.sha256(raw).hexdigest()[:32]


async def record_events(conn: asyncpg.Connection, request: Request, *, client_id: str | None,
                        events: list[dict[str, Any]], region: str | None, referrer: str | None) -> int:
    if not settings.events_enabled:
        return 0
    key = session_key(request, client_id)
    device = classify_device(request.headers.get("user-agent"))
    region = _clip((region or "").upper(), 2)
    rows = []
    for e in events[:25]:
        etype = e.get("type")
        if etype not in EVENT_TYPES:
            continue
        page_type = e.get("page_type") if e.get("page_type") in PAGE_TYPES else "other"
        phone_id = e.get("phone_id") if isinstance(e.get("phone_id"), int) else None
        value = e.get("value") if isinstance(e.get("value"), int) else None
        source = classify_source(e.get("referrer") or referrer, request.headers.get("host"))
        rows.append((key, etype, page_type, _clip(e.get("path"), 300), phone_id, _clip(e.get("target"), 300),
                     _clip(e.get("slot"), 80), value, source if etype == "page_view" else None, region, device))
    if not rows:
        return 0
    # phone_id is not a foreign key on purpose: events must never fail because
    # a phone was deleted after the page was rendered.
    await conn.executemany(
        "INSERT INTO events (session_key, type, page_type, path, phone_id, target, slot, value, source, region, device) "
        "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)", rows,
    )
    return len(rows)

async def record_compare(conn: asyncpg.Connection, request: Request, *, client_id: str | None,
                         phone_ids: list[int], entry: str | None) -> int:
    if not settings.events_enabled:
        return 0
    ids = sorted(set(i for i in phone_ids if isinstance(i, int)))[:4]
    if len(ids) < 2:
        return 0
    valid = {r["id"] for r in await conn.fetch("SELECT id FROM phones WHERE id = ANY($1::int[])", ids)}
    ids = [i for i in ids if i in valid]
    pairs = [(a, b) for n, a in enumerate(ids) for b in ids[n + 1:]]
    if not pairs:
        return 0
    key = session_key(request, client_id)
    entry = entry if entry in COMPARE_ENTRIES else "direct"
    set_key = "-".join(str(i) for i in ids)
    inserted = 0
    for a, b in pairs:
        exists = await conn.fetchval(
            "SELECT 1 FROM compare_events WHERE phone_a = $1 AND phone_b = $2 AND session_key = $3 "
            "AND created_at > now() - interval '1 hour' LIMIT 1", a, b, key,
        )
        if exists:
            continue
        await conn.execute(
            "INSERT INTO compare_events (phone_a, phone_b, session_key, entry, set_key, set_size) "
            "VALUES ($1,$2,$3,$4,$5,$6)", a, b, key, entry, set_key, len(ids),
        )
        inserted += 1
    return inserted
