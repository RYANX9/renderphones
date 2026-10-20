from __future__ import annotations

from typing import Any

import asyncpg

from ..core.config import settings
from ..core.slugs import MAX_COMPARE, compare_path, pair_slug, sorted_full_slugs, split_pair_slug
from . import next_steps, phone_repo, stats_repo

AXES = ("camera", "performance", "battery", "display", "build", "value", "durability")
AXIS_LABELS = {
    "camera": "camera", "performance": "performance", "battery": "battery", "display": "display",
    "build": "build", "value": "value", "durability": "durability",
}


def compare_winner(phones: list[dict[str, Any]]) -> int | None:
    """One rule used everywhere: highest overall_score wins; a tie goes to the
    lower price, then to the lower id. Depends only on the set of phones, never
    on the order. Phones without a score cannot win."""
    scored = [p for p in phones if p.get("overall_score") is not None]
    if len(phones) < 2 or not scored:
        return None

    def rank(p: dict[str, Any]) -> tuple[float, float, int]:
        price = p.get("price_usd")
        return (-float(p["overall_score"]), float(price) if price else float("inf"), p["id"])

    return min(scored, key=rank)["id"]


def axis_winners(phones: list[dict[str, Any]]) -> dict[str, int | None]:
    """Winner per score axis; None when tied or fewer than two phones have a score."""
    out: dict[str, int | None] = {}
    for axis in AXES:
        vals = [(float(p[f"{axis}_score"]), p["id"]) for p in phones if p.get(f"{axis}_score") is not None]
        if len(vals) < 2:
            out[axis] = None
            continue
        best = max(v for v, _ in vals)
        leaders = [i for v, i in vals if v == best]
        out[axis] = leaders[0] if len(leaders) == 1 else None
    return out


def win_summary(phones: list[dict[str, Any]], winners: dict[str, int | None]) -> dict[str, Any]:
    """'X wins camera and battery, Y wins value' as data plus a ready sentence."""
    wins: dict[int, list[str]] = {}
    for axis, pid in winners.items():
        if pid is not None:
            wins.setdefault(pid, []).append(AXIS_LABELS[axis])
    names = {p["id"]: p.get("display_name") or p["name"] for p in phones}
    parts = []
    items = []
    for pid, axes in sorted(wins.items(), key=lambda kv: -len(kv[1])):
        items.append({"phone_id": pid, "display_name": names[pid], "wins": axes})
        joined = axes[0] if len(axes) == 1 else ", ".join(axes[:-1]) + " and " + axes[-1]
        parts.append(f"{names[pid]} wins {joined}")
    return {"items": items, "text": "; ".join(parts) + "." if parts else None}


def verdict_key(phones: list[dict]) -> str:
    return "compare:v3:" + ",".join(str(i) for i in sorted(p["id"] for p in phones))


async def build_payload(conn: asyncpg.Connection, phones: list[dict], *, region: str | None = None,
                        include_next_steps: bool = True) -> dict[str, Any]:
    winner_id = compare_winner(phones)
    winners = axis_winners(phones)
    verdict = None
    if len(phones) >= 2:
        verdict = await phone_repo.get_cached_ai_copy(conn, verdict_key(phones), 14)
        if verdict and verdict.get("winner_id") != winner_id:
            verdict = None
    slugs = [p["full_slug"] for p in phones]
    ids = [p["id"] for p in phones]
    indexable = False
    if len(ids) == 2:
        indexable = await stats_repo.pair_is_indexable(conn, ids[0], ids[1], settings.compare_popular_min_sessions)
    payload: dict[str, Any] = {
        "phones": phones,
        "winner_id": winner_id,
        "axis_winners": winners,
        "summary": win_summary(phones, winners),
        "verdict": verdict,
        "canonical": {"slug": pair_slug(slugs), "path": compare_path(slugs)},
        # Only curated / popular two-phone pairs are indexable; everything else
        # should render with <meta name="robots" content="noindex">.
        "indexable": indexable,
    }
    if include_next_steps:
        payload["next_steps"] = await next_steps.for_compare(conn, phones, winner_id, region)
    return payload


async def load_pair(conn: asyncpg.Connection, pair: str) -> tuple[list[dict], bool]:
    """Exact full_slug match only (no fuzzy resolving). Returns (phones, needs_redirect):
    needs_redirect is True when the URL is not the canonical sorted form."""
    requested = split_pair_slug(pair)
    if not 1 <= len(requested) <= MAX_COMPARE:
        return [], False
    canonical = sorted_full_slugs(requested)
    phones = await phone_repo.compare(conn, full_slugs=canonical)
    if len(phones) != len(canonical):
        return [], False
    needs_redirect = pair != "-vs-".join(canonical)
    return phones, needs_redirect
