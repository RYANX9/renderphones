"""'Next steps': the links every page ends with so no page is a dead end.

Each step is {slot, kind, label, href, phone?}. `slot` is a stable id the
frontend sends back in `next_step_click` events, so the analytics can say which
exits people actually use. Max MAX_STEPS per page, de-duplicated by href, and
always chosen from data (never a static list), including for empty and
crowded pages.
"""
from __future__ import annotations

import asyncio
from typing import Any

import asyncpg

from ..core.category_config import CATEGORY_CONFIG, price_band_slug
from ..core.config import settings
from ..core.database import acquire_conn, row_to_dict
from ..core.shaping import shape_mini
from ..core.slugs import brand_path, category_path, compare_path, phone_path
from ..core.sql_fragments import LISTABLE, MINI_SELECT, NOT_REGIONAL
from . import phone_repo, stats_repo

MAX_STEPS = 6
BEST_HUB = "/best"


def step(slot: str, kind: str, label: str, href: str, phone: dict | None = None) -> dict:
    out = {"slot": slot, "kind": kind, "label": label, "href": href}
    if phone is not None:
        out["phone"] = phone
    return out


def _finish(candidates: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out = []
    for c in candidates:
        if c["href"] in seen:
            continue
        seen.add(c["href"])
        out.append(c)
        if len(out) == MAX_STEPS:
            break
    return out


def _other(pair: dict, phone_id: int) -> dict:
    return next(p for p in pair["phones"] if p["id"] != phone_id)


def _round_budget(price: float | None) -> int | None:
    if not price:
        return None
    step_ = 50 if price < 500 else 100
    return int(round(price / step_) * step_)


async def _brand_neighbor(conn: asyncpg.Connection, brand_slug: str, release_ts: int | None, newer: bool) -> dict | None:
    if release_ts is None:
        return None
    op, order = (">", "ASC") if newer else ("<", "DESC")
    row = await conn.fetchrow(
        f"SELECT {MINI_SELECT} FROM phone_catalog p WHERE p.brand_slug = $1 AND {LISTABLE} AND {NOT_REGIONAL} "
        f"AND p.release_ts {op} $2 AND p.release_ts <= EXTRACT(EPOCH FROM now()) "
        f"ORDER BY p.release_ts {order} LIMIT 1", brand_slug, release_ts,
    )
    return shape_mini(row_to_dict(row)) if row else None


# ── phone ────────────────────────────────────────────────────────────────────

async def _on(fn, *args):
    """Run one query on its own pooled connection so independent queries overlap."""
    async with acquire_conn() as conn:
        return await fn(conn, *args)


async def _categories(phone: dict) -> list[dict]:
    return phone.get("categories") or await _on(phone_repo.phone_categories, phone["id"])


async def for_phone(phone: dict, region: str | None = None) -> list[dict]:
    pid, full = phone["id"], phone["full_slug"]
    name = phone.get("display_name") or phone["name"]

    popular, riv, cats, newer = await asyncio.gather(
        _on(stats_repo.pairs_for_phone, pid, 2),
        _on(phone_repo.rivals, pid, 4, region),
        _categories(phone),
        _on(_brand_neighbor, phone["brand_slug"], phone.get("release_ts"), True),
    )

    pop_steps = [
        step("phone.most_compared", "compare", f"Compare with {_other(p, pid)['display_name']}",
             compare_path([full, _other(p, pid)["full_slug"]]), _other(p, pid))
        for p in popular
    ]
    popular_ids = {_other(p, pid)["id"] for p in popular}
    rival_steps = [
        step("phone.rival", "compare", f"Compare with {r['display_name']}", compare_path([full, r["full_slug"]]), r)
        for r in riv if r["id"] not in popular_ids
    ]

    cat_step = []
    if cats:
        c = cats[0]
        cat_step = [step("phone.category", "category", f"#{c['rank']} in {c['title']}", c["path"])]

    newer_step = [step("phone.newer_model", "phone", f"Newer: {newer['display_name']}", newer["path"], newer)] if newer else []
    if not newer_step:
        older = await _on(_brand_neighbor, phone["brand_slug"], phone.get("release_ts"), False)
        newer_step = [step("phone.older_model", "phone", f"Previous model: {older['display_name']}", older["path"], older)] if older else []

    budget = _round_budget(phone.get("price_usd"))
    pick_step = [step("phone.pick", "pick", f"Find a phone around ${budget}" if budget else f"Not sure about the {name}? Help me pick",
                      f"/pick?budget={budget}" if budget else "/pick")]
    trade_step = [step("phone.tradein", "tradein", f"Trade in your old phone for the {name}", f"/trade-in?phone={pid}")] \
        if phone.get("price_usd") else []

    order: list[dict] = []
    for group in (pop_steps[:1], rival_steps[:1], cat_step, newer_step, pick_step, trade_step,
                  pop_steps[1:2], rival_steps[1:2], rival_steps[2:]):
        order.extend(group)
    return _finish(order)


# ── category ─────────────────────────────────────────────────────────────────

async def for_category(conn: asyncpg.Connection, slug: str, ranked: list[dict]) -> list[dict]:
    steps: list[dict] = []
    if len(ranked) >= 2:
        a, b = ranked[0], ranked[1]
        steps.append(step("category.compare_top", "compare", f"Compare #1 {a['display_name']} vs #2 {b['display_name']}",
                          compare_path([a["full_slug"], b["full_slug"]]), a))

    overlap = [o for o in await phone_repo.category_overlap(conn, slug) if o["slug"] in CATEGORY_CONFIG]
    if len(ranked) < settings.category_min_phones_indexable and not overlap:
        counts = await phone_repo.category_counts(conn)
        overlap = [{"slug": s} for s, n in sorted(counts.items(), key=lambda kv: -kv[1]) if s != slug and s in CATEGORY_CONFIG]
    for o in overlap[:2]:
        steps.append(step("category.neighbor", "category", CATEGORY_CONFIG[o["slug"]]["title"], category_path(o["slug"])))

    if not slug.startswith("under-") and ranked and ranked[0].get("price_usd"):
        cap = price_band_slug(ranked[0]["price_usd"])
        if cap:
            steps.append(step("category.price_cap", "category", CATEGORY_CONFIG[cap]["title"], category_path(cap)))

    if ranked:
        top = ranked[0]
        steps.append(step("category.top_brand", "brand", f"More from {top['brand']}", brand_path(top["brand_slug"])))
    steps.append(step("category.pick", "pick", "Not sure? Help me pick", "/pick"))
    steps.append(step("category.hub", "hub", "All best-of lists", BEST_HUB))
    return _finish(steps)


# ── brand ────────────────────────────────────────────────────────────────────

async def for_brand(conn: asyncpg.Connection, brand: dict, all_brands: list[dict]) -> list[dict]:
    steps: list[dict] = []
    if brand.get("categories"):
        c = brand["categories"][0]
        steps.append(step("brand.top_category", "category", f"{brand['brand']} in {c['title']}", c["path"]))
    top = (brand.get("top_phones") or [None])[0]
    if top:
        steps.append(step("brand.top_phone", "phone", f"Top rated: {top['display_name']}", top["path"], top))

    avg = brand["price_range"].get("avg")
    others = [b for b in all_brands if b["brand_slug"] != brand["brand_slug"] and b.get("avg_price") and avg]
    others.sort(key=lambda b: abs(b["avg_price"] - avg))
    for b in others[:3]:
        steps.append(step("brand.rival_brand", "brand", f"{b['brand']} phones", b["path"]))

    steps.append(step("brand.pick", "pick", "Help me pick a phone", "/pick"))
    steps.append(step("brand.hub", "hub", "All best-of lists", BEST_HUB))
    return _finish(steps)


# ── compare ──────────────────────────────────────────────────────────────────

async def for_compare(conn: asyncpg.Connection, phones: list[dict], winner_id: int | None,
                      region: str | None = None) -> list[dict]:
    if not phones:
        return []
    ids = [p["id"] for p in phones]
    winner = next((p for p in phones if p["id"] == winner_id), None)
    scored = [p for p in phones if p.get("overall_score") is not None]
    loser = min(scored, key=lambda p: p["overall_score"]) if len(scored) >= 2 and winner else None

    steps: list[dict] = []
    if winner:
        steps.append(step("compare.winner", "phone", f"See the {winner['display_name']} in detail", winner["path"], winner))

    if loser and loser["id"] != (winner or {}).get("id"):
        riv = await phone_repo.rivals(conn, loser["id"], 6, region)
        better = [r for r in riv if r["id"] not in ids and (r.get("overall_score") or 0) > (loser.get("overall_score") or 0)]
        if better:
            alt = better[0]
            steps.append(step("compare.better_alternative", "phone", f"A stronger alternative to {loser['display_name']}: {alt['display_name']}",
                              alt["path"], alt))

    exclude = None
    if len(ids) == 2:
        exclude = (min(ids), max(ids))
    for p in await stats_repo.pairs_for_phones(conn, ids, exclude, 2):
        steps.append(step("compare.next_pair", "compare", f"{p['phones'][0]['display_name']} vs {p['phones'][1]['display_name']}", p["path"]))

    if winner:
        cats = await phone_repo.phone_categories(conn, winner["id"])
        if cats:
            steps.append(step("compare.category", "category", f"{winner['display_name']} is #{cats[0]['rank']} in {cats[0]['title']}", cats[0]["path"]))
        riv_w = await phone_repo.rivals(conn, winner["id"], 4, region)
        for r in riv_w:
            if r["id"] not in ids:
                steps.append(step("compare.rival_of_winner", "compare", f"{winner['display_name']} vs {r['display_name']}",
                                  compare_path([winner["full_slug"], r["full_slug"]]), r))
                break
    steps.append(step("compare.pick", "pick", "Help me pick the right one", "/pick"))
    return _finish(steps)


# ── home / generic ───────────────────────────────────────────────────────────

def home_steps() -> list[dict]:
    return [
        step("home.pick", "pick", "Help me pick a phone", "/pick"),
        step("home.compare", "compare", "Compare two phones", "/compare"),
        step("home.hub", "hub", "Browse best-of lists", BEST_HUB),
        step("home.brands", "hub", "All brands", "/brands"),
    ]
