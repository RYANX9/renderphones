"""
Admin routes for the v2 schema. Auth: header `X-Admin-Key` == settings.admin_api_key.

Price model recap (so edits here do the right thing):
  * the price shown everywhere = lowest available offer across the phone's
    variants, else phones.price_usd (the catalog price)  [view phone_prices]
  * price_history only feeds the chart: writing to it never changes a price

Endpoints
  GET    /admin/phones                         list/search/sort with price_basis + counts
  PUT    /admin/phones/{id}/price              set the catalog price (phones.price_usd)
  GET    /admin/phones/{id}/variants           variants with their lowest offer
  POST   /admin/phones/{id}/variants           create variant
  PUT    /admin/variants/{id}                  edit ram/storage/is_default
  DELETE /admin/variants/{id}                  delete variant (+ its offers and history)
  GET    /admin/phones/{id}/offers             all offers for a phone (scope-annotated)
  POST   /admin/variants/{id}/offers           create offer
  PUT    /admin/offers/{id}                    edit offer
  DELETE /admin/offers/{id}                    delete offer
  POST   /admin/offers/check                   broken-link sweep (updates offers.status)
  POST   /admin/variants/{id}/price-history    upsert one chart point
  GET    /admin/retailers                      retailers table, with offer counts
  GET    /admin/phones/{id}/region-preview     which offers make a phone visible in a region
  GET    /admin/stats/offers                   real / placeholder / missing counts
  GET    /admin/analytics/summary              slots, searches, pick funnel, top phones/pairs
  POST   /admin/jobs/refresh-derived           recompute category rankings
  POST   /admin/jobs/rollups                   run analytics rollups now
  POST   /admin/jobs/reload-reference          reload retailers, feature keys, FX
  POST   /admin/jobs/clear-ai-cache            clear cached AI copy
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import anyio
import asyncpg
import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel

from ..core import market
from ..core.cache import cache
from ..core.config import settings
from ..core.database import acquire_conn, row_to_dict, rows_to_list
from ..core.reference import load_reference_data
from ..jobs.refresh import refresh_all
from ..jobs.rollups import run_all_rollups

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])


def require_admin_key(x_admin_key: Optional[str] = Header(None)) -> None:
    if not settings.admin_api_key:
        raise HTTPException(status_code=503, detail="ADMIN_API_KEY not configured on server.")
    if not x_admin_key or x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing admin key.")


AUTH = [Depends(require_admin_key)]

_VALID_STATUSES = {"unchecked", "ok", "broken", "region_locked"}
_VALID_PRICE_SOURCES = {"retailer", "manual", "catalog"}

_SORTS = {
    "rank": "p.overall_score DESC NULLS LAST, p.id",
    "overall": "p.overall_score DESC NULLS LAST, p.id",
    "price": "p.price_usd DESC NULLS LAST, p.id",
    "newest": "p.release_ts DESC NULLS LAST, p.id",
    "name": "lower(p.display_name), p.id",
    "updated": "p.updated_at DESC NULLS LAST, p.id",
    "placeholder_first": "(p.price_basis = 'placeholder') DESC, (p.price_basis = 'catalog') DESC, p.overall_score DESC NULLS LAST",
    "catalog_first": "(p.price_basis = 'catalog') DESC, p.overall_score DESC NULLS LAST, p.id",
}


# ── phones ───────────────────────────────────────────────────────────────────

@router.get("/phones", dependencies=AUTH)
async def list_phones_for_admin(
    q: Optional[str] = Query(None), status: Optional[str] = Query(None),
    price_basis: Optional[str] = Query(None, description="offer | placeholder | catalog"),
    tier: Optional[str] = Query(None), sort: str = Query("rank"),
    page: int = Query(1, ge=1), page_size: int = Query(30, ge=1, le=100),
):
    conds, params = ["TRUE"], []

    def add(v) -> str:
        params.append(v)
        return f"${len(params)}"

    if q and q.strip():
        like = add(f"%{q.strip().lower()}%")
        conds.append(f"(lower(p.display_name) LIKE {like} OR p.search_doc LIKE {like})")
    if status:
        conds.append(f"p.status = {add(status)}")
    if price_basis in ("offer", "placeholder", "catalog"):
        conds.append(f"p.price_basis = {add(price_basis)}")
    if tier:
        conds.append(f"p.tier = {add(tier)}")
    where = " AND ".join(conds)
    order = _SORTS.get(sort, _SORTS["rank"])
    n = len(params)
    async with acquire_conn() as conn:
        rows = await conn.fetch(
            f"""SELECT p.id, p.brand, p.name, p.display_name, p.brand_slug, p.slug, p.full_slug, p.status,
p.main_image_url, p.release_year, p.tier, p.overall_score, p.camera_score, p.performance_score,
p.battery_score, p.display_score, p.build_score, p.value_score, p.durability_score,
p.price_usd, p.price_basis, p.price_region, p.catalog_price_usd, p.is_regional_edition, p.updated_at,
p.ram_options, p.storage_options,
(SELECT count(*)::int FROM variants v WHERE v.phone_id = p.id) AS variant_count,
(SELECT count(*)::int FROM offers o JOIN variants v ON v.id = o.variant_id WHERE v.phone_id = p.id) AS offer_count,
(SELECT count(*)::int FROM offers o JOIN variants v ON v.id = o.variant_id WHERE v.phone_id = p.id AND o.is_placeholder) AS placeholder_count,
(SELECT count(*)::int FROM price_history h JOIN variants v ON v.id = h.variant_id WHERE v.phone_id = p.id) AS history_points,
COUNT(*) OVER() AS _total
FROM phone_catalog p WHERE {where} ORDER BY {order} LIMIT ${n + 1} OFFSET ${n + 2}""",
            *params, page_size, (page - 1) * page_size,
        )
    results = rows_to_list(rows)
    total = results[0]["_total"] if results else 0
    for r in results:
        r.pop("_total", None)
    return {"total": total, "page": page, "page_size": page_size, "results": results}


class PriceUpdate(BaseModel):
    price_usd: Optional[float] = None


@router.put("/phones/{phone_id}/price", dependencies=AUTH)
async def update_catalog_price(phone_id: int, payload: PriceUpdate):
    """Sets phones.price_usd, the catalog fallback used only when a phone has no
    available offer. Never touches price_history."""
    if payload.price_usd is not None and payload.price_usd <= 0:
        raise HTTPException(status_code=422, detail="price_usd must be positive.")
    async with acquire_conn() as conn:
        row = await conn.fetchrow(
            "UPDATE phones SET price_usd = $1, price_updated_at = now() WHERE id = $2 RETURNING id, price_usd",
            payload.price_usd, phone_id,
        )
        if row is None:
            raise HTTPException(status_code=404, detail=f"Phone {phone_id} not found.")
        effective = await conn.fetchrow("SELECT price_usd, price_basis FROM phone_prices WHERE phone_id = $1", phone_id)
    await cache.clear()
    return {**row_to_dict(row), "effective_price_usd": effective["price_usd"], "price_basis": effective["price_basis"]}


# ── variants ─────────────────────────────────────────────────────────────────

@router.get("/phones/{phone_id}/variants", dependencies=AUTH)
async def get_phone_variants(phone_id: int):
    async with acquire_conn() as conn:
        rows = await conn.fetch(
            """SELECT v.id, v.phone_id, v.ram_gb, v.storage_gb, v.is_default, v.updated_at,
o.usd AS lowest_usd, o.url AS lowest_url, o.retailer_key AS lowest_retailer, o.is_placeholder AS lowest_is_placeholder,
(SELECT count(*)::int FROM offers x WHERE x.variant_id = v.id) AS offer_count
FROM variants v
LEFT JOIN LATERAL (SELECT f.usd, f.url, f.retailer_key, f.is_placeholder FROM offers f
  WHERE f.variant_id = v.id AND f.is_available AND f.usd IS NOT NULL AND f.usd > 0
  ORDER BY f.usd, f.is_placeholder, f.id LIMIT 1) o ON true
WHERE v.phone_id = $1 ORDER BY v.storage_gb NULLS FIRST, v.ram_gb NULLS FIRST, v.id""", phone_id,
        )
    return {"phone_id": phone_id, "variants": rows_to_list(rows)}


class VariantCreate(BaseModel):
    storage_gb: Optional[int] = None
    ram_gb: Optional[int] = None
    is_default: bool = False


class VariantUpdate(BaseModel):
    storage_gb: Optional[int] = None
    ram_gb: Optional[int] = None
    is_default: Optional[bool] = None


@router.post("/phones/{phone_id}/variants", status_code=201, dependencies=AUTH)
async def create_variant(phone_id: int, payload: VariantCreate):
    if payload.storage_gb is None and payload.ram_gb is None:
        raise HTTPException(status_code=422, detail="Provide storage_gb and/or ram_gb.")
    async with acquire_conn() as conn:
        if not await conn.fetchval("SELECT 1 FROM phones WHERE id = $1", phone_id):
            raise HTTPException(status_code=404, detail=f"Phone {phone_id} not found.")
        async with conn.transaction():
            if payload.is_default:
                await conn.execute("UPDATE variants SET is_default = FALSE WHERE phone_id = $1", phone_id)
            row = await conn.fetchrow(
                "INSERT INTO variants (phone_id, ram_gb, storage_gb, is_default) VALUES ($1,$2,$3,$4) RETURNING *",
                phone_id, payload.ram_gb, payload.storage_gb, payload.is_default,
            )
    await cache.clear()
    return row_to_dict(row)


@router.put("/variants/{variant_id}", dependencies=AUTH)
async def update_variant(variant_id: int, payload: VariantUpdate):
    fields = payload.model_fields_set
    if not fields:
        raise HTTPException(status_code=400, detail="Nothing to update.")
    async with acquire_conn() as conn:
        async with conn.transaction():
            phone_id = await conn.fetchval("SELECT phone_id FROM variants WHERE id = $1", variant_id)
            if phone_id is None:
                raise HTTPException(status_code=404, detail=f"Variant {variant_id} not found.")
            if payload.is_default:
                await conn.execute("UPDATE variants SET is_default = FALSE WHERE phone_id = $1", phone_id)
            sets, vals = [], []
            for k in ("ram_gb", "storage_gb", "is_default"):
                if k in fields:
                    vals.append(getattr(payload, k))
                    sets.append(f"{k} = ${len(vals) + 1}")
            row = await conn.fetchrow(
                f"UPDATE variants SET {', '.join(sets)}, updated_at = now() WHERE id = $1 RETURNING *", variant_id, *vals,
            )
    await cache.clear()
    return row_to_dict(row)


@router.delete("/variants/{variant_id}", dependencies=AUTH)
async def delete_variant(variant_id: int):
    """Removes the variant together with its offers and chart history, in one transaction."""
    async with acquire_conn() as conn:
        async with conn.transaction():
            phone_id = await conn.fetchval("SELECT phone_id FROM variants WHERE id = $1", variant_id)
            if phone_id is None:
                raise HTTPException(status_code=404, detail=f"Variant {variant_id} not found.")
            await conn.execute("DELETE FROM offers WHERE variant_id = $1", variant_id)
            await conn.execute("DELETE FROM price_history WHERE variant_id = $1", variant_id)
            await conn.execute("DELETE FROM variants WHERE id = $1", variant_id)
    await cache.clear()
    return {"deleted": True, "id": variant_id, "phone_id": phone_id}


# ── offers ───────────────────────────────────────────────────────────────────

class OfferCreate(BaseModel):
    retailer_key: str
    url: str
    usd: Optional[float] = None
    region: Optional[str] = None
    is_available: bool = True
    is_placeholder: bool = False
    price_source: str = "retailer"
    status: str = "unchecked"


class OfferUpdate(BaseModel):
    retailer_key: Optional[str] = None
    url: Optional[str] = None
    usd: Optional[float] = None
    region: Optional[str] = None
    is_available: Optional[bool] = None
    is_placeholder: Optional[bool] = None
    price_source: Optional[str] = None
    status: Optional[str] = None


def _retailer(key: str) -> str:
    k = key.strip().lower()
    if k not in market.RETAILERS:
        raise HTTPException(status_code=422, detail=f"retailer_key must be one of {sorted(market.RETAILERS)}.")
    return k


def _check_offer_fields(status: Optional[str], source: Optional[str], usd: Optional[float]) -> None:
    if status is not None and status not in _VALID_STATUSES:
        raise HTTPException(status_code=422, detail=f"status must be one of {sorted(_VALID_STATUSES)}.")
    if source is not None and source not in _VALID_PRICE_SOURCES:
        raise HTTPException(status_code=422, detail=f"price_source must be one of {sorted(_VALID_PRICE_SOURCES)}.")
    if usd is not None and usd <= 0:
        raise HTTPException(status_code=422, detail="usd must be positive.")


def _with_scope(row: dict) -> dict:
    row["scope"] = market.retailer_scope(row["retailer_key"])
    return row


@router.get("/phones/{phone_id}/offers", dependencies=AUTH)
async def list_offers(phone_id: int):
    async with acquire_conn() as conn:
        if not await conn.fetchval("SELECT 1 FROM phones WHERE id = $1", phone_id):
            raise HTTPException(status_code=404, detail=f"Phone {phone_id} not found.")
        rows = await conn.fetch(
            """SELECT o.id, o.variant_id, v.ram_gb, v.storage_gb, o.retailer_key, o.region, o.usd, o.url,
o.is_available, o.is_placeholder, o.price_source, o.status, o.checked_at, o.updated_at
FROM offers o JOIN variants v ON v.id = o.variant_id WHERE v.phone_id = $1
ORDER BY v.storage_gb NULLS FIRST, v.ram_gb NULLS FIRST, o.usd NULLS LAST, o.id""", phone_id,
        )
    return {"phone_id": phone_id, "offers": [_with_scope(r) for r in rows_to_list(rows)]}


@router.post("/variants/{variant_id}/offers", status_code=201, dependencies=AUTH)
async def create_offer(variant_id: int, payload: OfferCreate):
    retailer = _retailer(payload.retailer_key)
    _check_offer_fields(payload.status, payload.price_source, payload.usd)
    url = payload.url.strip()
    if not url:
        raise HTTPException(status_code=422, detail="url is required.")
    region = (payload.region or market.retailer_info(retailer).get("region") or "").strip().upper() or None
    async with acquire_conn() as conn:
        if not await conn.fetchval("SELECT 1 FROM variants WHERE id = $1", variant_id):
            raise HTTPException(status_code=404, detail=f"Variant {variant_id} not found.")
        try:
            row = await conn.fetchrow(
                """INSERT INTO offers (variant_id, retailer_key, usd, url, is_available, status, region, is_placeholder, price_source,
checked_at) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9, CASE WHEN $6 = 'unchecked' THEN NULL ELSE now() END) RETURNING *""",
                variant_id, retailer, payload.usd, url, payload.is_available, payload.status, region,
                payload.is_placeholder, payload.price_source,
            )
        except asyncpg.UniqueViolationError:
            raise HTTPException(status_code=409, detail="An offer for this variant + retailer already exists. Update it instead.")
    await cache.clear()
    return _with_scope(row_to_dict(row))


@router.put("/offers/{offer_id}", dependencies=AUTH)
async def update_offer(offer_id: int, payload: OfferUpdate):
    fields = payload.model_fields_set
    if not fields:
        raise HTTPException(status_code=400, detail="Nothing to update.")
    _check_offer_fields(payload.status, payload.price_source, payload.usd)
    updates: dict = {}
    for k in ("usd", "is_available", "is_placeholder", "price_source", "status", "region"):
        if k in fields:
            updates[k] = getattr(payload, k)
    if "retailer_key" in fields and payload.retailer_key is not None:
        updates["retailer_key"] = _retailer(payload.retailer_key)
    if "url" in fields:
        if not (payload.url or "").strip():
            raise HTTPException(status_code=422, detail="url cannot be empty.")
        updates["url"] = payload.url.strip()
    if "region" in updates and updates["region"]:
        updates["region"] = updates["region"].strip().upper()
    if "status" in updates:
        updates["checked_at"] = None  # replaced below with now()
    sets, vals = [], []
    for k, v in updates.items():
        if k == "checked_at":
            sets.append("checked_at = now()")
            continue
        vals.append(v)
        sets.append(f"{k} = ${len(vals) + 1}")
    async with acquire_conn() as conn:
        try:
            row = await conn.fetchrow(
                f"UPDATE offers SET {', '.join(sets)}, updated_at = now() WHERE id = $1 RETURNING *", offer_id, *vals,
            )
        except asyncpg.UniqueViolationError:
            raise HTTPException(status_code=409, detail="Another offer already exists for this variant + retailer.")
    if row is None:
        raise HTTPException(status_code=404, detail=f"Offer {offer_id} not found.")
    await cache.clear()
    return _with_scope(row_to_dict(row))


@router.delete("/offers/{offer_id}", dependencies=AUTH)
async def delete_offer(offer_id: int):
    async with acquire_conn() as conn:
        row = await conn.fetchrow(
            "DELETE FROM offers WHERE id = $1 RETURNING id, variant_id", offer_id,
        )
    if row is None:
        raise HTTPException(status_code=404, detail=f"Offer {offer_id} not found.")
    await cache.clear()
    return {"deleted": True, **row_to_dict(row)}


class OfferCheckRequest(BaseModel):
    ids: Optional[list[int]] = None
    limit: int = 200


_CHECK_TIMEOUT_S = 6.0
_CHECK_CONCURRENCY = 10


async def _check_one(client: httpx.AsyncClient, item_id: int, url: str) -> dict:
    try:
        resp = await client.head(url, timeout=_CHECK_TIMEOUT_S, follow_redirects=True)
        if resp.status_code >= 400:
            resp = await client.get(url, timeout=_CHECK_TIMEOUT_S, follow_redirects=True)
        return {"id": item_id, "url": url, "status_code": resp.status_code, "ok": resp.status_code < 400}
    except Exception as exc:
        return {"id": item_id, "url": url, "status_code": None, "ok": False, "error": str(exc)[:200]}


@router.post("/offers/check", dependencies=AUTH)
async def check_offers(payload: OfferCheckRequest):
    async with acquire_conn() as conn:
        if payload.ids:
            rows = await conn.fetch("SELECT id, url FROM offers WHERE id = ANY($1::bigint[])", payload.ids)
        else:
            rows = await conn.fetch("SELECT id, url FROM offers WHERE NOT is_placeholder ORDER BY checked_at NULLS FIRST, id LIMIT $1", payload.limit)
    targets = [(r["id"], r["url"]) for r in rows]
    if not targets:
        return {"checked": 0, "broken": [], "ok": []}

    sem = anyio.Semaphore(_CHECK_CONCURRENCY)
    results: list[dict] = []

    async def bounded(client, item_id, url):
        async with sem:
            results.append(await _check_one(client, item_id, url))

    async with httpx.AsyncClient(headers={"User-Agent": "Mozilla/5.0 (SpecmobLinkChecker)"}) as client:
        async with anyio.create_task_group() as tg:
            for item_id, url in targets:
                tg.start_soon(bounded, client, item_id, url)

    async with acquire_conn() as conn:
        await conn.executemany(
            "UPDATE offers SET status = $1, checked_at = now() WHERE id = $2",
            [("ok" if r["ok"] else "broken", r["id"]) for r in results],
        )
    await cache.clear()
    return {"checked": len(results), "broken": [r for r in results if not r["ok"]], "ok": [r for r in results if r["ok"]]}


# ── price history (chart only) ───────────────────────────────────────────────

class HistoryPoint(BaseModel):
    on_date: str
    usd: float


@router.post("/variants/{variant_id}/price-history", dependencies=AUTH)
async def upsert_history_point(variant_id: int, payload: HistoryPoint):
    """Adds or replaces one chart point. The chart is the ONLY consumer."""
    try:
        d = date.fromisoformat(payload.on_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="on_date must be YYYY-MM-DD.")
    if payload.usd <= 0:
        raise HTTPException(status_code=422, detail="usd must be positive.")
    async with acquire_conn() as conn:
        if not await conn.fetchval("SELECT 1 FROM variants WHERE id = $1", variant_id):
            raise HTTPException(status_code=404, detail=f"Variant {variant_id} not found.")
        res = await conn.execute("UPDATE price_history SET usd = $3 WHERE variant_id = $1 AND on_date = $2", variant_id, d, payload.usd)
        if res == "UPDATE 0":
            await conn.execute("INSERT INTO price_history (variant_id, on_date, usd) VALUES ($1,$2,$3)", variant_id, d, payload.usd)
    await cache.clear()
    return {"variant_id": variant_id, "on_date": str(d), "usd": payload.usd}


# ── retailers / region / stats ───────────────────────────────────────────────

@router.get("/retailers", dependencies=AUTH)
async def list_retailers():
    async with acquire_conn() as conn:
        rows = await conn.fetch(
            """SELECT r.key, r.name, r.scope, r.region, r.domain, count(o.id)::int AS offer_count
FROM retailers r LEFT JOIN offers o ON o.retailer_key = r.key GROUP BY r.key ORDER BY r.scope, r.key""")
    return {"retailers": rows_to_list(rows)}


@router.get("/phones/{phone_id}/region-preview", dependencies=AUTH)
async def preview_region(phone_id: int, region: str = Query(..., min_length=2, max_length=2)):
    async with acquire_conn() as conn:
        if not await conn.fetchval("SELECT 1 FROM phones WHERE id = $1", phone_id):
            raise HTTPException(status_code=404, detail=f"Phone {phone_id} not found.")
        rows = await conn.fetch(
            """SELECT o.id, o.retailer_key, o.region, o.is_available FROM offers o
JOIN variants v ON v.id = o.variant_id WHERE v.phone_id = $1""", phone_id,
        )
    offers = rows_to_list(rows)
    return {
        "phone_id": phone_id, "region": region.upper(),
        "visible": market.phone_allowed_for_region(offers, region),
        "offers": [{**o, "scope": market.retailer_scope(o["retailer_key"])} for o in offers],
    }


@router.get("/stats/offers", dependencies=AUTH)
async def offer_stats():
    async with acquire_conn() as conn:
        row = await conn.fetchrow(
            """SELECT count(*) FILTER (WHERE price_basis = 'offer')::int AS phones_with_real_offer,
count(*) FILTER (WHERE price_basis = 'placeholder')::int AS phones_placeholder_only,
count(*) FILTER (WHERE price_basis = 'catalog')::int AS phones_catalog_only,
count(*) FILTER (WHERE price_usd IS NULL)::int AS phones_without_price
FROM phone_prices""")
        offers = await conn.fetchrow(
            """SELECT count(*)::int AS total, count(*) FILTER (WHERE is_placeholder)::int AS placeholder,
count(*) FILTER (WHERE status = 'broken')::int AS broken, count(*) FILTER (WHERE status = 'unchecked')::int AS unchecked
FROM offers""")
    return {"phones": row_to_dict(row), "offers": row_to_dict(offers)}


# ── analytics ────────────────────────────────────────────────────────────────

@router.get("/analytics/summary", dependencies=AUTH)
async def analytics_summary(days: int = Query(30, ge=1, le=90)):
    """What the first-party events answer: which exits are used, which searches
    find nothing, where /pick drops off, what is viewed and compared."""
    async with acquire_conn() as conn:
        async def q(sql: str, *args):
            return rows_to_list(await conn.fetch(sql, *args))

        return {
            "days": days,
            "page_views_by_type": await q(
                """SELECT page_type, count(*)::int AS views, count(DISTINCT session_key)::int AS sessions
FROM events WHERE type = 'page_view' AND created_at > now() - make_interval(days => $1) GROUP BY 1 ORDER BY 2 DESC""", days),
            "next_step_slots": await q(
                """SELECT slot, count(*)::int AS clicks, count(DISTINCT session_key)::int AS sessions
FROM events WHERE type IN ('next_step_click','block_click') AND slot IS NOT NULL
AND created_at > now() - make_interval(days => $1) GROUP BY 1 ORDER BY 2 DESC LIMIT 40""", days),
            "dead_end_pages": await q(
                """SELECT v.page_type, count(*)::int AS views,
count(*) FILTER (WHERE NOT EXISTS (SELECT 1 FROM events c WHERE c.session_key = v.session_key AND c.created_at > v.created_at
  AND c.created_at < v.created_at + interval '30 minutes' AND c.type IN ('next_step_click','block_click','compare_add','offer_click')))::int AS no_exit
FROM events v WHERE v.type = 'page_view' AND v.created_at > now() - make_interval(days => $1) GROUP BY 1 ORDER BY no_exit DESC""", days),
            "searches_without_results": await q(
                """SELECT lower(target) AS query, count(*)::int AS times FROM events
WHERE type = 'search' AND value = 0 AND target IS NOT NULL AND created_at > now() - make_interval(days => $1)
GROUP BY 1 ORDER BY 2 DESC LIMIT 30""", days),
            "pick_funnel": await q(
                """SELECT CASE WHEN type = 'pick_complete' THEN 99 ELSE value END AS step, type, count(DISTINCT session_key)::int AS sessions
FROM events WHERE type IN ('pick_step','pick_complete') AND created_at > now() - make_interval(days => $1)
GROUP BY 1, 2 ORDER BY 1""", days),
            "top_phones": await q(
                """SELECT p.display_name, p.full_slug, count(*)::int AS views FROM events e JOIN phones p ON p.id = e.phone_id
WHERE e.type = 'page_view' AND e.page_type = 'phone' AND e.created_at > now() - make_interval(days => $1)
GROUP BY 1, 2 ORDER BY 3 DESC LIMIT 20""", days),
            "top_compared_pairs": await q(
                """SELECT a.display_name AS phone_a, b.display_name AS phone_b, s.uniq_30d, s.uniq_7d
FROM compare_pair_stats s JOIN phones a ON a.id = s.phone_a JOIN phones b ON b.id = s.phone_b
ORDER BY s.uniq_30d DESC LIMIT 20"""),
            "compare_to_offer_click": await q(
                """SELECT count(DISTINCT c.session_key)::int AS compare_sessions,
count(DISTINCT o.session_key)::int AS also_clicked_offer
FROM events c LEFT JOIN events o ON o.session_key = c.session_key AND o.type = 'offer_click' AND o.created_at > c.created_at
WHERE c.type = 'compare_add' AND c.created_at > now() - make_interval(days => $1)""", days),
        }


# ── jobs ─────────────────────────────────────────────────────────────────────

@router.post("/jobs/refresh-derived", dependencies=AUTH)
async def refresh_derived_job():
    async with acquire_conn() as conn:
        result = await refresh_all(conn)
    await cache.clear()
    return {"refreshed": True, **result}


@router.post("/jobs/rollups", dependencies=AUTH)
async def rollups_job():
    async with acquire_conn() as conn:
        result = await run_all_rollups(conn)
    await cache.clear()
    return {"ok": True, **result}


@router.post("/jobs/reload-reference", dependencies=AUTH)
async def reload_reference_job():
    async with acquire_conn() as conn:
        out = await load_reference_data(conn)
    await cache.clear()
    return {"reloaded": out}


@router.post("/jobs/clear-ai-cache", dependencies=AUTH)
async def clear_ai_cache_job():
    async with acquire_conn() as conn:
        deleted = await conn.fetchval("WITH d AS (DELETE FROM ai_copy_cache RETURNING 1) SELECT COUNT(*) FROM d")
    await cache.clear()
    return {"cleared": True, "deleted": deleted}
