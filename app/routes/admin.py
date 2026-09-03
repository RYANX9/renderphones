"""
Admin-only routes — full file.
Drop this into app/routes/admin.py, replacing the previous version.

Includes:
  - GET  /admin/phones                    — list/search/filter/sort with all 6 sub-scores
                                             + ram_options/storage_options (from phone_specs)
                                             + price_point_count (from price_points)
  - PUT  /admin/phones/{id}/link          — update amazon_link
  - PUT  /admin/phones/{id}/price         — update phones.price_usd only
  - POST /admin/links/check               — broken link checker (phones.amazon_link)
  - GET  /admin/phones/{id}/variants      — list variants for a phone
  - PUT  /admin/variants/{id}             — update existing variant price/url
  - DELETE /admin/variants/{id}           — delete a variant
  - POST /admin/phones/{id}/variants      — create new variant
  - POST /admin/phones/{id}/price-point   — add a manual price_points snapshot
                                            (optionally tied to a specific variant)
  - GET    /admin/phones/{id}/links       — list retailer_links for a phone (scope-annotated)
  - POST   /admin/phones/{id}/links       — create a retailer_links row
  - PUT    /admin/links/{id}              — update a retailer_links row
  - DELETE /admin/links/{id}              — delete a retailer_links row
  - POST   /admin/retailer-links/check    — broken link checker (retailer_links)
  - GET    /admin/retailers               — every retailer key market.py knows, with its scope
  - GET    /admin/phones/{id}/region-preview — eligibility preview for a phone in a given region

Change in this revision (region/scope alignment):
  market.py was refactored to a scope taxonomy (RETAILER_SCOPE dict +
  region_allowed_for_scope), replacing the old flat
  CHINA_DOMESTIC_RETAILERS / CROSS_BORDER_RETAILERS / GLOBAL_RETAILERS sets.
  This file previously still imported those three removed names (crash on
  startup) and additionally had TWO parallel, drifted copies of the
  retailer_links CRUD (one under /links with status+validation, a second
  looser one under /retailer-links with no validation). The second copy is
  deleted here — the /links set is the single source of truth for
  retailer_links now. _VALID_RETAILERS is now derived directly from
  market.py's RETAILER_SCOPE so admin can never create a link whose
  retailer key doesn't map to a real scope. Two new endpoints
  (GET /retailers, GET /phones/{id}/region-preview) give admin visibility
  into scope classification and per-phone regional eligibility, which was
  previously impossible to verify without reading source.

Auth: every route requires header `X-Admin-Key` matching settings.admin_api_key.

Wiring required in your repo (one-time, already done if you followed the previous steps):
  1. requirements.txt  -> httpx>=0.27.0
  2. app/core/config.py -> admin_api_key: str = ""  inside Settings
  3. Render env vars   -> ADMIN_API_KEY=<your key>
  4. main.py           ->
       from app.routes.admin import router as admin_router
       app.include_router(admin_router)
  5. Run migration_retailer_links.sql against the database once, before
     deploying this file — the retailer_links endpoints assume the table
     already exists.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Optional

import anyio
import asyncpg
import httpx
import re
from fastapi import APIRouter, Depends, HTTPException, Header, Query
from pydantic import BaseModel

from ..core.config import settings
from ..core.database import get_pool, row_to_dict, rows_to_list
from ..core.market import RETAILER_SCOPE, retailer_scope, phone_allowed_for_region

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


# ── AUTH ────────────────────────────────────────────────────────────────────

def require_admin_key(x_admin_key: Optional[str] = Header(None)) -> None:
    if not settings.admin_api_key:
        raise HTTPException(status_code=503, detail="ADMIN_API_KEY not configured on server.")
    if not x_admin_key or x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing admin key.")


# ── SCORE HELPERS ────────────────────────────────────────────────────────────

_SUBSCORE_FIELDS = (
    "camera_score", "performance_score", "battery_score",
    "display_score", "build_score", "value_score",
)

_SCORE_SORT_FIELDS = {
    "rank": "overall_score",
    "overall": "overall_score",
    "camera": "camera_score",
    "performance": "performance_score",
    "battery": "battery_score",
    "display": "display_score",
    "build": "build_score",
    "value": "value_score",
}


def _resolve_overall(row: dict) -> tuple[Optional[float], bool]:
    """(overall_score, is_estimated).
    Real AI overall_score wins outright; falls back to the average of
    whatever sub-scores exist (mirrors average_component_scores in
    core/scoring.py)."""
    if row.get("overall_score") is not None:
        return float(row["overall_score"]), False
    vals = [float(row[f]) for f in _SUBSCORE_FIELDS if row.get(f) is not None]
    if not vals:
        return None, False
    return round(sum(vals) / len(vals), 1), True


def _sort_phones(rows: list[dict], sort: str) -> list[dict]:
    if sort in ("release_date", "newest"):
        return sorted(
            rows,
            key=lambda r: (
                r.get("release_year") or 0,
                r.get("release_month") or 0,
                r.get("release_day") or 0,
            ),
            reverse=True,
        )
    if sort in _SCORE_SORT_FIELDS:
        field = _SCORE_SORT_FIELDS[sort]
        return sorted(rows, key=lambda r: (r.get(field) is None, -(r.get(field) or 0)))
    if sort == "price":
        return sorted(rows, key=lambda r: (r.get("price_usd") is None, -(r.get("price_usd") or 0)))
    if sort == "name":
        return sorted(rows, key=lambda r: f"{r.get('brand', '')} {r.get('model_name', '')}".lower())
    if sort == "missing_link_first":
        return sorted(rows, key=lambda r: (bool(r.get("amazon_link")), -(r.get("overall_score") or -1)))
    # default: rank
    return sorted(rows, key=lambda r: (r.get("overall_score") is None, -(r.get("overall_score") or 0)))


# ── LIST PHONES ──────────────────────────────────────────────────────────────

@router.get("/phones", dependencies=[Depends(require_admin_key)])
async def list_phones_for_admin(
    q: Optional[str] = Query(None),
    link_status: Optional[str] = Query(None, description="'missing' | 'present' | omit for all"),
    tier: Optional[str] = Query(None),
    sort: str = Query("rank"),
    page: int = Query(1, ge=1),
    page_size: int = Query(30, ge=1, le=100),
):
    conditions = ["1=1"]
    params: list = []
    i = 1

    if q and q.strip():
        conditions.append(f"(LOWER(p.model_name) LIKE ${i} OR LOWER(p.brand) LIKE ${i})")
        params.append(f"%{q.strip().lower()}%")
        i += 1

    if link_status == "missing":
        conditions.append("(p.amazon_link IS NULL OR p.amazon_link = '')")
    elif link_status == "present":
        conditions.append("(p.amazon_link IS NOT NULL AND p.amazon_link != '')")

    where = " AND ".join(conditions)

    async with get_pool().acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT
                p.id, p.model_name, p.brand, p.slug, p.main_image_url,
                p.amazon_link, p.price_usd, p.availability_status,
                p.release_year, p.release_month, p.release_day,
                sc.overall_score,
                sc.camera_score, sc.performance_score, sc.battery_score,
                sc.display_score, sc.build_score, sc.value_score,
                sc.tier,
                sp.ram_options, sp.storage_options,
                COUNT(DISTINCT pv.id)::int AS variant_count,
                COUNT(DISTINCT pp.id)::int AS price_point_count,
                COUNT(DISTINCT rl.id)::int AS retailer_link_count
            FROM phones p
            LEFT JOIN phone_smart_scores sc ON sc.phone_id = p.id
            LEFT JOIN phone_specs sp ON sp.phone_id = p.id
            LEFT JOIN phone_variants pv ON pv.phone_id = p.id
            LEFT JOIN price_points pp ON pp.phone_id = p.id
            LEFT JOIN retailer_links rl ON rl.phone_id = p.id
            WHERE {where}
            GROUP BY p.id, p.release_year, p.release_month, p.release_day,
                     sc.overall_score, sc.camera_score, sc.performance_score,
                     sc.battery_score, sc.display_score, sc.build_score,
                     sc.value_score, sc.tier, sp.ram_options, sp.storage_options
            ORDER BY p.id
            """,
            *params,
        )

    phones = rows_to_list(rows)

    # Resolve overall score (real AI value or computed average)
    for p in phones:
        overall, estimated = _resolve_overall(p)
        p["overall_score"] = overall
        p["overall_score_estimated"] = estimated

    # Tier filter is done in Python because it's a join field, cheaper here
    if tier:
        phones = [p for p in phones if p.get("tier") == tier]

    phones = _sort_phones(phones, sort)

    total = len(phones)
    offset = (page - 1) * page_size
    page_results = phones[offset: offset + page_size]

    return {"total": total, "page": page, "page_size": page_size, "results": page_results}


# ── UPDATE AFFILIATE LINK ────────────────────────────────────────────────────

class LinkUpdate(BaseModel):
    amazon_link: Optional[str] = None


@router.put("/phones/{phone_id}/link", dependencies=[Depends(require_admin_key)])
async def update_affiliate_link(phone_id: int, payload: LinkUpdate):
    link = (payload.amazon_link or "").strip() or None
    async with get_pool().acquire() as conn:
        row = await conn.fetchrow(
            "UPDATE phones SET amazon_link = $1 WHERE id = $2 RETURNING id, amazon_link",
            link, phone_id,
        )
    if row is None:
        raise HTTPException(status_code=404, detail=f"Phone {phone_id} not found.")
    return row_to_dict(row)


# ── UPDATE BASE PRICE ────────────────────────────────────────────────────────

class PriceUpdate(BaseModel):
    price_usd: Optional[float] = None


@router.put("/phones/{phone_id}/price", dependencies=[Depends(require_admin_key)])
async def update_phone_price(phone_id: int, payload: PriceUpdate):
    """Writes phones.price_usd only. price_points is a manually curated
    history — it's populated exclusively through
    POST /admin/phones/{id}/price-point against a specific variant (or the
    base phone). This route must never fabricate or overwrite a snapshot
    there just because the base price changed."""
    price = payload.price_usd
    if price is not None and price <= 0:
        raise HTTPException(status_code=422, detail="price_usd must be positive.")

    async with get_pool().acquire() as conn:
        row = await conn.fetchrow(
            "UPDATE phones SET price_usd = $1 WHERE id = $2 RETURNING id, price_usd",
            price, phone_id,
        )
    if row is None:
        raise HTTPException(status_code=404, detail=f"Phone {phone_id} not found.")

    return row_to_dict(row)


# ── BROKEN LINK CHECKER (phones.amazon_link) ─────────────────────────────────

class LinkCheckRequest(BaseModel):
    ids: Optional[list[int]] = None
    limit: int = 200


_CHECK_TIMEOUT_S = 6.0
_CHECK_CONCURRENCY = 10


async def _check_one(client: httpx.AsyncClient, item_id: int, url: str) -> dict:
    try:
        resp = await client.head(url, timeout=_CHECK_TIMEOUT_S, follow_redirects=True)
        if resp.status_code >= 400:
            # Some retailers 403/405 HEAD but serve GET fine
            resp = await client.get(url, timeout=_CHECK_TIMEOUT_S, follow_redirects=True)
        return {"id": item_id, "url": url, "status_code": resp.status_code, "ok": resp.status_code < 400}
    except Exception as exc:
        return {"id": item_id, "url": url, "status_code": None, "ok": False, "error": str(exc)[:200]}


@router.post("/links/check", dependencies=[Depends(require_admin_key)])
async def check_affiliate_links(payload: LinkCheckRequest):
    async with get_pool().acquire() as conn:
        if payload.ids:
            rows = await conn.fetch(
                """
                SELECT id, amazon_link FROM phones
                WHERE id = ANY($1::int[]) AND amazon_link IS NOT NULL AND amazon_link != ''
                """,
                payload.ids,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT id, amazon_link FROM phones
                WHERE amazon_link IS NOT NULL AND amazon_link != ''
                ORDER BY id LIMIT $1
                """,
                payload.limit,
            )

    targets = [(r["id"], r["amazon_link"]) for r in rows]
    if not targets:
        return {"checked": 0, "broken": [], "ok": []}

    semaphore = anyio.Semaphore(_CHECK_CONCURRENCY)
    results: list[dict] = []

    async def _bounded(client: httpx.AsyncClient, phone_id: int, url: str) -> None:
        async with semaphore:
            results.append(await _check_one(client, phone_id, url))

    async with httpx.AsyncClient(headers={"User-Agent": "Mozilla/5.0 (SpecmobLinkChecker)"}) as client:
        async with anyio.create_task_group() as tg:
            for phone_id, url in targets:
                tg.start_soon(_bounded, client, phone_id, url)

    broken = [r for r in results if not r["ok"]]
    ok = [r for r in results if r["ok"]]
    return {"checked": len(results), "broken": broken, "ok": ok}


# ── VARIANTS — LIST ──────────────────────────────────────────────────────────

@router.get("/phones/{phone_id}/variants", dependencies=[Depends(require_admin_key)])
async def get_phone_variants(phone_id: int):
    async with get_pool().acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, phone_id, ram_gb, storage_gb, price, url, created_at
            FROM phone_variants
            WHERE phone_id = $1
            ORDER BY storage_gb ASC, ram_gb ASC NULLS FIRST
            """,
            phone_id,
        )
    return {"phone_id": phone_id, "variants": rows_to_list(rows)}


# ── VARIANTS — UPDATE ────────────────────────────────────────────────────────

class VariantUpdate(BaseModel):
    price: Optional[float] = None
    url: Optional[str] = None


@router.put("/variants/{variant_id}", dependencies=[Depends(require_admin_key)])
async def update_variant(variant_id: int, payload: VariantUpdate):
    updates: dict = {}
    if payload.price is not None:
        updates["price"] = payload.price
    # Allow explicitly setting url to null/empty
    if "url" in payload.model_fields_set:
        updates["url"] = (payload.url or "").strip() or None

    if not updates:
        raise HTTPException(status_code=400, detail="Nothing to update.")

    set_clause = ", ".join(f"{k} = ${i + 2}" for i, k in enumerate(updates))
    values = list(updates.values())

    async with get_pool().acquire() as conn:
        row = await conn.fetchrow(
            f"UPDATE phone_variants SET {set_clause} WHERE id = $1 RETURNING *",
            variant_id, *values,
        )
    if row is None:
        raise HTTPException(status_code=404, detail=f"Variant {variant_id} not found.")
    return row_to_dict(row)


# ── VARIANTS — DELETE ────────────────────────────────────────────────────────

@router.delete("/variants/{variant_id}", dependencies=[Depends(require_admin_key)])
async def delete_variant(variant_id: int):
    """Deletes the variant. retailer_links rows attached to it are removed
    by the table's ON DELETE CASCADE. price_points rows that reference it
    are explicitly detached first (variant_id set to NULL) so a variant
    being retired never blocks on price_points and never silently drops
    historical snapshots — done inside one transaction so both writes
    succeed or neither does."""
    async with get_pool().acquire() as conn:
        async with conn.transaction():
            exists = await conn.fetchval(
                "SELECT phone_id FROM phone_variants WHERE id = $1", variant_id,
            )
            if exists is None:
                raise HTTPException(status_code=404, detail=f"Variant {variant_id} not found.")

            await conn.execute(
                "UPDATE price_points SET variant_id = NULL WHERE variant_id = $1",
                variant_id,
            )
            await conn.execute("DELETE FROM phone_variants WHERE id = $1", variant_id)

    return {"deleted": True, "id": variant_id, "phone_id": exists}


# ── LINK STATS (sidebar counter) ─────────────────────────────────────────────

_PLACEHOLDER_LINK_RE = re.compile(r"amazon\.[a-z.]+/s(\?|/)")


@router.get("/stats/links", dependencies=[Depends(require_admin_key)])
async def get_link_stats():
    """Cheap global count of amazon_link states, without the phone_smart_scores /
    phone_specs / phone_variants / price_points joins list_phones_for_admin carries.
    Mirrors the isPlaceholderLink() regex in admin.html so the two stay in sync."""
    async with get_pool().acquire() as conn:
        rows = await conn.fetch(
            "SELECT amazon_link FROM phones WHERE amazon_link IS NOT NULL AND amazon_link != ''"
        )

    real = placeholder = 0
    for r in rows:
        if _PLACEHOLDER_LINK_RE.search(r["amazon_link"]):
            placeholder += 1
        else:
            real += 1

    return {"real": real, "placeholder": placeholder, "total_with_link": real + placeholder}


# ── VARIANTS — CREATE ────────────────────────────────────────────────────────

class VariantCreate(BaseModel):
    storage_gb: int
    ram_gb: Optional[int] = None
    price: Optional[float] = None
    url: Optional[str] = None


@router.post("/phones/{phone_id}/variants", dependencies=[Depends(require_admin_key)])
async def create_variant(phone_id: int, payload: VariantCreate):
    if payload.storage_gb <= 0:
        raise HTTPException(status_code=422, detail="storage_gb must be positive.")
    url = (payload.url or "").strip() or None

    async with get_pool().acquire() as conn:
        exists = await conn.fetchval("SELECT id FROM phones WHERE id = $1", phone_id)
        if not exists:
            raise HTTPException(status_code=404, detail=f"Phone {phone_id} not found.")

        row = await conn.fetchrow(
            """
            INSERT INTO phone_variants (phone_id, ram_gb, storage_gb, price, url)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING *
            """,
            phone_id, payload.ram_gb, payload.storage_gb, payload.price, url,
        )
    return row_to_dict(row)


# ── PRICE POINT — ADD MANUAL SNAPSHOT ───────────────────────────────────────

class PricePointCreate(BaseModel):
    price_usd: float
    snapshot_date: str           # YYYY-MM-DD
    scope: str = "global"        # 'global' | 'local'
    variant_id: Optional[int] = None   # null = base phone, int = specific variant


@router.post("/phones/{phone_id}/price-point", dependencies=[Depends(require_admin_key)])
async def add_price_point(phone_id: int, payload: PricePointCreate):
    """Records a new price_points snapshot.
    variant_id lets you attribute the price to a specific RAM/storage config.
    If variant_id is null the snapshot is for the base phone (no specific config).
    Uses UPDATE then INSERT so it works whether or not a unique constraint on
    (phone_id, snapshot_date, scope) exists."""
    try:
        snap_date = date.fromisoformat(payload.snapshot_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid snapshot_date. Use YYYY-MM-DD.")

    if payload.price_usd <= 0:
        raise HTTPException(status_code=422, detail="price_usd must be > 0.")

    if payload.scope not in ("global", "local"):
        raise HTTPException(status_code=422, detail="scope must be 'global' or 'local'.")

    async with get_pool().acquire() as conn:
        exists = await conn.fetchval("SELECT id FROM phones WHERE id = $1", phone_id)
        if not exists:
            raise HTTPException(status_code=404, detail=f"Phone {phone_id} not found.")

        if payload.variant_id is not None:
            v_exists = await conn.fetchval(
                "SELECT id FROM phone_variants WHERE id = $1 AND phone_id = $2",
                payload.variant_id, phone_id,
            )
            if not v_exists:
                raise HTTPException(
                    status_code=400,
                    detail=f"Variant {payload.variant_id} does not belong to phone {phone_id}.",
                )

        # Try update first (avoids needing a unique constraint)
        result = await conn.execute(
            """
            UPDATE price_points
            SET price_usd = $1, variant_id = $2
            WHERE phone_id = $3 AND snapshot_date = $4 AND scope = $5
            """,
            payload.price_usd, payload.variant_id,
            phone_id, snap_date, payload.scope,
        )

        if result == "UPDATE 0":
            # No existing row — insert fresh
            await conn.execute(
                """
                INSERT INTO price_points (phone_id, variant_id, snapshot_date, scope, price_usd)
                VALUES ($1, $2, $3, $4, $5)
                """,
                phone_id, payload.variant_id, snap_date, payload.scope, payload.price_usd,
            )

    return {
        "phone_id": phone_id,
        "variant_id": payload.variant_id,
        "snapshot_date": str(snap_date),
        "scope": payload.scope,
        "price_usd": payload.price_usd,
    }


# ── RETAILER SCOPE VISIBILITY (new) ──────────────────────────────────────────
# market.py is the single source of truth for which retailer belongs to
# which market scope (global / europe / china_market / china_version /
# india_pakistan). These two endpoints expose that mapping to admin so a
# link can never be created against a retailer key whose scope isn't
# knowable, and so a phone's regional eligibility can be checked before
# it goes live instead of discovered later via a bug report.

@router.get("/retailers", dependencies=[Depends(require_admin_key)])
async def list_known_retailers():
    """Every retailer key market.py knows about, with its scope, so the
    admin UI can build a dropdown instead of free-typing a retailer name
    that silently resolves to the wrong (or 'global' fail-open) scope."""
    return {
        "retailers": sorted(
            [{"retailer": r, "scope": s} for r, s in RETAILER_SCOPE.items()],
            key=lambda x: (x["scope"], x["retailer"]),
        )
    }


@router.get("/phones/{phone_id}/region-preview", dependencies=[Depends(require_admin_key)])
async def preview_region_eligibility(phone_id: int, region: str = Query(...)):
    """Shows exactly which retailer_links row(s) make this phone visible
    (or not) for a given region, and which scope each row resolved to.
    Mirrors the same phone_allowed_for_region() call the live /pick and
    /categories routes use, so what admin sees here is what users get."""
    async with get_pool().acquire() as conn:
        exists = await conn.fetchval("SELECT id FROM phones WHERE id = $1", phone_id)
        if not exists:
            raise HTTPException(status_code=404, detail=f"Phone {phone_id} not found.")
        rows = await conn.fetch(
            "SELECT id, retailer, region, is_available FROM retailer_links WHERE phone_id = $1",
            phone_id,
        )
    links = rows_to_list(rows)
    visible = phone_allowed_for_region(links, region)
    return {
        "phone_id": phone_id,
        "region": region.upper(),
        "visible": visible,
        "links": [{**l, "scope": retailer_scope(l["retailer"])} for l in links],
    }


# ── RETAILER LINKS ────────────────────────────────────────────────────────
# Multi-retailer / multi-region tracking (migration_retailer_links.sql).
# Decoupled from phones.amazon_link and phone_variants.url — those remain
# the single "primary" link the frontend reads everywhere today. A link
# here can attach to a specific variant (variant_id set) or to the phone
# as a whole (variant_id null).
#
# _VALID_RETAILERS is derived from market.py's RETAILER_SCOPE keys rather
# than hand-maintained here — this is the actual fix for the region bugs
# that kept recurring: a link's eligibility is driven by WHICH RETAILER it
# is (retailer_scope() looks up the retailer key), not by the free-text
# `region` field admin types in. `region` is display/priority-sort metadata
# only (see market.py's _offer_priority). "other" stays valid on top of
# the known keys since retailer_scope() fail-opens unknown retailers to
# "global" by design — an unclassified retailer should never accidentally
# hide a phone from every region.

_VALID_RETAILERS = set(RETAILER_SCOPE.keys()) | {"other"}
_VALID_LINK_STATUSES = {"unchecked", "ok", "broken", "region_locked"}


class RetailerLinkCreate(BaseModel):
    variant_id: Optional[int] = None
    retailer: str
    region: str = "US"
    url: str
    price: Optional[float] = None
    currency: Optional[str] = None
    is_available: bool = True
    status: str = "unchecked"
    notes: Optional[str] = None


class RetailerLinkUpdate(BaseModel):
    retailer: Optional[str] = None
    region: Optional[str] = None
    url: Optional[str] = None
    price: Optional[float] = None
    currency: Optional[str] = None
    is_available: Optional[bool] = None
    status: Optional[str] = None
    notes: Optional[str] = None


def _validate_retailer(retailer: str) -> str:
    r = retailer.strip().lower()
    if r not in _VALID_RETAILERS:
        raise HTTPException(status_code=422, detail=f"retailer must be one of {sorted(_VALID_RETAILERS)}.")
    return r


def _validate_link_status(status: str) -> str:
    s = status.strip().lower()
    if s not in _VALID_LINK_STATUSES:
        raise HTTPException(status_code=422, detail=f"status must be one of {sorted(_VALID_LINK_STATUSES)}.")
    return s


@router.get("/phones/{phone_id}/links", dependencies=[Depends(require_admin_key)])
async def list_retailer_links(phone_id: int):
    async with get_pool().acquire() as conn:
        exists = await conn.fetchval("SELECT id FROM phones WHERE id = $1", phone_id)
        if not exists:
            raise HTTPException(status_code=404, detail=f"Phone {phone_id} not found.")

        rows = await conn.fetch(
            """
            SELECT id, phone_id, variant_id, retailer, region, url,
                   price, currency, is_available, status, notes,
                   checked_at, created_at, updated_at
            FROM retailer_links
            WHERE phone_id = $1
            ORDER BY variant_id ASC NULLS FIRST, retailer ASC, region ASC
            """,
            phone_id,
        )
    # scope is computed, not stored — always reflects market.py's current
    # classification even if it's edited after a link was created.
    links = [{**l, "scope": retailer_scope(l["retailer"])} for l in rows_to_list(rows)]
    return {"phone_id": phone_id, "links": links}


@router.post("/phones/{phone_id}/links", status_code=201, dependencies=[Depends(require_admin_key)])
async def create_retailer_link(phone_id: int, payload: RetailerLinkCreate):
    retailer = _validate_retailer(payload.retailer)
    status = _validate_link_status(payload.status)
    region = payload.region.strip().upper() or "US"
    url = payload.url.strip()
    if not url:
        raise HTTPException(status_code=422, detail="url is required.")

    async with get_pool().acquire() as conn:
        phone_exists = await conn.fetchval("SELECT id FROM phones WHERE id = $1", phone_id)
        if not phone_exists:
            raise HTTPException(status_code=404, detail=f"Phone {phone_id} not found.")

        if payload.variant_id is not None:
            variant_ok = await conn.fetchval(
                "SELECT id FROM phone_variants WHERE id = $1 AND phone_id = $2",
                payload.variant_id, phone_id,
            )
            if not variant_ok:
                raise HTTPException(
                    status_code=400,
                    detail=f"Variant {payload.variant_id} does not belong to phone {phone_id}.",
                )

        try:
            row = await conn.fetchrow(
                """
                INSERT INTO retailer_links
                    (phone_id, variant_id, retailer, region, url, price,
                     currency, is_available, status, notes, checked_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                        CASE WHEN $9 != 'unchecked' THEN CURRENT_TIMESTAMP ELSE NULL END)
                RETURNING *
                """,
                phone_id, payload.variant_id, retailer, region, url,
                payload.price, payload.currency, payload.is_available,
                status, payload.notes,
            )
        except asyncpg.UniqueViolationError:
            raise HTTPException(
                status_code=409,
                detail="A link for this phone/variant + retailer + region already exists — update it instead.",
            )

    result = row_to_dict(row)
    result["scope"] = retailer_scope(result["retailer"])
    return result


@router.put("/links/{link_id}", dependencies=[Depends(require_admin_key)])
async def update_retailer_link(link_id: int, payload: RetailerLinkUpdate):
    updates: dict = {}
    if payload.retailer is not None:
        updates["retailer"] = _validate_retailer(payload.retailer)
    if payload.region is not None:
        updates["region"] = payload.region.strip().upper() or "US"
    if payload.url is not None:
        url = payload.url.strip()
        if not url:
            raise HTTPException(status_code=422, detail="url cannot be empty.")
        updates["url"] = url
    if payload.price is not None:
        updates["price"] = payload.price
    if payload.currency is not None:
        updates["currency"] = payload.currency
    if payload.is_available is not None:
        updates["is_available"] = payload.is_available
    if payload.status is not None:
        updates["status"] = _validate_link_status(payload.status)
        updates["checked_at"] = datetime.utcnow()
    if payload.notes is not None:
        updates["notes"] = payload.notes

    if not updates:
        raise HTTPException(status_code=400, detail="Nothing to update.")

    updates["updated_at"] = datetime.utcnow()
    set_clause = ", ".join(f"{k} = ${i + 2}" for i, k in enumerate(updates))
    values = list(updates.values())

    async with get_pool().acquire() as conn:
        try:
            row = await conn.fetchrow(
                f"UPDATE retailer_links SET {set_clause} WHERE id = $1 RETURNING *",
                link_id, *values,
            )
        except asyncpg.UniqueViolationError:
            raise HTTPException(
                status_code=409,
                detail="Another link already exists for this phone/variant + retailer + region.",
            )
    if row is None:
        raise HTTPException(status_code=404, detail=f"Link {link_id} not found.")

    result = row_to_dict(row)
    result["scope"] = retailer_scope(result["retailer"])
    return result


@router.delete("/links/{link_id}", dependencies=[Depends(require_admin_key)])
async def delete_retailer_link(link_id: int):
    async with get_pool().acquire() as conn:
        row = await conn.fetchrow(
            "DELETE FROM retailer_links WHERE id = $1 RETURNING id, phone_id",
            link_id,
        )
    if row is None:
        raise HTTPException(status_code=404, detail=f"Link {link_id} not found.")
    return {"deleted": True, "id": row["id"], "phone_id": row["phone_id"]}


# Broken-link sweep for retailer_links, same pattern as check_affiliate_links.
class RetailerLinkCheckRequest(BaseModel):
    ids: Optional[list[int]] = None
    limit: int = 200


@router.post("/retailer-links/check", dependencies=[Depends(require_admin_key)])
async def check_retailer_links(payload: RetailerLinkCheckRequest):
    async with get_pool().acquire() as conn:
        if payload.ids:
            rows = await conn.fetch(
                "SELECT id, url FROM retailer_links WHERE id = ANY($1::int[])",
                payload.ids,
            )
        else:
            rows = await conn.fetch(
                "SELECT id, url FROM retailer_links ORDER BY id LIMIT $1",
                payload.limit,
            )

    targets = [(r["id"], r["url"]) for r in rows]
    if not targets:
        return {"checked": 0, "broken": [], "ok": []}

    semaphore = anyio.Semaphore(_CHECK_CONCURRENCY)
    results: list[dict] = []

    async def _bounded(client: httpx.AsyncClient, link_id: int, url: str) -> None:
        async with semaphore:
            results.append(await _check_one(client, link_id, url))

    async with httpx.AsyncClient(headers={"User-Agent": "Mozilla/5.0 (SpecmobLinkChecker)"}) as client:
        async with anyio.create_task_group() as tg:
            for link_id, url in targets:
                tg.start_soon(_bounded, client, link_id, url)

    async with get_pool().acquire() as conn:
        for r in results:
            await conn.execute(
                "UPDATE retailer_links SET status = $1, checked_at = CURRENT_TIMESTAMP WHERE id = $2",
                "ok" if r["ok"] else "broken", r["id"],
            )

    broken = [r for r in results if not r["ok"]]
    ok = [r for r in results if r["ok"]]
    return {"checked": len(results), "broken": broken, "ok": ok}