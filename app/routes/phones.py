from __future__ import annotations

from typing import Optional

import anyio
from fastapi import APIRouter, HTTPException, Query, Request, Response

from ..core.cache import cache, cached
from ..core.category_config import CATEGORY_CONFIG, PRIORITY_CATEGORY
from ..core.config import settings
from ..core.database import acquire_conn
from ..core.market import resolve_offers_for_region
from ..core.query import FilterParams
from ..core.seo import phone_seo
from ..core.slugs import category_path, MAX_COMPARE
from ..services import compare_service, next_steps, phone_repo, stats_repo
from ..services.compare_copy import generate_compare_verdict
from ..services.recommend_copy import generate_match_copy
from ..services.recommend_service import TIER_BOUNDS, TIER_RANGES, recommend as recommend_svc

router = APIRouter(prefix="/phones", tags=["phones"])


def qkey(request: Request) -> str:
    return "&".join(f"{k}={v}" for k, v in sorted(request.query_params.multi_items()))


def _region(value: Optional[str]) -> Optional[str]:
    return value.strip().upper()[:2] if value and value.strip() else None


def _csv(value: Optional[str]) -> list[str] | None:
    items = [t.strip() for t in value.split(",") if t.strip()] if value else None
    return items or None


# ── search ───────────────────────────────────────────────────────────────────

# A text search returns one page: the best matches, nothing deeper. Anyone who
# wants something else changes the search instead of paging through it.
SEARCH_RESULT_CAP = 24

@router.get("/search")
async def search_phones(
    request: Request,
    q: Optional[str] = Query(None, max_length=100), brand: Optional[str] = Query(None),
    brands: Optional[str] = Query(None, description="Comma-separated brand slugs"),
    chipset: Optional[str] = Query(None), min_price: Optional[float] = Query(None), max_price: Optional[float] = Query(None),
    min_ram: Optional[int] = Query(None), min_storage: Optional[int] = Query(None), min_battery: Optional[int] = Query(None),
    min_camera_mp: Optional[int] = Query(None), min_screen_size: Optional[float] = Query(None), max_screen_size: Optional[float] = Query(None),
    min_year: Optional[int] = Query(None), max_year: Optional[int] = Query(None), max_weight: Optional[int] = Query(None),
    min_charging_w: Optional[int] = Query(None), min_refresh_rate: Optional[int] = Query(None), min_antutu: Optional[int] = Query(None),
    tier: Optional[str] = Query(None), chipset_tier: Optional[str] = Query(None, description="Alias of `tier`"),
    has_nfc: Optional[bool] = Query(None), has_ois: Optional[bool] = Query(None),
    has_wireless_charging: Optional[bool] = Query(None), has_headphone_jack: Optional[bool] = Query(None),
    is_foldable: Optional[bool] = Query(None), is_premium_gaming: Optional[bool] = Query(None),
    camera_setup_type: Optional[str] = Query(None), water_resistant: Optional[bool] = Query(None),
    features: Optional[str] = Query(None, description="Comma-separated feature tag keys"),
    region: Optional[str] = Query(None), include_upcoming: bool = Query(True),
    sort_by: str = Query("relevance"), sort_order: str = Query("desc"),
    page: int = Query(1, ge=1, le=500), page_size: int = Query(24, ge=1, le=100),
):
    text_search = bool(q and q.strip())
    if text_search:
        page, page_size = 1, min(page_size, SEARCH_RESULT_CAP)

    filters = FilterParams(
        q=q, brand=brand, brands=_csv(brands), chipset=chipset, min_price=min_price, max_price=max_price,
        min_ram=min_ram, min_storage=min_storage, min_battery=min_battery, min_camera_mp=min_camera_mp,
        min_screen_size=min_screen_size, max_screen_size=max_screen_size, min_year=min_year, max_year=max_year,
        max_weight=max_weight, min_charging_w=min_charging_w, min_refresh_rate=min_refresh_rate, min_antutu=min_antutu,
        tier=tier or chipset_tier, has_nfc=has_nfc, has_ois=has_ois, has_wireless_charging=has_wireless_charging,
        has_headphone_jack=has_headphone_jack, is_foldable=is_foldable, is_premium_gaming=is_premium_gaming,
        camera_setup_type=camera_setup_type, water_resistant=water_resistant, features=_csv(features),
        region=_region(region), include_upcoming=include_upcoming, include_all=True,
    )

    async def fetch():
        async with acquire_conn() as conn:
            total, phones = await phone_repo.search(conn, filters=filters, sort_by=sort_by, sort_order=sort_order,
                                                    page=page, page_size=page_size)
            out = {"total": total, "page": page, "page_size": page_size,
                   "total_pages": 1 if text_search else max(1, -(-total // page_size)), "results": phones}
            if total == 0:
                out["empty"] = await _empty_search(conn, q)
            return out

    # ?page=N is ignored for text searches, so it must not create new cache entries.
    key = qkey(request) if not text_search else "&".join(
        f"{k}={v}" for k, v in sorted(request.query_params.multi_items()) if k not in ("page", "page_size")
    )
    return await cached(f"search:{key}:{page_size}", 300, fetch)


async def _empty_search(conn, q: Optional[str]) -> dict:
    """A search that finds nothing still leads somewhere: the closest names by
    trigram similarity, then popular categories."""
    closest: list[dict] = []
    if q and len(q.strip()) >= 2:
        rows = await conn.fetch(
            "SELECT p.id FROM phone_catalog p WHERE TRUE "
            "ORDER BY similarity(lower(p.display_name), $1) DESC, p.overall_score DESC NULLS LAST LIMIT 6", q.strip().lower(),
        )
        minis = await phone_repo.minis_by_ids(conn, [r["id"] for r in rows])
        closest = [minis[r["id"]] for r in rows if r["id"] in minis]
    cats = [{"slug": s, "title": c["title"], "path": category_path(s)} for s, c in list(CATEGORY_CONFIG.items())[:5]]
    return {"closest": closest, "categories": cats}


# ── curated lists ────────────────────────────────────────────────────────────

@router.get("/latest")
async def latest_phones(limit: int = Query(20, ge=1, le=100)):
    async def fetch():
        async with acquire_conn() as conn:
            return {"phones": await phone_repo.latest(conn, limit)}
    return await cached(f"phones:latest:{limit}", settings.cache_ttl_stable, fetch)


@router.get("/upcoming")
async def upcoming_phones(limit: int = Query(20, ge=1, le=100)):
    async def fetch():
        async with acquire_conn() as conn:
            return {"phones": await phone_repo.upcoming(conn, limit)}
    return await cached(f"phones:upcoming:{limit}", settings.cache_ttl_stable, fetch)


@router.get("/trending")
async def trending_phones(limit: int = Query(10, ge=1, le=50), region: Optional[str] = Query(None)):
    region = _region(region)
    async def fetch():
        async with acquire_conn() as conn:
            return {"phones": await phone_repo.trending(conn, limit, region)}
    return await cached(f"phones:trending:{limit}:{region}", settings.cache_ttl_trending, fetch)


@router.get("/top-rated")
async def top_rated_phones(limit: int = Query(10, ge=1, le=50), region: Optional[str] = Query(None)):
    region = _region(region)
    async def fetch():
        async with acquire_conn() as conn:
            return {"phones": await phone_repo.top_rated(conn, limit, region)}
    return await cached(f"phones:top:{limit}:{region}", settings.cache_ttl_stable, fetch)


@router.get("/by-ids")
async def phones_by_ids(ids: str = Query(..., description="Comma-separated phone ids, max 12")):
    """Hydrates the visitor's recently-viewed / saved-compare lists (ids live in the browser)."""
    id_list = [int(i) for i in ids.split(",") if i.strip().isdigit()][:12]
    if not id_list:
        raise HTTPException(status_code=400, detail="No valid ids provided.")
    async with acquire_conn() as conn:
        minis = await phone_repo.minis_by_ids(conn, id_list)
    return {"phones": [minis[i] for i in id_list if i in minis]}


@router.get("/slugs")
async def phone_slugs():
    async def fetch():
        async with acquire_conn() as conn:
            return {"phones": await phone_repo.slug_index(conn)}
    return await cached("phones:slugs", 86400, fetch)


@router.get("/resolve/{full_slug}")
async def resolve_full_slug(full_slug: str):
    """Old-style URL support: /brand/apple/apple-iphone-12 -> canonical path."""
    async def fetch():
        async with acquire_conn() as conn:
            return await phone_repo.resolve_full_slug(conn, full_slug)
    found = await cached(f"resolve:{full_slug.lower()}", 3600, fetch)
    if not found:
        raise HTTPException(status_code=404, detail="Phone not found.")
    return found


# ── detail ───────────────────────────────────────────────────────────────────

async def _phone_response(phone: dict, region: Optional[str]) -> dict:
    phone = dict(phone)
    offer_rows = phone.pop("_offer_rows", [])
    resolved = resolve_offers_for_region(offer_rows, region, include_ineligible=True)
    phone["offers"] = resolved
    phone["seo"] = phone_seo(phone, offer_rows)

    async def steps():
        async with acquire_conn() as conn:
            return await next_steps.for_phone(conn, phone, region)
    phone["next_steps"] = await cached(f"phone-steps:{phone['id']}:{region}", 900, steps)

    self_mini = {k: phone.get(k) for k in (
        "id", "brand", "brand_slug", "name", "slug", "full_slug", "display_name", "main_image_url",
        "price_usd", "price_basis", "price_region", "tier", "overall_score", "path")}
    rival_steps = [s for s in phone["next_steps"] if s["kind"] == "compare" and s.get("phone")]

    async def pairs():
        async with acquire_conn() as conn:
            popular = await stats_repo.pairs_for_phone(conn, phone["id"], 3)
            if popular:
                return popular
        return [{"path": s["href"], "phones": [self_mini, s["phone"]], "source": "rival"} for s in rival_steps][:3]
    phone["compare_with"] = await cached(f"phone-pairs:{phone['id']}", 900, pairs)
    return phone


@router.get("/by-slug/{brand_slug}/{slug}")
async def get_phone_by_slug(brand_slug: str, slug: str, region: Optional[str] = Query(None)):
    """Canonical lookup. Brand-stripped slugs are only unique per brand, so a
    phone is always found by (brand_slug, slug), never by slug alone."""
    region = _region(region)
    async def fetch():
        async with acquire_conn() as conn:
            return await phone_repo.get_detail(conn, brand_slug=brand_slug, slug=slug)
    phone = await cached(f"phone:{brand_slug.lower()}:{slug.lower()}", 600, fetch)
    if phone is None:
        raise HTTPException(status_code=404, detail="Phone not found.")
    return await _phone_response(phone, region)


# ── compare ──────────────────────────────────────────────────────────────────

def _compare_params(ids: Optional[str], slugs: Optional[str]) -> tuple[list[int] | None, list[str] | None]:
    if not ids and not slugs:
        raise HTTPException(status_code=400, detail="Provide `ids` or `slugs` (full slugs).")
    if ids:
        id_list = list(dict.fromkeys(int(i) for i in ids.split(",") if i.strip().isdigit()))[:MAX_COMPARE]
        if not id_list:
            raise HTTPException(status_code=400, detail="No valid ids provided.")
        return id_list, None
    slug_list = list(dict.fromkeys(s.strip().lower() for s in slugs.split(",") if s.strip()))[:MAX_COMPARE]
    if not slug_list:
        raise HTTPException(status_code=400, detail="No valid slugs provided.")
    return None, slug_list


async def _load_compare(id_list, slug_list) -> list[dict]:
    async with acquire_conn() as conn:
        phones = await phone_repo.compare(conn, ids=id_list, full_slugs=slug_list)
    if not phones:
        raise HTTPException(status_code=404, detail="No phones found for the given ids/slugs.")
    return phones


@router.get("/compare")
async def compare_phones(request: Request, ids: Optional[str] = Query(None), slugs: Optional[str] = Query(None),
                         region: Optional[str] = Query(None)):
    id_list, slug_list = _compare_params(ids, slugs)
    region = _region(region)

    async def fetch():
        phones = await _load_compare(id_list, slug_list)
        async with acquire_conn() as conn:
            return await compare_service.build_payload(conn, phones, region=region)

    return await cached(f"compare:{qkey(request)}", 300, fetch)


@router.get("/compare/verdict")
async def compare_verdict(request: Request, response: Response, ids: Optional[str] = Query(None),
                          slugs: Optional[str] = Query(None)):
    id_list, slug_list = _compare_params(ids, slugs)
    phones = await cached(f"cmpphones:{qkey(request)}", 300, lambda: _load_compare(id_list, slug_list))
    if len(phones) < 2:
        response.headers["Cache-Control"] = "no-store"
        return {"winner_id": None, "verdict": None}

    winner_id = compare_service.compare_winner(phones)
    key = compare_service.verdict_key(phones)
    miss_key = f"verdict-miss:{key}"

    async def generate():
        async with acquire_conn() as conn:
            stored = await phone_repo.get_cached_ai_copy(conn, key, 14)
        if stored and stored.get("winner_id") == winner_id:
            return stored
        if await cache.get(miss_key):
            return None
        verdict = await anyio.to_thread.run_sync(generate_compare_verdict, phones, winner_id)
        if verdict:
            async with acquire_conn() as conn:
                await phone_repo.put_cached_ai_copy(conn, key, verdict)
            return verdict
        await cache.set(miss_key, True, 30)
        return None

    verdict = await cached(f"verdict:{key}:{winner_id}", 3600, generate)
    response.headers["Cache-Control"] = (
        "public, max-age=60, s-maxage=300, stale-while-revalidate=600" if verdict else "no-store"
    )
    return {"winner_id": winner_id, "verdict": verdict}


# ── recommend (/pick) ────────────────────────────────────────────────────────

def _match_copy_key(priorities: list[str], budget_label: str, phones: list[dict]) -> str:
    ids = ",".join(str(i) for i in sorted(p["id"] for p in phones))
    return "recommend:v2:" + ",".join(sorted(priorities)) + ":" + budget_label + ":" + ids


async def _get_match_copy(priorities, budget_label, phones):
    key = _match_copy_key(priorities, budget_label, phones)
    miss_key = f"match-miss:{key}"

    async def generate():
        async with acquire_conn() as conn:
            stored = await phone_repo.get_cached_ai_copy(conn, key, 14)
        if stored:
            return stored
        if await cache.get(miss_key):
            return None
        match_copy = await anyio.to_thread.run_sync(generate_match_copy, phones, priorities, budget_label)
        if match_copy:
            payload = {str(pid): copy for pid, copy in match_copy.items()}
            async with acquire_conn() as conn:
                await phone_repo.put_cached_ai_copy(conn, key, payload)
            return payload
        await cache.set(miss_key, True, 30)
        return None

    return await cached(f"matchcopy:{key}", 3600, generate)


def _budget_label(rng: dict) -> str:
    lo, hi = rng.get("min"), rng.get("max")
    if lo and hi:
        return f"${lo:.0f}-${hi:.0f}"
    if hi:
        return f"under ${hi:.0f}"
    if lo:
        return f"${lo:.0f}+"
    return "any budget"


def _first_sentence(text: str | None) -> str | None:
    if not text:
        return None
    for end in (". ", ".\n"):
        if end in text:
            return text.split(end, 1)[0].strip() + "."
    return text.strip()


# app/routes/phones.py
# Replace the whole `recommend_phones` function (the "recommend (/pick)" section
# keeps _match_copy_key, _get_match_copy, _budget_label, _first_sentence as they are).

@router.get("/recommend")
async def recommend_phones(
    request: Request,
    priorities: str = Query("", description="Comma-separated priority ids; empty ranks by overall score"),
    tier: Optional[str] = Query(None, description="price tier s|a|b|c|d. Uses the tier's fixed range and overrides budget."),
    budget: Optional[float] = Query(None, ge=50, le=5000, description="Dial value (ceiling). Range is 80%-105%. Ignored when tier is set."),
    min_price: Optional[float] = Query(None), max_price: Optional[float] = Query(None),
    limit: int = Query(5, ge=1, le=20), region: Optional[str] = Query(None),
):
    # Tier OR dial, never both: a valid tier wins and the dial value is dropped.
    if tier not in TIER_RANGES:
        tier = None
    if tier:
        budget = None
    elif budget is None and min_price is None and max_price is None:
        pass
    priority_list = [p.strip() for p in priorities.split(",") if p.strip()]
    region = _region(region)

    async def fetch():
        async with acquire_conn() as conn:
            result = await recommend_svc(
                conn, priorities=priority_list, min_price=min_price, max_price=max_price,
                limit=limit, region=region, budget=budget, tier=tier,
            )
            analyses = {}
            for p in result["phones"]:
                a = await phone_repo.fetch_analysis(conn, p["id"])
                analyses[p["id"]] = _first_sentence((a or {}).get("reasoning"))

        if result["phones"]:
            match_copy = None
            # AI copy only when priorities exist, so a plain dial drag never calls a model.
            if result["priorities"]:
                label = (
                    f"up to ${budget:.0f}" if result["mode"] == "dial"
                    else _budget_label(result["requested_price_range"])
                )
                match_copy = await _get_match_copy(result["priorities"], label, result["phones"])
            for p in result["phones"]:
                copy = (match_copy or {}).get(str(p["id"]))
                if copy:
                    p["match_line"], p["tradeoff_line"] = copy["match_line"], copy["tradeoff_line"]
                p["reason"] = analyses.get(p["id"])

            top = result["phones"]
            exits = []
            first_cat = next((PRIORITY_CATEGORY[x] for x in result["priorities"] if x in PRIORITY_CATEGORY), None)
            if len(top) >= 2:
                exits.append(next_steps.step(
                    "pick.compare_top", "compare",
                    f"Compare {top[0]['display_name']} vs {top[1]['display_name']}",
                    f"/compare/{'-vs-'.join(sorted([top[0]['full_slug'], top[1]['full_slug']]))}", top[0],
                ))
            if first_cat:
                exits.append(next_steps.step(
                    "pick.category", "category", CATEGORY_CONFIG[first_cat]["title"], category_path(first_cat),
                ))
            exits.append(next_steps.step("pick.hub", "hub", "All best-of lists", next_steps.BEST_HUB))
            result["next_steps"] = exits
        else:
            result["next_steps"] = [next_steps.step("pick.hub", "hub", "Browse best-of lists", next_steps.BEST_HUB)]
        return result

    return await cached(f"recommend:{qkey(request)}", 300, fetch)


# ── by id (kept for admin tools and the compare tray) ────────────────────────

@router.get("/{phone_id}")
async def get_phone(phone_id: int, region: Optional[str] = Query(None)):
    region = _region(region)
    async def fetch():
        async with acquire_conn() as conn:
            return await phone_repo.get_detail(conn, phone_id=phone_id)
    phone = await cached(f"phone-id:{phone_id}", 600, fetch)
    if phone is None:
        raise HTTPException(status_code=404, detail=f"Phone {phone_id} not found.")
    return await _phone_response(phone, region)


async def _require(phone_id: int) -> None:
    async with acquire_conn() as conn:
        if not await phone_repo.phone_exists(conn, phone_id):
            raise HTTPException(status_code=404, detail=f"Phone {phone_id} not found.")


@router.get("/{phone_id}/full-specs")
async def get_full_specs(phone_id: int):
    async def fetch():
        async with acquire_conn() as conn:
            return await phone_repo.fetch_full_specifications(conn, phone_id)
    specs = await cached(f"fullspecs:{phone_id}", 86400, fetch)
    if specs is None:
        raise HTTPException(status_code=404, detail="No full specifications for this phone.")
    return {"phone_id": phone_id, "full_specifications": specs["full_specifications"]}


@router.get("/{phone_id}/variants")
async def get_phone_variants(phone_id: int):
    await _require(phone_id)
    async def fetch():
        async with acquire_conn() as conn:
            return {"phone_id": phone_id, "variants": await phone_repo.fetch_variants(conn, phone_id)}
    return await cached(f"variants:{phone_id}", 600, fetch)


@router.get("/{phone_id}/similar")
async def similar_phones_route(phone_id: int, limit: int = Query(12, ge=1, le=30), region: Optional[str] = Query(None)):
    await _require(phone_id)
    region = _region(region)
    async def fetch():
        async with acquire_conn() as conn:
            return {"phones": await phone_repo.similar_phones(conn, phone_id, limit, region=region)}
    return await cached(f"similar:{phone_id}:{limit}:{region}", 21600, fetch)


@router.get("/{phone_id}/price-history")
async def price_history(phone_id: int):
    """Chart data only. Nothing else in the API reads price_history."""
    await _require(phone_id)
    async def fetch():
        async with acquire_conn() as conn:
            return await phone_repo.price_history(conn, phone_id)
    return await cached(f"pricehist:{phone_id}", 21600, fetch)


@router.get("/{phone_id}/offers")
async def get_phone_offers(phone_id: int, region: Optional[str] = Query(None)):
    await _require(phone_id)
    async with acquire_conn() as conn:
        rows = await phone_repo.fetch_offers(conn, phone_id)
    return resolve_offers_for_region(rows, _region(region), include_ineligible=True)
