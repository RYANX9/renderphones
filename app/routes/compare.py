from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from ..core.cache import cached
from ..core.database import acquire_conn
from ..services import compare_service
from .phones import _region

router = APIRouter(prefix="/compare", tags=["compare"])


@router.get("/pair/{pair}")
async def compare_by_pair(pair: str, region: Optional[str] = Query(None)):
    """/compare/{full_slug}-vs-{full_slug}[-vs-...]. Exact full_slug match only.

    If the URL is not the canonical sorted form, the response is
    {"redirect": "/compare/<canonical>"} and the frontend must answer with a
    301 to it. Unknown slugs return 404; there is no fuzzy matching.
    """
    region = _region(region)
    pair = pair.strip().lower()

    async def fetch():
        async with acquire_conn() as conn:
            phones, needs_redirect = await compare_service.load_pair(conn, pair)
            if not phones:
                return None
            if needs_redirect:
                return {"redirect": compare_service.compare_path([p["full_slug"] for p in phones])}
            return await compare_service.build_payload(conn, phones, region=region)

    result = await cached(f"comparepair:{pair}:{region}", 300, fetch)
    if result is None:
        raise HTTPException(status_code=404, detail="One or more phones were not found.")
    return result
