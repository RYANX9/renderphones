from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ..core.cache import cached
from ..core.config import settings
from ..core.database import acquire_conn
from ..services import phone_repo

logger = logging.getLogger(__name__)
router = APIRouter(tags=["system"])


@router.get("/filters/stats")
async def get_filter_stats():
    async def fetch():
        async with acquire_conn() as conn:
            return await phone_repo.catalog_stats(conn)
    stats = await cached("filters:stats", settings.cache_ttl_stable, fetch)
    if not stats or not stats.get("total_phones"):
        raise HTTPException(status_code=503, detail="Stats not available.")
    return stats


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/health/db")
async def health_db():
    try:
        async with acquire_conn() as conn:
            await conn.fetchval("SELECT 1")
        return {"status": "ok", "db": "connected"}
    except Exception as exc:
        logger.error("Health check failed: %s", exc)
        return JSONResponse(status_code=503, content={"status": "error", "db": "unreachable"})


@router.get("/")
async def root():
    return {"status": "online", "service": "Specmob API", "version": settings.app_version, "docs": "/docs"}
