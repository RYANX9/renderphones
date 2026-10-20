from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, ORJSONResponse

from app.core.config import get_settings
from app.core.reference import load_reference_data
from app.core.database import acquire_conn, close_pool, create_pool
from app.core.http_cache import CacheControlMiddleware
from app.core.middleware import RateLimitMiddleware, RequestContextMiddleware
from app.jobs import scheduler
from app.routes.admin import router as admin_router
from app.routes.brands import router as brands_router
from app.routes.categories import router as categories_router
from app.routes.compare import router as compare_router
from app.routes.discover import router as discover_router
from app.routes.events import router as events_router
from app.routes.feedback import router as feedback_router
from app.routes.misc import router as misc_router
from app.routes.phones import router as phones_router
from app.routes.seo import router as seo_router
from app.routes.tradein import router as tradein_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s %(message)s")
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_pool(
        settings.database_url, min_size=settings.db_pool_min, max_size=settings.db_pool_max,
        command_timeout=settings.db_command_timeout,
    )
    async with acquire_conn() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
        logger.info("reference data: %s", await load_reference_data(conn))
    scheduler.start()
    yield
    await scheduler.stop()
    await close_pool()


app = FastAPI(title="Specmob API", version=settings.app_version, lifespan=lifespan, default_response_class=ORJSONResponse)

app.add_middleware(
    CORSMiddleware, allow_origins=settings.cors_origins_list, allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)
app.add_middleware(CacheControlMiddleware)
app.add_middleware(RequestContextMiddleware)
app.add_middleware(
    RateLimitMiddleware, requests=settings.rate_limit_requests, window=settings.rate_limit_window,
    internal_key=settings.internal_api_key,
)
app.add_middleware(GZipMiddleware, minimum_size=800)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Real failures are 5xx so the Next.js layer can tell them apart from a true
    # 404 and refuse to cache them. Details stay in the logs, not the response.
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500, content={"detail": "Internal server error"},
        headers={"Access-Control-Allow-Origin": "*", "Access-Control-Allow-Methods": "*", "Access-Control-Allow-Headers": "*"},
    )


for r in (phones_router, compare_router, brands_router, categories_router, discover_router, events_router,
          seo_router, misc_router, tradein_router, feedback_router, admin_router):
    app.include_router(r)
