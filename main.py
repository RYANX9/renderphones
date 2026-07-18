from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.database import create_pool, close_pool
from app.core.middleware import RequestContextMiddleware, RateLimitMiddleware
from app.routes.phones import router as phones_router
from app.routes.brands import router as brands_router
from app.routes.categories import router as categories_router
from app.routes.misc import router as misc_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_pool(
        settings.database_url,
        min_size=settings.db_pool_min,
        max_size=settings.db_pool_max,
        command_timeout=settings.db_command_timeout,
    )
    # trigram similarity (used by search's typo-tolerant matching) needs
    # this extension enabled once per database.
    from app.core.database import get_pool
    async with get_pool().acquire() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    yield
    await close_pool()


app = FastAPI(
    title="Specmob API",
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestContextMiddleware)
app.add_middleware(
    RateLimitMiddleware,
    requests=settings.rate_limit_requests,
    window=settings.rate_limit_window,
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        },
    )


app.include_router(phones_router)
app.include_router(brands_router)
app.include_router(categories_router)
app.include_router(misc_router)
