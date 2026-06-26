from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.database import create_pool, close_pool
from app.middleware import RequestContextMiddleware, RateLimitMiddleware
from routes.phones import router as phones_router
from routes.brands import router as brands_router
from routes.categories import router as categories_router
from routes.misc import router as misc_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
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
    logger.info("Mobylite API v%s ready", settings.app_version)
    yield
    await close_pool()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Mobylite API",
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    app.add_middleware(RequestContextMiddleware)

    app.add_middleware(
        RateLimitMiddleware,
        requests=settings.rate_limit_requests,
        window=settings.rate_limit_window,
    )

    app.include_router(phones_router)
    app.include_router(brands_router)
    app.include_router(categories_router)
    app.include_router(misc_router)

    return app


app = create_app()
