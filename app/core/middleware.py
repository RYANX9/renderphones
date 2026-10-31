# app/core/middleware.py
from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict, deque
from typing import Callable

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .trusted import error_cors_headers

logger = logging.getLogger(__name__)

_RATE_LIMIT_SKIP = frozenset({"/", "/health", "/docs", "/redoc", "/openapi.json"})

class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        rid = uuid.uuid4().hex[:10]
        t0 = time.monotonic()
        request.state.request_id = rid

        try:
            response = await call_next(request)
        except Exception:
            # BaseHTTPMiddleware runs call_next inside an anyio task
            # group; an exception here can surface as an ExceptionGroup
            # that the app-level @app.exception_handler(Exception) does
            # not reliably catch. Handle it here instead of letting it
            # propagate raw past this middleware.
            elapsed_ms = (time.monotonic() - t0) * 1_000
            logger.exception(
                '"%s %s" 500 %.1fms rid=%s (unhandled in request pipeline)',
                request.method, request.url.path, elapsed_ms, rid,
            )
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error", "request_id": rid},
                headers={**error_cors_headers(request.headers.get("origin")), "X-Request-ID": rid},
            )

        elapsed_ms = (time.monotonic() - t0) * 1_000
        response.headers["X-Request-ID"] = rid
        response.headers["X-Response-Time"] = f"{elapsed_ms:.1f}ms"
        response.headers["Server-Timing"] = f"app;dur={elapsed_ms:.1f}"

        logger.info(
            '"%s %s" %d %.1fms rid=%s',
            request.method, request.url.path, response.status_code, elapsed_ms, rid,
        )
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, *, requests: int, window: int, internal_key: str = "") -> None:
        super().__init__(app)
        self._limit = requests
        self._window = window
        self._internal_key = internal_key
        self._clients: dict[str, deque[float]] = defaultdict(deque)
        self._last_prune = time.monotonic()

    def _prune(self, now: float) -> None:
        """Drop idle clients so the table cannot grow without bound."""
        if now - self._last_prune < 120:
            return
        self._last_prune = now
        cutoff = now - self._window
        for ip in [ip for ip, dq in self._clients.items() if not dq or dq[-1] < cutoff]:
            del self._clients[ip]

    def _client_ip(self, request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in _RATE_LIMIT_SKIP:
            return await call_next(request)

        # Trusted server-to-server traffic (Next.js SSR) is never throttled.
        if self._internal_key and request.headers.get("X-Internal-Key") == self._internal_key:
            return await call_next(request)

        ip = self._client_ip(request)
        now = time.monotonic()
        self._prune(now)
        window = self._clients[ip]

        while window and window[0] < now - self._window:
            window.popleft()

        if len(window) >= self._limit:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"error": "rate_limit_exceeded", "retry_after": self._window},
                headers={"Retry-After": str(self._window), **error_cors_headers(request.headers.get("origin"))},
            )

        window.append(now)

        try:
            response = await call_next(request)
        except Exception:
            # Same rationale as RequestContextMiddleware — don't let a
            # downstream exception escape this middleware raw.
            logger.exception('"%s %s" 500 (unhandled)', request.method, request.url.path)
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"},
                headers=error_cors_headers(request.headers.get("origin")),
            )

        response.headers["X-RateLimit-Limit"] = str(self._limit)
        response.headers["X-RateLimit-Remaining"] = str(max(0, self._limit - len(window)))
        return response
