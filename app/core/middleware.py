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

logger = logging.getLogger(__name__)

_RATE_LIMIT_SKIP = frozenset({"/", "/health", "/docs", "/redoc", "/openapi.json"})


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        rid = uuid.uuid4().hex[:10]
        t0 = time.monotonic()
        request.state.request_id = rid

        response = await call_next(request)

        elapsed_ms = (time.monotonic() - t0) * 1_000
        response.headers["X-Request-ID"] = rid
        response.headers["X-Response-Time"] = f"{elapsed_ms:.1f}ms"

        logger.info(
            '"%s %s" %d %.1fms rid=%s',
            request.method, request.url.path, response.status_code, elapsed_ms, rid,
        )
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, *, requests: int, window: int) -> None:
        super().__init__(app)
        self._limit = requests
        self._window = window
        self._clients: dict[str, deque[float]] = defaultdict(deque)

    def _client_ip(self, request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in _RATE_LIMIT_SKIP:
            return await call_next(request)

        ip = self._client_ip(request)
        now = time.monotonic()
        window = self._clients[ip]

        while window and window[0] < now - self._window:
            window.popleft()

        if len(window) >= self._limit:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"error": "rate_limit_exceeded", "retry_after": self._window},
                headers={"Retry-After": str(self._window)},
            )

        window.append(now)
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self._limit)
        response.headers["X-RateLimit-Remaining"] = str(max(0, self._limit - len(window)))
        return response
