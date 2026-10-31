"""Who may call this API.

A request is accepted when it is one of:
  1. server-to-server with the shared secret in `X-Internal-Key` (Next.js SSR/ISR/build);
  2. from a browser whose `Origin` is in the allowlist (CORS_ORIGINS);
  3. a health probe or an /admin route (the admin routes check X-Admin-Key themselves).

Anything else gets 403 with no CORS headers. Origins listed in READ_ONLY_ORIGINS
(the URL generator) may only read: any write from them is refused, so testing
links can never create events, feedback or trade-in rows.

The Origin header is set by browsers and cannot be forged by a page, so this
stops other websites from using the API. A non-browser client can send any
Origin it likes; only the internal key is a real secret.
"""
from __future__ import annotations

import hmac
import logging
from typing import Iterable

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from .config import settings

logger = logging.getLogger(__name__)

SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
DEFAULT_EXEMPT_PATHS = frozenset({"/", "/health", "/health/db"})
MODES = ("off", "log", "enforce")


def normalize_origin(value: str) -> str:
    return value.strip().lower().rstrip("/")


def error_cors_headers(origin: str | None) -> dict[str, str]:
    """CORS headers for error responses built outside CORSMiddleware: only a
    trusted origin is echoed back, never a wildcard."""
    if origin and normalize_origin(origin) in settings.cors_origins_list:
        return {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
            "Vary": "Origin",
        }
    return {}


class TrustedSourceMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        *,
        origins: Iterable[str],
        read_only_origins: Iterable[str] = (),
        internal_key: str = "",
        mode: str = "enforce",
        exempt_paths: Iterable[str] = DEFAULT_EXEMPT_PATHS,
        exempt_prefixes: Iterable[str] = ("/admin/",),
    ) -> None:
        self.app = app
        self._origins = frozenset(normalize_origin(o) for o in origins)
        self._read_only = frozenset(normalize_origin(o) for o in read_only_origins)
        self._key = internal_key.encode()
        self._mode = mode if mode in MODES else "enforce"
        self._exempt_paths = frozenset(exempt_paths) | {"/admin"}
        self._exempt_prefixes = tuple(exempt_prefixes)

    def _reason(self, method: str, headers: Headers) -> str | None:
        if self._key:
            supplied = headers.get("x-internal-key")
            if supplied and hmac.compare_digest(supplied.encode(), self._key):
                return None

        origin = headers.get("origin")
        if origin is None:
            return "no_origin_no_key" if self._key else None

        origin = normalize_origin(origin)
        if origin not in self._origins:
            return "untrusted_origin"
        if origin in self._read_only:
            intended = headers.get("access-control-request-method", method) if method == "OPTIONS" else method
            if intended.upper() not in SAFE_METHODS:
                return "read_only_origin"
        return None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or self._mode == "off":
            await self.app(scope, receive, send)
            return

        path = scope["path"]
        if path in self._exempt_paths or path.startswith(self._exempt_prefixes):
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        reason = self._reason(scope["method"], headers)
        if reason is None:
            await self.app(scope, receive, send)
            return

        logger.warning(
            "trusted-source %s: %s %s origin=%s mode=%s",
            reason, scope["method"], path, headers.get("origin"), self._mode,
        )
        if self._mode == "log":
            await self.app(scope, receive, send)
            return

        response = JSONResponse({"detail": "Forbidden"}, status_code=403, headers={"Vary": "Origin"})
        await response(scope, receive, send)
