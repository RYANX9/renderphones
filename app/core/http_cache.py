from __future__ import annotations

import re

from starlette.middleware.base import BaseHTTPMiddleware

_RULES = [
    (re.compile(r"^/(admin|feedback|tradein|events|health)(/|$)"), "no-store"),
    (re.compile(r"^/$"), "no-store"),
    (re.compile(r"^/phones/slugs$"), "public, s-maxage=86400, stale-while-revalidate=86400"),
    (re.compile(r"^/phones/search-index$"), "public, max-age=300, s-maxage=3600, stale-while-revalidate=86400"),
    (re.compile(r"^/phones/resolve/[^/]+$"), "public, max-age=300, s-maxage=3600, stale-while-revalidate=86400"),
    (re.compile(r"^/phones/(search)$"), "public, max-age=60, s-maxage=300, stale-while-revalidate=1800"),
    (re.compile(r"^/phones/(compare|recommend)$|^/compare/pair/[^/]+$"), "public, max-age=60, s-maxage=300, stale-while-revalidate=600"),
    (re.compile(r"^/phones/(latest|upcoming|trending|top-rated)$"), "public, max-age=300, s-maxage=900, stale-while-revalidate=3600"),
    (re.compile(r"^/discover/[^/]+$"), "public, max-age=300, s-maxage=900, stale-while-revalidate=3600"),
    (re.compile(r"^/phones/[^/]+/offers$"), "public, max-age=60, s-maxage=600, stale-while-revalidate=1800"),
    (re.compile(r"^/phones/by-slug/[^/]+/[^/]+$"), "public, max-age=60, s-maxage=900, stale-while-revalidate=86400"),
    (re.compile(r"^/phones/[^/]+(/(full-specs|price-history|similar|variants|related))?$"), "public, max-age=60, s-maxage=3600, stale-while-revalidate=86400"),
    (re.compile(r"^/brands/[^/]+/phones$"), "public, max-age=60, s-maxage=300, stale-while-revalidate=1800"),
    (re.compile(r"^/(brands|categories)(/[^/]+)?$|^/filters/stats$"), "public, max-age=300, s-maxage=3600, stale-while-revalidate=86400"),
    (re.compile(r"^/seo/[^/]+$"), "public, s-maxage=3600, stale-while-revalidate=86400"),
]


class CacheControlMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        if request.method != "GET" or response.status_code != 200 or "Cache-Control" in response.headers:
            return response
        for pattern, value in _RULES:
            if pattern.match(request.url.path):
                response.headers["Cache-Control"] = value
                break
        return response
