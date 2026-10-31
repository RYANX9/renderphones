from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_ORIGINS = (
    "https://specmob.com",
    "https://www.specmob.com",
    "https://specmob.vercel.app",
    "https://specmoburls.netlify.app",
)


def _split_origins(raw: str) -> list[str]:
    return [o.strip().lower().rstrip("/") for o in raw.split(",") if o.strip()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    database_url: str
    db_pool_min: int = 2
    db_pool_max: int = 15
    db_command_timeout: float = 30.0

    cache_ttl_trending: int = 900
    cache_ttl_stable: int = 3_600
    cache_ttl_phone_detail: int = 21_600

    rate_limit_requests: int = 180
    rate_limit_window: int = 60
    # Requests carrying this value in `X-Internal-Key` skip the rate limiter.
    # Set the same secret in the Next.js server env so SSR traffic (which all
    # arrives from shared Vercel egress IPs) is never throttled.
    internal_api_key: str = ""

    # Browser origins allowed to call the API. A literal "*" is ignored.
    cors_origins: str = ",".join(DEFAULT_ORIGINS)
    # Subset of the above that may only read (GET/HEAD/OPTIONS): the URL generator.
    read_only_origins: str = "https://specmoburls.netlify.app"
    # off | log | enforce. Use `log` first to see what would be refused.
    trusted_sources_mode: str = "enforce"
    debug: bool = False
    app_version: str = "4.0.0"
    admin_api_key: str = ""

    # ── analytics ──────────────────────────────────────────────────────────
    # Mixed into the daily session hash. Change it to invalidate every session.
    analytics_salt: str = "change-me"
    events_enabled: bool = True
    events_retention_days: int = 90
    # A compared pair needs this many unique sessions in 30 days before the API
    # calls it "popular" (shown in Most Compared) and marks it indexable.
    compare_popular_min_sessions: int = 10
    # Sessions needed before a pair is DISPLAYED (Most compared, People also compare).
    # Indexing still uses compare_popular_min_sessions.
    compare_display_min_sessions: int = 1
    # Below this many phones a category page is thin: API marks it noindex.
    category_min_phones_indexable: int = 5

    # ── background jobs ────────────────────────────────────────────────────
    scheduler_enabled: bool = True
    rollup_interval_minutes: int = 60
    category_refresh_interval_hours: int = 24
    fx_refresh_interval_minutes: int = 60

    @property
    def cors_origins_list(self) -> list[str]:
        return [o for o in _split_origins(self.cors_origins) if o != "*"] or list(DEFAULT_ORIGINS)

    @property
    def read_only_origins_list(self) -> list[str]:
        return _split_origins(self.read_only_origins)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
