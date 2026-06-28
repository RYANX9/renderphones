from __future__ import annotations

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    database_url: str

    db_pool_min: int = 2
    db_pool_max: int = 10
    db_command_timeout: float = 30.0

    cache_ttl_trending: int = 900
    cache_ttl_stable: int = 3_600
    cache_ttl_phone_detail: int = 86_400

    rate_limit_requests: int = 120
    rate_limit_window: int = 60

    cors_origins: str = "https://mobylite.vercel.app"

    debug: bool = False
    app_version: str = "2.0.0"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# Module-level singleton — routes import this directly: `from config import settings`
settings = get_settings()
