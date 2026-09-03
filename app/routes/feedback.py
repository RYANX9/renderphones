from __future__ import annotations

import re

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field, field_validator

from ..core.database import get_pool

router = APIRouter(prefix="/feedback", tags=["feedback"])

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class FeedbackIn(BaseModel):
    message: str = Field(..., min_length=1, max_length=10_000)
    email: str | None = Field(default=None, max_length=255)
    page_url: str | None = Field(default=None, max_length=2048)
    website: str | None = None  # honeypot -- real users never see/fill this

    @field_validator("email")
    @classmethod
    def _validate_email(cls, v: str | None) -> str | None:
        if v is None or v.strip() == "":
            return None
        v = v.strip()
        if not _EMAIL_RE.match(v):
            raise ValueError("invalid email address")
        return v

    @field_validator("message")
    @classmethod
    def _clean_message(cls, v: str) -> str:
        return v.strip()


@router.post("", status_code=201)
async def submit_feedback(payload: FeedbackIn, request: Request):
    if payload.website:
        # Honeypot tripped -- pretend to succeed, drop it silently.
        return {"status": "ok"}

    if not payload.message:
        return {"status": "ok"}

    async with get_pool().acquire() as conn:
        await conn.execute(
            """
            INSERT INTO feedback (message, email, page_url, user_agent)
            VALUES ($1, $2, $3, $4)
            """,
            payload.message,
            payload.email,
            payload.page_url,
            request.headers.get("user-agent"),
        )

    return {"status": "ok"}
