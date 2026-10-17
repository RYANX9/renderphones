from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, Field

from ..core.database import acquire_conn
from ..services import analytics

router = APIRouter(prefix="/events", tags=["events"])


class EventIn(BaseModel):
    type: str
    page_type: Optional[str] = None
    path: Optional[str] = Field(default=None, max_length=300)
    phone_id: Optional[int] = None
    target: Optional[str] = Field(default=None, max_length=300)
    slot: Optional[str] = Field(default=None, max_length=80)
    value: Optional[int] = None
    referrer: Optional[str] = Field(default=None, max_length=500)


class EventBatch(BaseModel):
    client_id: Optional[str] = Field(default=None, max_length=64)
    region: Optional[str] = Field(default=None, max_length=2)
    referrer: Optional[str] = Field(default=None, max_length=500)
    events: list[EventIn] = Field(default_factory=list, max_length=25)


class CompareIn(BaseModel):
    client_id: Optional[str] = Field(default=None, max_length=64)
    phone_ids: list[int] = Field(min_length=2, max_length=4)
    entry: Optional[str] = None


@router.post("", status_code=202)
async def post_events(batch: EventBatch, request: Request) -> Response:
    """Fire-and-forget. Always 202: analytics must never break a page, so
    failures are logged and swallowed."""
    try:
        async with acquire_conn() as conn:
            await analytics.record_events(
                conn, request, client_id=batch.client_id, region=batch.region, referrer=batch.referrer,
                events=[e.model_dump() for e in batch.events],
            )
    except Exception:
        import logging
        logging.getLogger(__name__).warning("event write failed", exc_info=True)
    return Response(status_code=202)


@router.post("/compare", status_code=202)
async def post_compare(payload: CompareIn, request: Request) -> Response:
    try:
        async with acquire_conn() as conn:
            await analytics.record_compare(conn, request, client_id=payload.client_id,
                                           phone_ids=payload.phone_ids, entry=payload.entry)
    except Exception:
        import logging
        logging.getLogger(__name__).warning("compare event write failed", exc_info=True)
    return Response(status_code=202)
