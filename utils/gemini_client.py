from __future__ import annotations

import json
import logging
import os
from typing import Any

import requests

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AQ.Ab8RN6IahIF2BSG7yyACBKRu3mRThzhcbnuiGcv5fjLAdSeyUg")
MODEL_NAME = "gemini-3.1-flash-lite"  # fixed per requirement, do not change

_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"
_TIMEOUT_S = 8.0


def call(prompt: str, schema: dict[str, Any], *, temperature: float = 0.4) -> dict[str, Any] | None:
    """Single generateContent call constrained to `schema`. Returns the
    parsed JSON payload, or None on any failure (bad status, timeout,
    malformed response). Callers must always have a non-Gemini fallback
    since this is a best-effort enhancement, never a hard dependency."""
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "response_mime_type": "application/json",
            "response_schema": schema,
        },
    }
    try:
        resp = requests.post(
            _ENDPOINT,
            params={"key": GEMINI_API_KEY},
            json=body,
            timeout=_TIMEOUT_S,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(text)
    except Exception:
        logger.warning("Gemini call failed, falling back to static copy", exc_info=True)
        return None
