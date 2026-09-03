# app/core/ai_client.py
from __future__ import annotations

import json
import logging
import os
from typing import Any

import requests

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL_NAME = "gemini-3.1-flash-lite"
_GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent"

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
_OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
# Ordered by accuracy then speed, per benchmark against this app's actual
# prompt shapes. First model that returns valid JSON wins.
_OPENROUTER_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
]

_TIMEOUT_S = 8.0


def _call_openrouter(prompt: str, schema: dict[str, Any], temperature: float) -> dict[str, Any] | None:
    if not OPENROUTER_API_KEY:
        return None
    for model in _OPENROUTER_MODELS:
        try:
            resp = requests.post(
                _OPENROUTER_ENDPOINT,
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"name": "response", "schema": schema},
                    },
                },
                timeout=_TIMEOUT_S,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
            return json.loads(text)
        except Exception:
            logger.warning("OpenRouter model %s failed", model, exc_info=True)
            continue
    return None


def _call_gemini(prompt: str, schema: dict[str, Any], temperature: float) -> dict[str, Any] | None:
    if not GEMINI_API_KEY:
        return None
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "response_mime_type": "application/json",
            "response_schema": schema,
        },
    }
    try:
        resp = requests.post(_GEMINI_ENDPOINT, params={"key": GEMINI_API_KEY}, json=body, timeout=_TIMEOUT_S)
        resp.raise_for_status()
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(text)
    except Exception:
        logger.warning("Gemini call failed", exc_info=True)
        return None


def call(prompt: str, schema: dict[str, Any], *, temperature: float = 0.4) -> dict[str, Any] | None:
    """OpenRouter (free models) first, Gemini second, None third. Always
    best-effort — every caller must have a deterministic fallback, since
    AI copy generation can fail or be disabled entirely."""
    result = _call_openrouter(prompt, schema, temperature)
    if result is not None:
        return result
    result = _call_gemini(prompt, schema, temperature)
    if result is not None:
        return result
    logger.warning("All AI providers failed, falling back to deterministic copy")
    return None
