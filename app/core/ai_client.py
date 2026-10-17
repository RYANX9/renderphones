# app/core/ai_client.py
from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed, Future, TimeoutError as FutureTimeoutError
from typing import Any

import requests

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL_NAME = "gemini-3.1-flash-lite"
_GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent"

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
_OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
_OPENROUTER_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
]

_TIMEOUT_S = 8.0
_RACE_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="ai-race")


def _schema_conforms(result: Any, schema: dict[str, Any]) -> bool:
    """A quota/safety block from Gemini often comes back as HTTP 200 with
    valid-but-wrong-shape JSON, not an exception. json.loads succeeding is
    not the same as "this is the object we asked for" -- check the
    top-level required keys are actually present before trusting it."""
    if not isinstance(result, dict):
        return False
    return all(key in result for key in schema.get("required", []))


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
            result = json.loads(text)
            if _schema_conforms(result, schema):
                return result
            logger.warning("OpenRouter model %s returned a non-conforming body", model)
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

        candidates = resp.json().get("candidates") or []
        if not candidates:
            logger.warning("Gemini returned zero candidates (quota or safety block)")
            return None

        text = candidates[0]["content"]["parts"][0]["text"]
        result = json.loads(text)
        if not _schema_conforms(result, schema):
            logger.warning("Gemini returned a non-conforming body")
            return None
        return result
    except Exception:
        logger.warning("Gemini call failed", exc_info=True)
        return None


def call(prompt: str, schema: dict[str, Any], *, temperature: float = 0.4) -> dict[str, Any] | None:
    """Fires OpenRouter and Gemini concurrently, returns whichever comes
    back first with a schema-conforming body. A provider that answers
    fast with garbage doesn't win the race -- it's rejected and the loop
    keeps waiting on whatever's left. Every caller must have its own
    deterministic fallback since this can return None."""
    futures: dict[Future, str] = {}
    if OPENROUTER_API_KEY:
        futures[_RACE_EXECUTOR.submit(_call_openrouter, prompt, schema, temperature)] = "openrouter"
    if GEMINI_API_KEY:
        futures[_RACE_EXECUTOR.submit(_call_gemini, prompt, schema, temperature)] = "gemini"

    if not futures:
        logger.warning("No AI provider configured")
        return None

    result: dict[str, Any] | None = None
    try:
        for future in as_completed(futures, timeout=_TIMEOUT_S + 2.0):
            provider = futures[future]
            try:
                candidate = future.result()
            except Exception:
                logger.warning("%s race future raised", provider, exc_info=True)
                continue
            if candidate is not None:
                result = candidate
                break
    except FutureTimeoutError:
        logger.warning("Both AI providers exceeded the race timeout")

    if result is None:
        logger.warning("All AI providers failed or returned non-conforming bodies")
    return result
