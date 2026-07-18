from __future__ import annotations

from typing import Any

from .scoring import (
    chipset_tier_fallback,
    compute_value_score,
    normalize_popularity,
    resolve_tier,
)

_SMART_KEYS = (
    "smart_overall_score", "smart_camera_score", "smart_performance_score",
    "smart_battery_score", "smart_display_score", "smart_build_score",
    "smart_value_score", "smart_tier", "smart_reasoning", "smart_strengths",
    "smart_weaknesses", "smart_model_version", "smart_scored_at",
)


def pop_smart_score(d: dict) -> dict | None:
    """Extracts the flat smart_* columns produced by the JOIN into a
    nested `smart_score` object, mirroring the SmartScore Pydantic model.
    Mutates `d` in place (removes the flat keys)."""
    has_score = d.get("smart_overall_score") is not None
    out = None
    if has_score:
        out = {
            "overall_score": d.get("smart_overall_score"),
            "camera_score": d.get("smart_camera_score"),
            "performance_score": d.get("smart_performance_score"),
            "battery_score": d.get("smart_battery_score"),
            "display_score": d.get("smart_display_score"),
            "build_score": d.get("smart_build_score"),
            "value_score": d.get("smart_value_score"),
            "tier": d.get("smart_tier"),
            "reasoning": d.get("smart_reasoning"),
            "strengths": d.get("smart_strengths"),
            "weaknesses": d.get("smart_weaknesses"),
            "model_version": d.get("smart_model_version"),
            "scored_at": d.get("smart_scored_at"),
        }
    for k in _SMART_KEYS:
        d.pop(k, None)
    return out


def attach_computed_fields(phones: list[dict], peers: list[dict] | None = None) -> list[dict]:
    """Adds `chipset_tier` (dict: id+label) and `value_score` to each row.
    `value_score` prefers the AI smart_value_score, then a specs-per-dollar
    composite scored against `peers` (defaults to `phones` itself, which is
    correct for list endpoints where the page is a real comparison set, and
    wrong for a single-phone lookup — those callers must pass explicit peers).
    """
    peer_set = peers if peers is not None else phones
    for p in phones:
        p["chipset_tier"] = resolve_tier(p.get("smart_tier"), p.get("chipset"))
        p["popularity"] = normalize_popularity(p.get("popularity"))

        smart_value = p.get("smart_value_score")
        if smart_value is not None:
            p["value_score"] = round(float(smart_value), 1)
        elif p.get("value_score") is None:
            p["value_score"] = compute_value_score(p, peer_set)
    return phones


def finalize_phone_row(p: dict) -> dict:
    """Runs the full pop + attach pipeline for a single row and returns it.
    Convenience wrapper for endpoints handling one phone (detail) where the
    peer set for value_score has already been resolved separately."""
    p["smart_score"] = pop_smart_score(p)
    return p
