from __future__ import annotations

from typing import Any

from .currency import to_local
from .insights import TIER_LABELS, compute_insights, short_best_for
from .slugs import phone_path

# Columns used only to compute labels; stripped from card payloads.
_INTERNAL = ("offer_scopes", "search_doc", "_total")


def shape_tier(row: dict) -> dict:
    tier = row.get("tier")
    row["tier_label"] = TIER_LABELS.get(tier) if tier else None
    return row


def shape_price(row: dict) -> dict:
    """Adds price_local when the price came from an offer in a non-USD region.
    USD (`price_usd`) is always present and always the value to display unless
    the viewer is in `price_region`."""
    basis = row.get("price_basis")
    row["price_local"] = (
        to_local(row.get("price_usd"), row.get("price_region")) if basis == "offer" else None
    )
    return row


def shape_card(row: dict, *, with_insight: bool = True) -> dict:
    shape_tier(row)
    shape_price(row)
    if row.get("brand_slug") and row.get("slug"):
        row["path"] = phone_path(row["brand_slug"], row["slug"])
    if with_insight:
        row["best_for"] = short_best_for(row)
    for k in _INTERNAL:
        row.pop(k, None)
    return row


def shape_cards(rows) -> list[dict]:
    return [shape_card(r) for r in rows]


def shape_mini(row: dict) -> dict:
    shape_tier(row)
    shape_price(row)
    row["path"] = phone_path(row["brand_slug"], row["slug"])
    return row


def shape_detail(row: dict) -> dict:
    shape_card(row, with_insight=False)
    row["insights"] = compute_insights(row)
    row["best_for"] = row["insights"]["best_for"][:2]
    return row


def build_analysis(row: dict[str, Any] | None) -> dict[str, Any] | None:
    """phone_analysis row -> API shape. None when the phone has no analysis."""
    if not row:
        return None
    return {
        "reasoning": row.get("reasoning"),
        "strengths": row.get("strengths") or [],
        "weaknesses": row.get("weaknesses") or [],
        "model_version": row.get("model_version"),
        "scored_at": row.get("scored_at"),
    }
