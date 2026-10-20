"""Deterministic 'what is this phone really for' labels.

No AI, no network: derived from the phone's own score axes and a few spec
facts, so the same phone always gets the same label and it can be rendered on
the server for SEO. Scores are 0-10.
"""
from __future__ import annotations

from typing import Any

# Tie-break order when two axes have the same score (earlier wins).
_AXES = ("camera", "performance", "battery", "display", "build", "value", "durability")

BEST_FOR = {
    "camera": "great photos and video",
    "performance": "fast performance and gaming",
    "battery": "all-day battery life",
    "display": "a standout screen",
    "build": "a premium build",
    "value": "value for money",
    "durability": "durability",
}

NOT_FOR_IF = {
    "camera": "camera quality is your top priority",
    "performance": "you want heavy gaming or the fastest chip",
    "battery": "you need long battery life",
    "display": "you want a bright, smooth, high-end screen",
    "build": "you want a premium feel",
    "value": "you want the most for your money",
    "durability": "you need water and dust resistance",
}

TIER_LABELS = {
    "ultra_flagship": "Ultra Flagship",
    "flagship": "Flagship",
    "upper_mid_range": "Upper Mid-Range",
    "mid_range": "Mid-Range",
    "budget": "Budget",
}

STRONG = 7.5          # an axis must reach this to count as a strength
WEAK = 6.5            # ...and this or lower to count as a weakness
MIN_GAP = 1.5         # weakest axis must sit this far below the best one


def _f(v: Any) -> float | None:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _axis_scores(row: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for axis in _AXES:
        v = _f(row.get(f"{axis}_score"))
        if v is not None:
            out[axis] = v
    return out


def _spec_tags(row: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    weight, screen = _f(row.get("weight_g")), _f(row.get("screen_size"))
    charge = _f(row.get("fast_charging_w")) or 0
    if row.get("is_foldable"):
        tags.append("Foldable")
    if screen is not None and screen <= 6.3 and not row.get("is_foldable"):
        tags.append("Compact")
    if weight is not None and 80 <= weight <= 180 and not row.get("is_foldable"):
        tags.append("Lightweight")
    if charge >= 65:
        tags.append("Very fast charging")
    if row.get("has_wireless_charging"):
        tags.append("Wireless charging")
    if row.get("has_headphone_jack"):
        tags.append("Headphone jack")
    if row.get("is_premium_gaming"):
        tags.append("Gaming-focused")
    if row.get("is_water_resistant"):
        tags.append("Water resistant")
    return tags


def _join(labels: list[str]) -> str:
    if len(labels) <= 1:
        return "".join(labels)
    return ", ".join(labels[:-1]) + " and " + labels[-1]


def compute_insights(row: dict[str, Any]) -> dict[str, Any]:
    """{'best_for': [...], 'not_for_if': str|None, 'tags': [...], 'summary': str|None}"""
    scores = _axis_scores(row)
    tier = row.get("tier")
    tier_label = TIER_LABELS.get(tier or "")

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], _AXES.index(kv[0])))
    strong = [(a, s) for a, s in ranked if s >= STRONG][:2]
    best_axes = [a for a, _ in strong]
    best_for = [BEST_FOR[a] for a in best_axes]

    not_for = None
    if scores:
        worst_axis, worst = min(scores.items(), key=lambda kv: (kv[1], -_AXES.index(kv[0])))
        top = ranked[0][1]
        if worst <= WEAK and top - worst >= MIN_GAP and worst_axis not in best_axes:
            not_for = NOT_FOR_IF[worst_axis]

    summary = None
    if best_for:
        noun = f"{tier_label.lower()} phone" if tier_label else "phone"
        article = "An" if noun[0] in "aeiou" else "A"
        summary = f"{article} {noun} best for {_join(best_for)}."
    elif tier_label:
        summary = f"A {tier_label.lower()} phone with no standout strength."

    return {"best_for": best_for, "not_for_if": not_for, "tags": _spec_tags(row), "summary": summary}


def short_best_for(row: dict[str, Any]) -> list[str]:
    """Card-sized label (max 2 phrases)."""
    return compute_insights(row)["best_for"]


# ── per-category one-line reasons (for ranked lists) ────────────────────────

def _n(v: Any, unit: str = "") -> str | None:
    f = _f(v)
    if f is None:
        return None
    return f"{int(f) if f == int(f) else f}{unit}"


_CAMERA_LABELS = {"main": "main sensor", "telephoto": "telephoto", "video": "video", "ultrawide": "ultrawide", "selfie": "selfie camera"}


def category_reason(slug: str, row: dict[str, Any]) -> str | None:
    """One line explaining the ranking. Uses the stored score breakdown so the
    text always describes what was actually scored (never megapixels)."""
    bd = row.get("breakdown") or {}
    if slug == "camera-phones":
        subs = [(k, _f(bd.get(k))) for k in _CAMERA_LABELS]
        subs = sorted([(k, v) for k, v in subs if v is not None and v >= 6.0], key=lambda kv: -kv[1])[:2]
        if subs:
            return ", ".join(f"{_CAMERA_LABELS[k]} {v:.1f}/10" for k, v in subs)
        parts = ["OIS" if row.get("has_ois") else None, "telephoto" if row.get("has_telephoto") else None]
        return ", ".join(p for p in parts if p) or None
    if slug == "battery-life":
        h, wh = _f(bd.get("est_hours")), _f(bd.get("wh"))
        if h is not None:
            return f"~{h:g} h estimated screen-on ({wh:g} Wh)" if wh else f"~{h:g} h estimated screen-on"
        return _n(row.get("battery_capacity"), " mAh battery")
    if slug == "fast-charging":
        g = _f(bd.get("hours_gained"))
        w = _n(row.get("fast_charging_w"), "W")
        return f"{w}, about +{g:g} h in 15 min" if g is not None and w else w
    if slug == "gaming-phones":
        a, r = _f(row.get("antutu_score")), _n(row.get("refresh_rate_hz"), "Hz display")
        src = bd.get("performance_source")
        perf = f"AnTuTu {int(a):,}" if a else ("chipset-class performance" if src == "chipset_median" else None)
        return ", ".join(p for p in (perf, r) if p) or None
    if slug == "lightweight":
        return _n(bd.get("weight_min_g") or row.get("weight_g"), " g")
    if slug == "compact-phones":
        return _n(row.get("screen_size"), '" screen')
    if slug == "foldables":
        return "Foldable form factor"
    if slug == "water-resistant":
        return row.get("water_resistance")
    if slug.startswith("under-"):
        p = _f(row.get("price_usd"))
        est = " (estimated price)" if bd.get("price_estimated") else ""
        return f"From ${p:,.0f}{est}" if p else None
    return None


# The axis that matters for each category list; its label goes first in best_for.
CATEGORY_AXIS = {"camera-phones": "camera", "gaming-phones": "performance", "battery-life": "battery",
                 "fast-charging": "battery", "water-resistant": "durability"}


def best_for_for_category(slug: str, best_for: list[str], row: dict[str, Any]) -> list[str]:
    """In a category list, lead with that category's strength when the phone has it
    (axis score >= STRONG), even if other axes scored higher. Max 2 labels."""
    axis = CATEGORY_AXIS.get(slug)
    label = BEST_FOR.get(axis) if axis else None
    score = _f(row.get("durability_score" if axis == "durability" else f"{axis}_score")) if axis else None
    if label and score is not None and score >= STRONG:
        return ([label] + [b for b in best_for if b != label])[:2]
    return best_for