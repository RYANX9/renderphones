from __future__ import annotations

import re
from typing import Any

from . import category_config as cfg

Row = dict[str, Any]

_TIER_WORDS = frozenset({"ultra", "pro", "max", "plus", "xl", "fe", "lite", "neo", "se", "mini"})
_NOISE = frozenset({"5g", "4g"})
_LOW = frozenset({"fe", "lite", "neo", "se", "mini"})

_EDITION = [
    re.compile(r"\(\s*(china|global|international|india|japan|cn)\s*(edition|version|variant)?\s*\)", re.I),
    re.compile(r"\b(cn|global|china)\s+(edition|version)\b", re.I),
]


def _tokens(name: Any, brand: Any) -> list[str]:
    n = (name or "").lower().strip()
    b = (brand or "").lower().strip()
    if b and n.startswith(b + " "):
        n = n[len(b) + 1:]
    n = re.sub(r"\(.*?\)", " ", n).replace("+", " plus ")
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", n)


def classify(name: Any, brand: Any) -> tuple[tuple[str, str], int]:
    toks = _tokens(name, brand)
    t = set(toks)
    if "ultra" in t:
        tier = 5
    elif "max" in t or ("pro" in t and "plus" in t):
        tier = 4
    elif "pro" in t:
        tier = 3
    elif "plus" in t:
        tier = 2
    elif t & _LOW:
        tier = 0
    else:
        tier = 1
    family = " ".join(x for x in toks if x not in _TIER_WORDS and x not in _NOISE)
    return ((brand or "").strip().lower(), family), tier


def _ts(row: Row) -> float | None:
    try:
        v = float(row.get("release_ts"))
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def _near(a: float | None, b: float | None) -> bool:
    return a is None or b is None or abs(a - b) <= cfg.LINEUP_WINDOW_DAYS * 86400


def apply_guard(items: list[dict], rows_by_id: dict[Any, Row]) -> None:
    groups: dict[tuple[str, str], list[dict]] = {}
    for it in items:
        row = rows_by_id[it["phone_id"]]
        key, tier = classify(row.get("name"), row.get("brand"))
        it["_tier"], it["_ts"] = tier, _ts(row)
        groups.setdefault(key, []).append(it)

    for members in groups.values():
        if len({m["_tier"] for m in members}) < 2:
            continue
        members.sort(key=lambda m: m["_tier"])
        for i, m in enumerate(members):
            lower = [
                o for o in members[:i]
                if o["_tier"] < m["_tier"] and _near(o["_ts"], m["_ts"])
            ]
            if not lower:
                continue
            ref = max(lower, key=lambda o: o["_raw_score"])
            floor = min(10.0, ref["_raw_score"] + cfg.LINEUP_MARGIN)
            if m["_raw_score"] < floor:
                m["breakdown"] = {
                    **m["breakdown"],
                    "lineup_lift": {
                        "from": round(m["_raw_score"], 2),
                        "over": ref["phone_id"],
                    },
                }
                m["_raw_score"] = floor


def _canonical(name: Any) -> str:
    n = name or ""
    for p in _EDITION:
        n = p.sub("", n)
    return re.sub(r"\s{2,}", " ", n).strip().lower()


def prefer_global(rows: list[Row]) -> list[Row]:
    best: dict[tuple[str, str], Row] = {}
    for r in rows:
        key = ((r.get("brand") or "").strip().lower(), _canonical(r.get("name")))
        cur = best.get(key)
        if cur is None or (
            cur.get("is_regional_edition") and not r.get("is_regional_edition")
        ):
            best[key] = r
    return list(best.values())
