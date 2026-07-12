from __future__ import annotations

from typing import Any

from utils.gemini_client import call


_PRIORITY_LABEL = {
    "camera": "camera quality",
    "battery": "battery life",
    "performance": "performance",
    "compact": "compact size",
    "lightweight": "low weight",
    "display": "display quality",
    "fast_charging": "fast charging",
    "value": "best value",
}

_SCHEMA = {
    "type": "object",
    "properties": {
        "phones": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "match_line": {"type": "string"},
                    "tradeoff_line": {"type": "string"},
                },
                "required": ["id", "match_line", "tradeoff_line"],
            },
        }
    },
    "required": ["phones"],
}


def _phone_summary(p: dict[str, Any]) -> str:
    fields = [
        f"id={p['id']}",
        f"name={p.get('brand', '')} {p.get('model_name', '')}",
        f"price=${p['price_usd']:.0f}" if p.get("price_usd") else "price=unknown",
        f"camera={p['main_camera_mp']}MP" if p.get("main_camera_mp") else None,
        f"battery={p['battery_capacity']}mAh" if p.get("battery_capacity") else None,
        f"chipset={p['chipset']}" if p.get("chipset") else None,
        f"chipset_tier={p['chipset_tier']}" if p.get("chipset_tier") else None,
        f"charging={p['fast_charging_w']}W" if p.get("fast_charging_w") else None,
        f"screen={p['screen_size']}in" if p.get("screen_size") else None,
        f"weight={p['weight_g']}g" if p.get("weight_g") else None,
    ]
    return " | ".join(f for f in fields if f)


def generate_match_copy(
    phones: list[dict[str, Any]],
    priorities: list[str],
    budget_label: str,
) -> dict[int, dict[str, str]] | None:
    """One call covering every recommended phone. Returns
    {phone_id: {match_line, tradeoff_line}} or None if the call failed —
    caller falls back to the existing spec-based logic."""
    if not phones:
        return None

    priority_labels = [_PRIORITY_LABEL.get(p, p) for p in priorities]
    phone_lines = "\n".join(_phone_summary(p) for p in phones)

    prompt = f"""A shopper set a budget of {budget_label} and said these priorities matter most: {', '.join(priority_labels)}.

Here are the phones matched to them, one per line:
{phone_lines}

For each phone, write two short plain-sentence lines a shopper would read on a comparison card:
- match_line: one sentence (max 22 words) explaining why this phone fits their stated priorities, using its actual specs. Direct and concrete, no hedging, no filler like "this phone offers".
- tradeoff_line: one sentence (max 18 words) naming the clearest compromise versus their priorities or budget. If there's no real compromise, describe the weakest relative spec plainly.

Write in plain, confident, editorial tone. Never mention scores, algorithms, models, or that this text was generated."""

    result = call(prompt, _SCHEMA, temperature=0.5)
    if not result:
        return None

    out: dict[int, dict[str, str]] = {}
    for row in result.get("phones", []):
        pid = row.get("id")
        if pid is None:
            continue
        out[int(pid)] = {
            "match_line": row.get("match_line", ""),
            "tradeoff_line": row.get("tradeoff_line", ""),
        }
    return out or None
