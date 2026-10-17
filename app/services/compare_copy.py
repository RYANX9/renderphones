from __future__ import annotations

from typing import Any

from ..core.ai_client import call

_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string"},
        "picks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "for_label": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["id", "for_label", "reason"],
            },
        },
    },
    "required": ["verdict", "picks"],
}


def _phone_summary(p: dict[str, Any]) -> str:
    fields = [
        f"id={p['id']}",
        f"name={p.get('display_name') or (str(p.get('brand', '')) + ' ' + str(p.get('name', '')))}",
        f"price=${p['price_usd']:.0f}" if p.get("price_usd") else "price=unknown",
        f"camera={p['main_camera_mp']}MP" if p.get("main_camera_mp") else None,
        f"battery={p['battery_capacity']}mAh" if p.get("battery_capacity") else None,
        f"antutu={p['antutu_score']}" if p.get("antutu_score") else None,
        f"chipset={p['chipset']}" if p.get("chipset") else None,
        f"charging={p['fast_charging_w']}W" if p.get("fast_charging_w") else None,
        f"weight={p['weight_g']}g" if p.get("weight_g") else None,
        f"overall={p['overall_score']}/10" if p.get("overall_score") is not None else None,
    ]
    return " | ".join(f for f in fields if f)


def _display_name(p: dict[str, Any]) -> str:
    return (p.get("display_name") or f"{p.get('brand', '')} {p.get('name', '')}").strip()


def generate_compare_verdict(phones: list[dict[str, Any]], winner_id: int | None = None) -> dict[str, Any] | None:
    if len(phones) < 2:
        return None

    phone_lines = "\n".join(_phone_summary(p) for p in phones)
    winner = next((p for p in phones if p["id"] == winner_id), None) if winner_id is not None else None

    if winner is not None:
        winner_name = _display_name(winner)
        pick_line = f"\nOur pick is {winner_name} (id={winner['id']}). This is already decided.\n"
        verdict_rule = (
            f"- verdict: 2-3 sentences (max 55 words total) that recommend the {winner_name} as the pick, "
            "name what it wins on using actual specs, and say where the others compensate. "
            "Never recommend a different phone overall."
        )
    else:
        pick_line = ""
        verdict_rule = (
            "- verdict: 2-3 sentences (max 55 words total) giving a direct, holistic comparison — which phone pulls ahead "
            "overall and on what specific strengths, and where the others compensate. Reference actual specs, not generic praise."
        )

    prompt = f"""A shopper is comparing these {len(phones)} phones side by side:
{phone_lines}
{pick_line}
Write:
{verdict_rule}
- picks: for each phone, one short label (2-4 words, e.g. "Best value", "Best camera", "Longest battery life") describing who it's the right choice for, and one sentence (max 18 words) reason grounded in its specs.

Plain, confident, editorial tone. Never mention scores, algorithms, models, or that this text was generated."""

    result = call(prompt, _SCHEMA, temperature=0.5)
    if result is not None:
        result["winner_id"] = winner_id
    return result
