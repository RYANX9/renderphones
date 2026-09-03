from __future__ import annotations

SCREEN_POINTS = {
    "perfect": 30,
    "minor_scratches": 22,
    "deep_scratches": 14,
    "cracked_touch_ok": 6,
    "cracked_unresponsive": 0,
}

BODY_POINTS = {
    "flawless": 20,
    "light_wear": 15,
    "moderate_wear": 10,
    "heavy_wear": 5,
    "cracked_back": 0,
}

FUNCTIONAL_DEDUCTIONS = {
    "camera": 7,
    "biometric": 6,
    "audio": 5,
    "charging_port": 5,
    "buttons": 4,
}

BATTERY_NON_ORIGINAL_PENALTY = 4

BRAND_TIER_1 = {"apple"}
BRAND_TIER_2 = {"google", "oneplus"}

CONDITION_TIERS: tuple[tuple[float, float, str], ...] = (
    (90, 100, "excellent"),
    (70, 89, "good"),
    (45, 69, "fair"),
    (0, 44, "poor"),
)

# deduction[condition_tier][price_bucket] = (low_pct, high_pct)
DEDUCTION_TABLE: dict[str, dict[str, tuple[float, float]]] = {
    "excellent": {"s": (0.08, 0.14), "ab": (0.06, 0.12), "cd": (0.04, 0.10)},
    "good":      {"s": (0.18, 0.26), "ab": (0.16, 0.24), "cd": (0.15, 0.22)},
    "fair":      {"s": (0.32, 0.42), "ab": (0.30, 0.40), "cd": (0.28, 0.38)},
    "poor":      {"s": (0.45, 0.60), "ab": (0.48, 0.62), "cd": (0.55, 0.70)},
}

PRICE_TIER_TO_BUCKET = {"s": "s", "a": "ab", "b": "ab", "c": "cd", "d": "cd"}


def score_screen(condition: str) -> int:
    return SCREEN_POINTS.get(condition, 0)


def score_body(condition: str) -> int:
    return BODY_POINTS.get(condition, 0)


def score_battery(health: int, non_original: bool) -> float:
    health = max(0, min(10, health))
    raw = health * 2.5
    if non_original:
        raw -= BATTERY_NON_ORIGINAL_PENALTY
    return max(0.0, raw)


def score_functional(broken_components: list[str]) -> int:
    score = 25
    for component in broken_components:
        score -= FUNCTIONAL_DEDUCTIONS.get(component, 0)
    return max(0, score)


def brand_bonus(brand: str, model_name: str) -> int:
    b = brand.lower().strip()
    if b in BRAND_TIER_1:
        return 10
    if b == "samsung":
        model_lower = model_name.lower()
        return 10 if ("ultra" in model_lower or " s2" in model_lower or " s1" in model_lower) else 5
    if b in BRAND_TIER_2:
        return 5
    return 0


def condition_tier_for_score(normalized: float) -> str:
    for lo, hi, tier in CONDITION_TIERS:
        if lo <= normalized <= hi:
            return tier
    return "poor"


def price_bucket_for_tier(price_tier_id: str) -> str:
    return PRICE_TIER_TO_BUCKET.get(price_tier_id, "cd")


def estimate_trade_in(
    *,
    baseline_price: float,
    brand: str,
    model_name: str,
    price_tier_id: str,
    screen_condition: str,
    body_condition: str,
    battery_health: int,
    battery_non_original: bool,
    broken_components: list[str],
) -> dict:
    screen = score_screen(screen_condition)
    body = score_body(body_condition)
    battery = score_battery(battery_health, battery_non_original)
    functional = score_functional(broken_components)
    bonus = brand_bonus(brand, model_name)

    raw_total = screen + body + battery + functional + bonus
    normalized = min(raw_total, 100.0)

    tier = condition_tier_for_score(normalized)
    bucket = price_bucket_for_tier(price_tier_id)
    low_pct, high_pct = DEDUCTION_TABLE[tier][bucket]

    return {
        "score_breakdown": {
            "screen": screen,
            "body": body,
            "battery": round(battery, 1),
            "functional": functional,
            "brand_bonus": bonus,
            "raw_total": round(raw_total, 1),
            "normalized": round(normalized, 1),
        },
        "condition_tier": tier,
        "price_tier_bucket": bucket,
        "deduction_range": {"low_pct": low_pct, "high_pct": high_pct},
        "estimated_range": {
            "low": round(baseline_price * (1 - high_pct), 2),
            "high": round(baseline_price * (1 - low_pct), 2),
        },
    }
