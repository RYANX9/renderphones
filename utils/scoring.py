from __future__ import annotations

import re

_FLAGSHIP = re.compile(
    r"""
    # Snapdragon 8 / 8+ / 8s — Elite or Gen N
    snapdragon \s+ 8 [+s]? \s* (?: elite | gen \s* \d+ )
    # Qualcomm model numbers: SM8150 (SD855) → SM8750 (SD8 Elite)
    | \b sm8[1-9]\d{2} \b
    # MediaTek Dimensity 9xxx
    | dimensity \s+ 9\d{3}
    # MediaTek Dimensity 9xxx model numbers (MT6989=9300, MT6991=9400)
    | \b mt699\d \b
    # Samsung Exynos 2xxx
    | exynos \s+ 2\d{3}
    # Apple A14 and above (bionic / pro / max suffix optional)
    | apple \s+ a (?: 1[4-9] | [2-9]\d ) (?: \s+ (?: bionic | pro | max ) )?
    # Google Tensor G3+
    | tensor \s+ g [3-9]
    # Huawei / HiSilicon Kirin 9xxx
    | kirin \s+ 9\d{3}
    """,
    re.VERBOSE | re.IGNORECASE,
)

_MID = re.compile(
    r"""
    # Snapdragon 7 / 7+ / 7s Gen N (requires "Gen" — excludes 730, 720, etc.)
    snapdragon \s+ 7 [+s]? \s* gen
    # Snapdragon 6 Gen N only (excludes 695, 690, 680 — those are entry)
    | snapdragon \s+ 6 \s+ gen
    # Qualcomm SM7xxx model numbers
    | \b sm7[1-6]\d{2} \b
    # MediaTek Dimensity 7xxx / 8xxx
    | dimensity \s+ [78]\d{2,3}
    # MediaTek 7/8xx model numbers (MT67xx, MT68xx)
    | \b mt6[78]\d{2} \b
    # Samsung Exynos 1xxx
    | exynos \s+ 1\d{3}
    # Huawei Kirin 8xx
    | kirin \s+ 8\d{2}
    # Google Tensor G1 / G2
    | tensor \s+ g [12]
    # Apple A10–A13
    | apple \s+ a1[0-3] (?: \s+ (?: bionic | pro ) )?
    """,
    re.VERBOSE | re.IGNORECASE,
)


def chipset_tier(chipset: str | None) -> str:
    if not chipset:
        return "unknown"
    if _FLAGSHIP.search(chipset):
        return "flagship"
    if _MID.search(chipset):
        return "mid"
    return "entry"


def compute_value_score(phone: dict, peers: list[dict]) -> float | None:
    if not phone.get("price_usd") or not peers:
        return None

    def spec_score(p: dict) -> float:
        s = 0.0
        if p.get("antutu_score"):     s += min(p["antutu_score"] / 2_000_000, 1.0) * 3.0
        if p.get("main_camera_mp"):   s += min(p["main_camera_mp"] / 200, 1.0) * 2.0
        if p.get("battery_capacity"): s += min(p["battery_capacity"] / 7_000, 1.0) * 2.0
        if p.get("ram_options"):      s += min(max(p["ram_options"]) / 16, 1.0) * 1.5
        if p.get("fast_charging_w"):  s += min(p["fast_charging_w"] / 100, 1.0) * 1.0
        if p.get("screen_size"):      s += min(p["screen_size"] / 7.0, 1.0) * 0.5
        return s

    this = spec_score(phone)
    peer_scores = [spec_score(p) for p in peers if p.get("price_usd")]
    if not peer_scores:
        return None
    peak = max(peer_scores) or 1.0
    return round(min(this / peak * 10, 10.0), 1)
