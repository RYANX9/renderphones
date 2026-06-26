from __future__ import annotations

import re

_FLAGSHIP = re.compile(
    r"""
    snapdragon\s+8\s+elite              |
    snapdragon\s+8s\s+elite             |
    snapdragon\s+8\s+gen\s*[1-9]        |
    snapdragon\s+8s\s+gen\s*[1-9]       |
    dimensity\s+9[0-9]{3}               |
    exynos\s+2[0-9]{3}                  |
    apple\s+a1[4-9](?:\s+(?:bionic|pro))?   |
    apple\s+a[2-9][0-9](?:\s+(?:bionic|pro))?   |
    tensor\s+g[3-9]                     |
    kirin\s+9[0-9]{3}
    """,
    re.VERBOSE | re.IGNORECASE,
)

_MID = re.compile(
    r"""
    snapdragon\s+7                  |
    snapdragon\s+6                  |
    dimensity\s+[78][0-9]{2,3}      |
    exynos\s+1[0-9]{3}              |
    kirin\s+8[0-9]{2}               |
    tensor\s+g[12]
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
