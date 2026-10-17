from __future__ import annotations

import math
import re
import statistics
from datetime import datetime, timezone
from typing import Any, Callable

from . import category_config as cfg
from . import lineup

Row = dict[str, Any]
Scored = dict[str, Any]
Scorer = Callable[[Row, dict, dict], "Scored | None"]

_A = cfg.ANCHORS
_INF = float("inf")


def _f(v: Any) -> float | None:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def scale(
    v: Any,
    lo: float,
    hi: float,
    *,
    log: bool = False,
    invert: bool = False,
) -> float | None:
    x = _f(v)
    if x is None:
        return None

    if log:
        x, lo, hi = math.log(max(x, 1e-9)), math.log(lo), math.log(hi)

    t = min(1.0, max(0.0, (x - lo) / (hi - lo)))
    return 1.0 - t if invert else t


def flag(value: Any, if_true: float | None) -> float | None:
    """None is unknown and is dropped; False is a real absence and scores zero."""
    if value is None:
        return None
    return if_true if value else 0.0


def blend(parts: list[tuple[float | None, float]]) -> tuple[float | None, float]:
    total = sum(w for _, w in parts)

    known = [(v, w) for v, w in parts if v is not None]

    if not known:
        return None, 0.0

    covered = sum(w for _, w in known)

    return (
        sum(v * w for v, w in known) / covered,
        covered / total,
    )


def parse_weight_min(
    text: str | None,
    fallback: Any = None,
) -> float | None:
    lo, hi = cfg.WEIGHT_VALID_RANGE
    candidates: list[float] = []

    if text:
        m = re.match(r"^\s*([\d.\s/,]+?)\s*g\b", text)

        if m:
            candidates = [
                float(n)
                for n in re.findall(
                    r"\d+(?:\.\d+)?",
                    m.group(1),
                )
            ]

    valid = [w for w in candidates if lo <= w <= hi]

    if valid:
        return min(valid)

    fb = _f(fallback)

    return (
        fb
        if fb is not None and lo <= fb <= hi
        else None
    )


def display_area_cm2(
    diagonal_in: float,
    resolution: str | None,
) -> float:
    ratio = 2.2

    if resolution:
        m = re.search(
            r"(\d+)\s*x\s*(\d+)",
            resolution,
            re.I,
        )

        if m:
            a, b = float(m.group(1)), float(m.group(2))

            if a > 0 and b > 0:
                ratio = max(a, b) / min(a, b)

    return (
        diagonal_in ** 2
        * ratio
        / (1 + ratio ** 2)
        * 6.4516
    )


def _energy_wh(row: Row) -> float | None:
    cap = _f(row.get("battery_capacity"))

    return (
        cap * cfg.CELL_VOLTAGE / 1000
        if cap
        else None
    )


def _draw_w(row: Row) -> tuple[float, float] | None:
    size = _f(row.get("screen_size"))

    if not size:
        return None

    area = display_area_cm2(
        size,
        row.get("screen_resolution"),
    )

    return (
        cfg.P0_W + cfg.K_W_PER_CM2 * area,
        area,
    )


def _subscores(
    row: Row,
    keys: tuple[str, ...],
) -> tuple[float | None, float]:
    return blend([
        (
            None
            if _f(row.get(k)) is None
            else _f(row[k]) / 10,
            1.0,
        )
        for k in keys
    ])


def _ratio(v: Any) -> float | None:
    x = _f(v)

    return None if x is None else x / 10


def _camera(
    row: Row,
    ctx: dict,
    spec: dict,
) -> Scored | None:
    area = _f(row.get("main_sensor_area_mm2"))
    ap = _f(row.get("main_aperture"))

    light = (
        scale(
            area / ap ** 2,
            *_A["main_light"],
        )
        if area and ap
        else None
    )

    main, _ = blend([
        (light, 0.8),
        (flag(row.get("has_ois"), 1.0), 0.2),
    ])

    tele_inner, _ = blend([
        (
            scale(
                row.get("optical_zoom_x"),
                *_A["zoom_x"],
                log=True,
            ),
            0.5,
        ),
        (
            flag(
                row.get("is_periscope"),
                1.0,
            ),
            0.2,
        ),
        (
            scale(
                row.get("telephoto_mp"),
                *_A["mp"],
            ),
            0.3,
        ),
    ])

    video, _ = blend([
        (
            scale(
                row.get("video_max_4k_fps"),
                *_A["fps_4k"],
            ),
            0.5,
        ),
        (
            flag(
                row.get("has_8k"),
                1.0,
            ),
            0.25,
        ),
        (
            flag(
                row.get("has_dolby_vision"),
                1.0,
            ),
            0.25,
        ),
    ])

    subs = {
        "main": main,
        "telephoto": flag(
            row.get("has_telephoto"),
            tele_inner,
        ),
        "video": video,
        "ultrawide": flag(
            row.get("has_ultrawide"),
            scale(
                row.get("ultrawide_mp"),
                *_A["mp"],
            ),
        ),
        "selfie": scale(
            row.get("selfie_mp"),
            *_A["mp"],
        ),
    }

    value, coverage = blend([
        (subs[k], w)
        for k, w in cfg.CAMERA_WEIGHTS.items()
    ])

    if value is None:
        return None

    return {
        "spec": value,
        "ai": _ratio(row.get("camera_score")),
        "completeness": coverage,
        "breakdown": {
            k: None
            if v is None
            else round(v * 10, 2)
            for k, v in subs.items()
        },
    }


def _battery(
    row: Row,
    ctx: dict,
    spec: dict,
) -> Scored | None:
    wh = _energy_wh(row)
    draw = _draw_w(row)

    if wh is None or draw is None:
        return None

    hours = wh / draw[0]

    return {
        "spec": scale(
            hours,
            *_A["battery_hours"],
        ),
        "ai": None,
        "completeness": 1.0,
        "breakdown": {
            "est_hours": round(hours, 1),
            "wh": round(wh, 1),
            "estimated": True,
        },
    }


def _fast_charging(
    row: Row,
    ctx: dict,
    spec: dict,
) -> Scored | None:
    watts = _f(row.get("fast_charging_w"))
    wh = _energy_wh(row)
    draw = _draw_w(row)

    if (
        not watts
        or watts < cfg.FAST_CHARGING_MIN_W
        or wh is None
        or draw is None
    ):
        return None

    wh_in = min(
        watts
        * cfg.CHARGE_EFFICIENCY
        * cfg.CHARGE_WINDOW_H,
        wh * cfg.CHARGE_TAPER_CAP,
    )

    gained = wh_in / draw[0]

    return {
        "spec": scale(
            gained,
            *_A["charge_hours_gained"],
        ),
        "ai": None,
        "completeness": 1.0,
        "breakdown": {
            "hours_gained": round(gained, 1),
            "wh_in_window": round(wh_in, 1),
            "estimated": True,
        },
    }


def _lightweight(
    row: Row,
    ctx: dict,
    spec: dict,
) -> Scored | None:
    weight = parse_weight_min(
        row.get("weight_text"),
        row.get("weight_g"),
    )

    screen = _f(row.get("screen_size"))

    if weight is None or weight > cfg.WEIGHT_CEILING_G:
        return None

    if screen is None or screen < cfg.LIGHTWEIGHT_MIN_SCREEN:
        return None

    if (
        _f(row.get("overall_score")) or 0
    ) < cfg.LIGHTWEIGHT_MIN_OVERALL:
        return None

    band = math.floor(
        weight / cfg.WEIGHT_BAND_G
    )

    return {
        "spec": scale(
            band * cfg.WEIGHT_BAND_G,
            *_A["weight_g"],
            invert=True,
        ),
        "ai": None,
        "completeness": 1.0,
        "breakdown": {
            "weight_min_g": weight,
            "weight_band": band,
        },
    }


def _compact(
    row: Row,
    ctx: dict,
    spec: dict,
) -> Scored | None:
    screen = _f(row.get("screen_size"))
    lo, hi = cfg.COMPACT_SCREEN_RANGE

    if (
        screen is None
        or not lo <= screen <= hi
    ):
        return None

    quality, q_cov = _subscores(
        row,
        (
            "performance_score",
            "battery_score",
            "display_score",
            "build_score",
        ),
    )

    weight = parse_weight_min(
        row.get("weight_text"),
        row.get("weight_g"),
    )

    footprint, _ = blend([
        (
            scale(
                screen,
                *_A["compact_screen"],
                invert=True,
            ),
            0.6,
        ),
        (
            scale(
                weight,
                *_A["compact_weight"],
                invert=True,
            ),
            0.4,
        ),
    ])

    value, coverage = blend([
        (quality, 0.75),
        (footprint, 0.25),
    ])

    if value is None:
        return None

    return {
        "spec": value,
        "ai": None,
        "completeness": min(
            coverage,
            q_cov,
        ),
        "breakdown": {
            "quality": None
            if quality is None
            else round(quality * 10, 2),
            "footprint": None
            if footprint is None
            else round(footprint * 10, 2),
        },
    }


def _gaming(
    row: Row,
    ctx: dict,
    spec: dict,
) -> Scored | None:
    antutu = _f(row.get("antutu_score"))
    source = "antutu"

    if antutu is None and row.get("chipset"):
        antutu = ctx["chipset_antutu"].get(
            row["chipset"]
        )
        source = "chipset_median"

    perf = scale(
        antutu,
        *_A["antutu"],
    )

    if perf is None:
        perf = _ratio(
            row.get("performance_score")
        )
        source = "ai_performance"

    ram = scale(
        row.get("ram_max_gb"),
        *_A["ram_gb"],
    )

    parts = [
        (
            perf,
            cfg.GAMING_WEIGHTS[
                "performance"
            ],
        ),
        (
            ram,
            cfg.GAMING_WEIGHTS["ram"],
        ),
        (
            scale(
                row.get("refresh_rate_hz"),
                *_A["refresh_hz"],
            ),
            cfg.GAMING_WEIGHTS[
                "refresh"
            ],
        ),
        (
            flag(
                row.get("is_premium_gaming"),
                1.0,
            ),
            cfg.GAMING_WEIGHTS[
                "premium"
            ],
        ),
    ]

    value, coverage = blend(parts)

    if value is None or perf is None:
        return None

    return {
        "spec": value,
        "ai": _ratio(
            row.get("performance_score")
        ),
        "completeness": coverage,
        "breakdown": {
            "performance_source": source
        },
    }


def _overall(row: Row) -> Scored | None:
    v = _ratio(row.get("overall_score"))

    if v is None:
        return None

    return {
        "spec": v,
        "ai": None,
        "completeness": 1.0,
        "breakdown": {},
    }


def _foldable(
    row: Row,
    ctx: dict,
    spec: dict,
) -> Scored | None:
    return (
        _overall(row)
        if row.get("is_foldable") is True
        else None
    )


def _water(
    row: Row,
    ctx: dict,
    spec: dict,
) -> Scored | None:
    m = re.search(
        r"IP\s?(\d)(\d)",
        row.get("water_resistance") or "",
        re.I,
    )

    if (
        not m
        or (
            int(m.group(1)),
            int(m.group(2)),
        ) < cfg.WATER_MIN_IP
    ):
        return None

    return _overall(row)


def _budget(
    row: Row,
    ctx: dict,
    spec: dict,
) -> Scored | None:
    price = _f(row.get("price_usd"))

    # Placeholder prices are not real market prices.
    # Do not use them for "under $X" categories.
    if row.get("price_basis") == "placeholder":
        return None

    if (
        price is None
        or price <= 0
        or price <= spec.get("price_min", 0.0)
        or price > spec["price_limit"]
    ):
        return None

    quality, coverage = _subscores(
        row,
        (
            "performance_score",
            "battery_score",
            "display_score",
            "build_score",
        ),
    )

    if quality is None:
        return None

    return {
        "spec": quality,
        "ai": None,
        "completeness": coverage,
        "breakdown": {
            "price_estimated": False,
        },
    }


SCORERS: dict[str, Scorer] = {
    "camera": _camera,
    "battery": _battery,
    "gaming": _gaming,
    "fast_charging": _fast_charging,
    "lightweight": _lightweight,
    "compact": _compact,
    "foldable": _foldable,
    "water": _water,
    "budget": _budget,
}


def build_context(
    rows: list[Row],
) -> dict:
    by_chipset: dict[str, list[float]] = {}

    for r in rows:
        a = _f(r.get("antutu_score"))

        if a and r.get("chipset"):
            by_chipset.setdefault(
                r["chipset"],
                [],
            ).append(a)

    return {
        "chipset_antutu": {
            k: statistics.median(v)
            for k, v in by_chipset.items()
        }
    }



def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _release_ts(row: Row) -> float | None:
    return _f(row.get("release_ts"))


def _status(row: Row) -> str:
    return str(row.get("status") or "").strip().lower()


def is_category_eligible(row: Row, now: datetime | None = None) -> bool:
    """Allow only current/previous calendar-year releases, plus upcoming phones."""
    now = now or _now_utc()
    status = _status(row)
    ts = _release_ts(row)

    if cfg.INCLUDE_UPCOMING and status == "upcoming":
        return True

    if ts is None:
        return False

    start = datetime(now.year - cfg.RELEASE_YEARS_BACK, 1, 1, tzinfo=timezone.utc)
    return start.timestamp() <= ts <= now.timestamp()


def _release_bucket(row: Row, now: datetime) -> int:
    """2=current year, 1=previous eligible year, 0=upcoming/unknown."""
    ts = _release_ts(row)
    if ts is None or _status(row) == "upcoming":
        return 0
    year = datetime.fromtimestamp(ts, tz=timezone.utc).year
    if year == now.year:
        return 2
    if year == now.year - 1:
        return 1
    return 0


def _freshness_bonus(row: Row, now: datetime) -> float:
    return cfg.CURRENT_YEAR_FRESHNESS_BONUS if _release_bucket(row, now) == 2 else 0.0


def _estimate_missing_ai(ai_values: list[tuple[float, float]], spec_value: float) -> float | None:
    """Estimate missing AI scores using the median AI-vs-spec offset in this pool."""
    if not ai_values:
        return None
    offsets = [ai - spec for spec, ai in ai_values]
    estimate = spec_value + statistics.median(offsets)
    return max(0.0, min(1.0, estimate))

def _sort_key(spec: dict, item: dict) -> tuple:
    price = item["_price"]

    if spec.get("order") == "weight_band":
        # Weight band remains the primary purpose of the category.
        # Within the same 5g band, the 60% physical-lightness score and
        # 40% category-quality/AI score decide the order.
        return (
            item["breakdown"]["weight_band"],
            -item["_raw_score"],
            -item["_overall"],
            _INF if price is None else price,
            -item["phone_id"],
        )

    return (
        -item["_raw_score"],
        -item.get("_release_bucket", 0),
        -item["_tier"],
        -item["_overall"],
        _INF if price is None else -price,
        -item["phone_id"],
    )


def _apply_unique_display_scores(
    items: list[dict],
) -> None:
    """
    Make the displayed category_score unique when two or more phones
    have the same two-decimal score.

    Important:
    - The real/raw score is preserved in _raw_score.
    - Only tied two-decimal display scores are separated.
    - Existing ranking rules determine who gets the higher displayed score.
    - Maximum movement is normally only a few hundredths.
    """

    used: set[float] = set()

    # Items are already in their correct ranking order.
    for item in items:
        base = round(
            item["_raw_score"],
            2,
        )

        score = base

        # Move downward by 0.01 until this displayed
        # score is unique.
        while score in used:
            score = round(
                score - 0.01,
                2,
            )

        item["category_score"] = score
        used.add(score)


def rank_category(
    slug: str,
    rows: list[Row],
    limit: int | None = cfg.CATEGORY_STORE_LIMIT,
) -> list[dict]:
    spec = cfg.CATEGORY_CONFIG[slug]
    scorer = SCORERS[spec["scorer"]]
    now = _now_utc()

    # Hard eligibility gate. This is intentionally repeated here even though
    # CANDIDATE_SQL also filters, because rank_category may be called directly
    # by jobs/tests with an arbitrary row list.
    rows = [r for r in rows if is_category_eligible(r, now)]
    rows = lineup.prefer_global(rows)
    ctx = build_context(rows)
    ai_w = float(spec.get("ai_weight", 0.0) or 0.0)

    scored: list[tuple[Row, Scored]] = []
    for row in rows:
        res = scorer(row, ctx, spec)
        if res is None or res["spec"] is None or res["completeness"] < cfg.MIN_COMPLETENESS:
            continue
        scored.append((row, res))

    complete = [res["spec"] for _, res in scored if res["completeness"] >= 0.999]
    prior = statistics.median(complete) if complete else None

    # Build the reference set for consistent missing-AI estimation.
    ai_reference: list[tuple[float, float]] = []
    for _, res in scored:
        ai = res.get("ai")
        if ai is not None and res["spec"] is not None:
            ai_reference.append((float(res["spec"]), float(ai)))

    items: list[dict] = []
    for row, res in scored:
        c = float(res["completeness"])
        base = float(res["spec"])
        if c < 1.0 and prior is not None:
            base = c * base + (1.0 - c) * prior

        # Category-specific AI signal. The scorer can provide an explicit AI
        # value; otherwise use the field configured for this category.
        raw_ai = res.get("ai")
        if raw_ai is None and spec.get("ai_field"):
            raw_ai = _ratio(row.get(spec["ai_field"]))

        ai = raw_ai
        ai_estimated = False
        if ai is None and ai_w:
            ai = _estimate_missing_ai(ai_reference, base)
            ai_estimated = ai is not None

        if ai is None or not ai_w:
            final = base
        else:
            final = (1.0 - ai_w) * base + ai_w * ai

        score10 = final * 10.0
        freshness = _freshness_bonus(row, now)
        score10 = min(10.0, score10 + freshness)

        release_bucket = _release_bucket(row, now)
        breakdown = dict(res["breakdown"])
        if ai_estimated:
            breakdown["ai_estimated"] = True
        if freshness:
            breakdown["freshness_bonus"] = round(freshness, 2)

        items.append({
            "phone_id": row["id"],
            "category_score": None,
            "spec_part": round(base * 10.0, 2),
            "ai_part": None if ai is None else round(ai * 10.0, 2),
            "completeness": round(c, 2),
            "breakdown": breakdown,
            "_raw_score": score10,
            "_tier": 0,
            "_ts": _release_ts(row),
            "_release_bucket": release_bucket,
            "_overall": _f(row.get("overall_score")) or 0.0,
            "_price": _f(row.get("price_usd")),
        })

    if spec.get("lineup_guard"):
        lineup.apply_guard(items, {r["id"]: r for r in rows})

    items.sort(key=lambda i: _sort_key(spec, i))
    _apply_unique_display_scores(items)

    # Enforce the configured brand cap in the stored ranking as well as at the
    # API display layer. This prevents a high-volume brand from consuming the
    # whole category pool before the endpoint applies its own cap.
    cap = int(cfg.CATEGORY_BRAND_CAP)
    if cap > 0:
        kept: list[dict] = []
        brand_counts: dict[str, int] = {}
        for item in items:
            row = next((r for r in rows if r["id"] == item["phone_id"]), None)
            brand = str((row or {}).get("brand") or "").strip().lower()
            if brand_counts.get(brand, 0) >= cap:
                continue
            brand_counts[brand] = brand_counts.get(brand, 0) + 1
            kept.append(item)
            if limit is not None and len(kept) >= limit:
                break
    else:
        kept = items if limit is None else items[:limit]

    for rank, item in enumerate(kept, start=1):
        item["rank"] = rank
        for k in ("_raw_score", "_tier", "_ts", "_release_bucket", "_overall", "_price"):
            item.pop(k, None)

    return kept

