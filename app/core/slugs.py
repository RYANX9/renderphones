"""URL and slug helpers. Every public path the API hands out is built here so
the frontend and the API can never disagree about URL shape.

Canonical shapes (these must match the Next.js routes):
    /brand/{brand_slug}/{slug}        phone
    /brand/{brand_slug}               brand
    /best/{category_slug}             category
    /compare/{full_slug}-vs-{full_slug}   compare (full slugs, sorted)
"""
from __future__ import annotations

import re

PAIR_SEPARATOR = "-vs-"
MAX_COMPARE = 4


def brand_slug_of(value: str) -> str:
    """Normalise user/URL input to a brand slug."""
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")


def phone_path(brand_slug: str, slug: str) -> str:
    return f"/brand/{brand_slug}/{slug}"


def brand_path(brand_slug: str) -> str:
    return f"/brand/{brand_slug}"


def category_path(category_slug: str) -> str:
    return f"/best/{category_slug}"


def sorted_full_slugs(full_slugs: list[str]) -> list[str]:
    return sorted(dict.fromkeys(full_slugs))


def pair_slug(full_slugs: list[str]) -> str:
    """Canonical compare slug: unique full slugs, alphabetically sorted."""
    return PAIR_SEPARATOR.join(sorted_full_slugs(full_slugs))


def compare_path(full_slugs: list[str]) -> str:
    return f"/compare/{pair_slug(full_slugs)}"


def split_pair_slug(value: str) -> list[str]:
    parts = [p.strip().lower() for p in value.split(PAIR_SEPARATOR)]
    return [p for p in parts if p]
