from __future__ import annotations

import logging

from . import market, query
from .currency import load_fx

logger = logging.getLogger(__name__)


async def load_reference_data(conn) -> dict:
    """Retailers, feature keys and FX rates live in the database. Each load is
    isolated: one failing never stops the API from starting."""
    out: dict = {}
    for name, fn in (("retailers", market.load_retailers), ("feature_keys", query.load_feature_keys), ("fx_rates", load_fx)):
        try:
            out[name] = await fn(conn)
        except Exception:
            logger.warning("Failed to load %s", name, exc_info=True)
            out[name] = None
    return out
