# Specmob API v2

Full rebuild against your real normalized schema (`phones` / `phone_specs` /
`phone_smart_scores` / `phone_variants` / `phone_images` / `phone_features` /
`price_points` / `price_history`). The previous API assumed a flat `phones`
table with every spec as a direct column — that no longer matches your DB
and would fail on `s.main_camera_mp`-style queries entirely.

## Layout

```
app/
  core/
    config.py          settings (env-driven)
    database.py         pool + row serialization + build_material repair
    sql_fragments.py    canonical phones+specs+scores JOIN, single source of truth
    query.py            composable WHERE-builder (search/filter/recommend all share this)
    scoring.py           chipset tier, value score, similarity score, priority weights
    shaping.py           smart_score nesting + computed field attachment
    cache.py             TTL+LRU cache (unchanged from before)
    middleware.py         request id + rate limiting (unchanged from before)
    ai_client.py          Gemini wrapper (unchanged behavior, now optional if no key)
  services/
    phone_repo.py         search, detail, similar, compare, latest, trending
    recommend_service.py  priority-weighted recommend with hard-filter budget widening
    recommend_copy.py     AI copy per recommended phone
    compare_copy.py        AI verdict for compare
  routes/
    phones.py, brands.py, categories.py, misc.py
models/
  phone.py                Pydantic response contracts
main.py                    app entrypoint
```

## What changed vs the old API

- **Schema-correct joins.** Every list/search/detail endpoint now joins
  `phones` + `phone_specs` + `phone_smart_scores` explicitly. No more
  guessed columns like `p.has_ois` or `p.is_premium_gaming` on `phones` —
  those live on `phone_specs` in your real DB.
- **Search**: typo-tolerant via `pg_trgm` similarity in addition to
  substring matching; brand/line alias expansion (`s24` -> `galaxy s24`,
  `iphone` -> `apple iphone`) is wider than before.
- **Filtering**: every filter param is validated against a real typed
  column — `ram_options`/`storage_options` filter via `unnest()` against
  the actual array columns, booleans (`has_nfc`, `has_ois`,
  `has_wireless_charging`, `is_foldable`, `is_premium_gaming`,
  `has_headphone_jack`) are real filters now, not missing entirely.
  Added `water_resistant`, `camera_setup_type`, `min_refresh_rate`,
  `min_antutu`, `min_storage`, multi-brand (`brands=Apple,Samsung`).
- **Similar phones**: multi-factor weighted score (price proximity, brand,
  chipset tier, camera/battery/screen closeness) instead of a plain
  price-band SQL ORDER BY. Pulls a candidate pool, re-ranks in Python.
- **Recommend**: every priority expression now references real
  `phone_specs`/`phone_smart_scores` columns. Hard filters (foldable,
  headphone_jack, nfc) gate results and progressively widen the price band
  only when needed; soft-only searches never widen (honest catalog
  signal). Value/match scores are true 0-10 (averaged, not summed).
- **Corrupted-field defense**: `build_material` in your DB is corrupted
  at the scrape/import stage — comma-joined single characters
  (`"G, l, a, s, s, ..."`). The database layer detects and repairs this
  transparently at read time (`core/database.py:repair_char_split`), but
  you should still fix it at the import script since the underlying data
  is broken, not just the display.
- **Categories**: added `foldables` and `water-resistant`, gaming category
  now boosts phones flagged `is_premium_gaming` in the legacy fallback
  formula.
- **Filter stats**: now reports `storage_options`, `antutu_range`,
  `refresh_rate_range`, and `chipset_tiers` counts, alongside the
  previous ranges.

## Endpoints

```
GET  /phones/search              full filter+search, see FilterParams for all params
GET  /phones/latest
GET  /phones/trending
GET  /phones/compare?ids=1,2,3   or ?slugs=a,b,c
GET  /phones/recommend?priorities=camera,battery&tier=b
GET  /phones/{id_or_slug}         detail + variants + images + features
GET  /phones/{id}/variants
GET  /phones/{id}/similar
GET  /phones/{id}/price-history
GET  /brands
GET  /brands/{name}
GET  /brands/{name}/phones
GET  /categories
GET  /categories/{slug}
GET  /filters/stats
GET  /sitemap.xml
GET  /health
```

## Setup

```bash
cp .env.example .env   # fill in DATABASE_URL, optionally GEMINI_API_KEY
pip install -r requirements.txt
./start.sh             # or: uvicorn main:app --reload
```

Requires the `pg_trgm` Postgres extension (auto-created on startup via
`CREATE EXTENSION IF NOT EXISTS pg_trgm` — needs `CREATEDB`/superuser-ish
privilege on first run; if your DB user lacks that, run it once manually
as a privileged user instead).

## Known upstream data issue to fix at the source

`phone_specs.build_material` is being written as a character-exploded
string (`",".join(list(original_string))`) somewhere in your scraper/import
pipeline for at least Apple devices. The API repairs it at read time, but
new imports will keep writing corrupted data until the import script
itself is fixed. Worth checking whatever step builds that field.
