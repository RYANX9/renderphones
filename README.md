# Specmob API

FastAPI backend powering [Specmob](https://mobylite.vercel.app) — a phone search, comparison, and recommendation service. Handles filtering, ranking, trade-in valuation, and price history over a Postgres catalog of phone specs.

## Features

- **Faceted search** — filter by brand, chipset, price, RAM, storage, battery, camera, screen size, refresh rate, AnTuTu score, and boolean feature flags (NFC, OIS, wireless charging, foldable, water resistance, etc.), with typo-tolerant matching via Postgres trigram similarity (`pg_trgm`).
- **Recommendation engine** — tiered scoring (`s`/`a`/`b`/`c`/`d` price bands) with adaptive range widening when a query returns nothing, plus point-price padding for narrow/dial-drag queries.
- **Comparison** — side-by-side spec comparison with generated verdict copy.
- **Trade-in estimator** — condition-based valuation (screen/body condition, battery health, non-original battery, broken components) against the phone's current market price.
- **Catalog browsing** — brands, categories, trending, latest, similar phones, variants, and per-phone price history.
- **Operational hardening** — request-context middleware, sliding-window rate limiting, tiered response caching (trending vs. stable vs. detail pages), global exception handling, and a generated sitemap.

## Tech stack

- **FastAPI** + **Uvicorn**
- **asyncpg** — Postgres connection pooling, no ORM
- **Pydantic v2** / **pydantic-settings** for config and validation
- Postgres extension: `pg_trgm`

## API overview

| Area | Endpoints |
|---|---|
| Phones | `GET /phones/search`, `/phones/latest`, `/phones/trending`, `/phones/compare`, `/phones/recommend`, `/phones/{id}`, `/phones/{id}/full-specs`, `/phones/{id}/variants`, `/phones/{id}/similar`, `/phones/{id}/price-history` |
| Brands | `GET /brands`, `/brands/{brand_name}`, `/brands/{brand_name}/phones` |
| Categories | `GET /categories`, `/categories/{category_slug}` |
| Trade-in | `POST /tradein/estimate` |
| System | `GET /filters/stats`, `/sitemap.xml`, `/health`, `POST /history/views` |

## Project structure

```
app/
  core/       config, db pool, caching, rate limiting, query building, scoring, trade-in logic
  models/     phone data model
  routes/     phones, brands, categories, tradein, misc
  services/   phone repository, recommendation service, compare/recommend copy generation
main.py       FastAPI app, middleware, router wiring
start.sh      uvicorn entrypoint
```

## Getting started

**Requirements:** Python 3.11+, a Postgres database.

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```
DATABASE_URL=postgresql://user:password@host:5432/dbname
CORS_ORIGINS=*
DEBUG=false
```

Run the server:

```bash
./start.sh
# or
uvicorn main:app --reload
# or
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

The API is served at `http://localhost:8000`, with interactive docs at `/docs`.

## Notes

This service is the data layer for the [Specmob](https://github.com/RYANX9/Specmob) frontend — the frontend calls it directly for search, comparison, recommendations, and trade-in pricing.
