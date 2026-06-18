# trade-circleci-cron

Daily AI recommendations for 22 tickers via CircleCI.

## Thresholds & defaults

All trading limits, transaction charges, thesis-break rules, LLM defaults, and CircleCI settings are documented in **[thresholds.md](thresholds.md)**.

## Setup

1. Push to GitHub (`main` branch).
2. Connect repo in [CircleCI](https://circleci.com).
3. Add env vars in Project Settings (see `env.example`):
   - `DATABASE_URL`
   - `RECOMMENDATION_TICKERS` (comma-separated, no spaces)
   - `Z_API_KEY` or `GLM_API_KEY` (+ `LLM_PROVIDER=glm`, see `env.example`)
4. Test once: CircleCI → **Pipelines** → **Trigger Pipeline** on `main` (leave `trade_date` empty).
5. Cron runs automatically **Mon–Fri 10:30 IST** (`0 5 * * 1-5` UTC) — **not on git push**.

## When the pipeline runs

| Trigger | Runs batch jobs? | Notes |
|---------|------------------|-------|
| Git push to `main` | **No** | Workflows filtered by `pipeline.trigger_source`; push = `webhook` |
| Cron schedule (10:30 IST weekdays) | **Yes** | `scheduled-ai-recommendations` workflow only |
| **Trigger Pipeline** in CircleCI UI | **Yes** | `manual-ai-recommendations` workflow only (`api` trigger) |

Push may still create an empty pipeline entry in CircleCI (no jobs). That is expected.

**Cron:** `0 5 * * 1-5` UTC = 10:30 IST, Monday–Friday only. Do not add a second schedule in Project Settings → Triggers unless you want duplicate daily runs.

See [thresholds.md](thresholds.md) for full cron and trigger details.

## Trade date

Leave `trade_date` empty in CircleCI — it uses **today in IST**. Saturday/Sunday roll back to Friday (NSE weekdays only; holidays not skipped).

## Pipeline modes

- `batch` — recommendations + paper trades (daily default)
- `price_refresh` — refresh stale cache rows
- `both` — batch then price refresh
