# trade-circleci-cron

Daily AI recommendations for 22 tickers via CircleCI.

## Setup

1. Push to GitHub (`main` branch).
2. Connect repo in [CircleCI](https://circleci.com).
3. Add env vars in Project Settings (see `env.example`):
   - `DATABASE_URL`
   - `RECOMMENDATION_TICKERS` (comma-separated, no spaces)
   - `ANTHROPIC_AUTH_TOKEN` or `Z_API_KEY` (+ `ANTHROPIC_BASE_URL` if needed)
4. Run pipeline manually once on `main` (leave `trade_date` empty — auto IST market day).
5. Schedule is in `.circleci/config.yml`: **Mon–Fri 10:00 IST** (`30 4 * * 1-5` UTC).

## Trade date

Leave `trade_date` empty in CircleCI — it uses **today in IST**. Saturday/Sunday roll back to Friday (NSE weekdays only; holidays not skipped).

## Pipeline modes

- `batch` — recommendations + paper trades (daily default)
- `price_refresh` — refresh stale cache rows
- `both` — batch then price refresh
