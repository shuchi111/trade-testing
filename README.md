# trade-circleci-cron

Daily AI recommendations for 22 tickers via CircleCI. Self-contained — no private repo clone.

## Setup

1. Push to GitHub (`main` branch).
2. Connect repo in [CircleCI](https://circleci.com).
3. Add env vars in Project Settings (see `env.example`):
   - `DATABASE_URL`
   - `RECOMMENDATION_TICKERS` (comma-separated, no spaces)
   - `ANTHROPIC_AUTH_TOKEN` or `Z_API_KEY` (+ `ANTHROPIC_BASE_URL` if needed)
4. Run pipeline manually once, then add a schedule trigger: `0 4 * * *`, parameter `mode=batch`.
5. Turn off GitHub Actions `schedule:` in swing-trader if you still have it.

## Pipeline modes

- `batch` — recommendations + paper trades (daily default)
- `price_refresh` — refresh stale cache rows
- `both` — batch then price refresh
