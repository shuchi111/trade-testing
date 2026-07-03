# trade-circleci-cron

Daily AI recommendations for 30 tickers via GitHub Actions.

## Thresholds & defaults

All trading limits, transaction charges, thesis-break rules, LLM defaults, and scheduler settings are documented in **[thresholds.md](thresholds.md)**.

## Setup

1. Push to GitHub (`main` branch).
2. Enable GitHub Actions for the repository.
3. Add repository secrets (see `env.example`):
   - `DATABASE_URL`
   - `RECOMMENDATION_TICKERS` (comma-separated, no spaces)
   - `ANTHROPIC_AUTH_TOKEN`, `Z_API_KEY`, or `GLM_API_KEY`
4. Test once: GitHub → **Actions** → **AI recommendations (cache)** → **Run workflow** on `main` (leave `trade_date` empty).
5. GitHub Actions schedule runs automatically **Mon-Fri 9:00 AM IST** (`30 3 * * 1-5` UTC). Optional **3:00 PM IST** run is commented in the workflow — uncomment when needed.

## When the pipeline runs

| Trigger | Runs batch jobs? | Notes |
|---------|------------------|-------|
| Git push to `main` | **No** | Workflow is scheduled/manual only |
| GitHub Actions schedule (9:00 AM IST weekdays) | **Yes** | Runs `.github/workflows/ai-recommendations.yml` from the default branch |
| **Run workflow** in GitHub Actions UI | **Yes** | Use `mode=batch` or `mode=both` |
| Repository dispatch `ai-recommendation` | **Yes** | Can run one dispatched ticker payload |

GitHub scheduled workflows run only from the default branch and can start a few minutes later than the exact cron time.

**GitHub Actions schedule:** `30 3 * * 1-5` UTC = 9:00 AM IST, Monday-Friday. Optional `30 9 * * 1-5` UTC = 3:00 PM IST (commented in workflow).

See [thresholds.md](thresholds.md) for full cron and trigger details.

## Trade date

Leave `trade_date` empty in GitHub Actions — it uses **today in IST**. Saturday/Sunday roll back to Friday (NSE weekdays only; holidays not skipped).

## Pipeline modes

- `batch` — recommendations + paper trades (daily default)
- `price_refresh` — refresh stale cache rows
- `both` — batch then price refresh
