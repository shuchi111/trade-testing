# swing-trader-circleci-cron

**Self-contained public repo** for daily AI recommendations on **22 tickers** — no access to your private swing-trader repo required.

Includes the essential `agent/` Python code (~60 files) copied from swing-trader. CircleCI checks out **this repo only** and runs the cron.

---

## Quick start

1. Push this folder to a **public** GitHub repo.
2. Connect in [CircleCI](https://circleci.com).
3. Set secrets: `DATABASE_URL`, `RECOMMENDATION_TICKERS`, LLM keys — see [SETUP.md](./SETUP.md).
4. Schedule: cron `0 4 * * *`, `mode=batch`.
5. Disable GitHub Actions `schedule:` in your private swing-trader repo.

---

## What's included

### CI + scripts (repo root)

| Path | Purpose |
|------|---------|
| `.circleci/config.yml` | Pipeline |
| `scripts/run-batch.sh` | 22 tickers + execute trades |
| `scripts/run-price-refresh.sh` | Stale cache refresh |
| `scripts/verify-secrets.sh` | Env checks |
| `scripts/install-deps.sh` | `pip install -r agent/requirements.txt` |
| `tickers.example.txt` | 22-symbol line for CircleCI |

### Python agent (essential cron only)

| Path | Purpose |
|------|---------|
| `agent/write_recommendation_cache.py` | LLM recommendation → DB |
| `agent/execute_ai_trades.py` | Paper-trade executor |
| `agent/refresh_stale_recommendations.py` | Price-drift refresh |
| `agent/db_url.py` | Postgres URL helper |
| `agent/portfolio_db.py` | Holdings / trades |
| `agent/recommendation_bucket.py` | BUY/SELL/HOLD buckets |
| `agent/tradingagents/` | Full LangGraph agent package |
| `agent/requirements.txt` | Cron deps (no backtest/vectorbt) |

**Not included:** `backtest/`, `tests/`, `eval_results/` (not needed for cron).

---

## CircleCI secrets

| Variable | Required |
|----------|----------|
| `DATABASE_URL` | Yes |
| `RECOMMENDATION_TICKERS` | Yes (22 tickers) |
| `ANTHROPIC_AUTH_TOKEN` or `Z_API_KEY` | Yes |
| `ANTHROPIC_BASE_URL` | Usually yes |

No `MAIN_REPO_URL` or clone token — everything runs from this repo.

---

## Updating agent code

When you change agent logic in private swing-trader, copy updated files into this repo's `agent/` folder. See [AGENT-SYNC.md](./AGENT-SYNC.md).

---

Full setup: [SETUP.md](./SETUP.md)
