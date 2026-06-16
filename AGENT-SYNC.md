# Keeping agent/ in sync with private swing-trader

This repo contains a **copy** of the essential cron files from `swing-trader/agent/`.
When you update agent logic in the private repo, refresh these paths here.

## Files to copy (from swing-trader/agent/)

```
write_recommendation_cache.py
execute_ai_trades.py
refresh_stale_recommendations.py
db_url.py
portfolio_db.py
recommendation_bucket.py
tradingagents/          (entire folder, except data_cache/*.csv and __pycache__)
```

## PowerShell one-liner (run from TradeAgent root)

```powershell
$src = "swing-trader\agent"
$dst = "swing-trader-circleci-cron\agent"
@("write_recommendation_cache.py","execute_ai_trades.py","refresh_stale_recommendations.py","db_url.py","portfolio_db.py","recommendation_bucket.py") | ForEach-Object { Copy-Item "$src\$_" "$dst\$_" -Force }
robocopy "$src\tradingagents" "$dst\tradingagents" /E /XD data_cache __pycache__ /NFL /NDL
```

Then commit and push the public cron repo.

## Not copied (not needed for cron)

- `backtest/` — VectorBT backtests
- `tests/` — unit tests
- `run_propagate.py` — local CLI only
- `eval_results/` — runtime logs

## requirements.txt

Cron uses a trimmed `agent/requirements.txt` (no vectorbt/supabase/numba).
If you add new Python deps in swing-trader agent code, update both files.
