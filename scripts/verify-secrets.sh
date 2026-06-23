#!/usr/bin/env bash
set -euo pipefail

errors=0

if [ -z "${DATABASE_URL:-}" ]; then
  echo "ERROR: DATABASE_URL is empty."
  errors=$((errors + 1))
fi

if [ -z "${ANTHROPIC_AUTH_TOKEN:-}" ] && [ -z "${Z_API_KEY:-}" ] && [ -z "${GLM_API_KEY:-}" ]; then
  echo "ERROR: Set ANTHROPIC_AUTH_TOKEN, Z_API_KEY, or GLM_API_KEY."
  errors=$((errors + 1))
fi

mode="${PIPELINE_MODE:-batch}"
override="${TICKERS_OVERRIDE:-}"
tickers_secret="${RECOMMENDATION_TICKERS:-}"

case "${override}" in
  "empty"|"EMPTY"|"null"|"NULL") override="" ;;
esac

if [ "${mode}" = "batch" ] || [ "${mode}" = "both" ]; then
  if [ -z "${override}" ] && [ -z "${tickers_secret}" ]; then
    echo "ERROR: Set RECOMMENDATION_TICKERS (all 22 — see tickers.example.txt)."
    errors=$((errors + 1))
  fi
fi

if [ ! -f "agent/write_recommendation_cache.py" ]; then
  echo "ERROR: agent/write_recommendation_cache.py missing — agent/ must be in this repo."
  errors=$((errors + 1))
fi

if [ "${errors}" -gt 0 ]; then
  exit 1
fi

echo "Secret checks passed."

mode="${PIPELINE_MODE:-batch}"
if [ "${mode}" = "batch" ] || [ "${mode}" = "both" ]; then
  bash scripts/verify-llm.sh
fi
