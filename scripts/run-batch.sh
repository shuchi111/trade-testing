#!/usr/bin/env bash
# Full batch: loops RECOMMENDATION_TICKERS (22). Uses agent/ in this repo.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [ ! -d "agent" ]; then
  echo "ERROR: agent/ directory missing at ${ROOT_DIR}/agent"
  exit 1
fi

override="${TICKERS_OVERRIDE:-}"
trade_date_input="${TRADE_DATE_INPUT:-}"
recommendation_tickers="${RECOMMENDATION_TICKERS:-}"

if [ -n "${override}" ]; then
  TICKERS="${override}"
  SOURCE="circleci_manual"
else
  TICKERS="${recommendation_tickers}"
  if [ -n "${trade_date_input}" ]; then
    SOURCE="circleci_manual"
  else
    SOURCE="circleci_batch"
  fi
fi

if [ -z "${trade_date_input}" ]; then
  DATE="$(date -u +'%Y-%m-%d')"
else
  DATE="${trade_date_input}"
fi

TICKERS="$(echo "${TICKERS}" | tr -d '[:space:]')"

if [ -z "${TICKERS}" ]; then
  echo "ERROR: No tickers. Set RECOMMENDATION_TICKERS in CircleCI."
  exit 1
fi

echo "=== Batch run ==="
echo "Trade date: ${DATE}"
echo "Source tag: ${SOURCE}"
echo "Tickers:    ${TICKERS}"
echo "Repo root:  ${ROOT_DIR}"
echo

COUNT=0
OK=0
FAILED=()

IFS=',' read -ra ARR <<< "${TICKERS}"
for raw in "${ARR[@]}"; do
  sym="$(echo "${raw}" | xargs)"
  [ -z "${sym}" ] && continue
  COUNT=$((COUNT + 1))
  echo "--- [${COUNT}] ${sym} | trade_date=${DATE} | source=${SOURCE} ---"
  if python agent/write_recommendation_cache.py \
      --ticker "${sym}" \
      --trade-date "${DATE}" \
      --source "${SOURCE}" \
      --from-portfolio; then
    OK=$((OK + 1))
  else
    FAILED+=("${sym}")
    echo "WARN: Recommendation failed for ${sym} (continuing batch)"
  fi
done

echo
echo "--- AI paper-trade executor ---"
python agent/execute_ai_trades.py --all --trade-date "${DATE}" \
  || echo "WARN: execute_ai_trades.py exited non-zero"

echo
echo "=== Batch finished ==="
echo "Tickers attempted: ${COUNT}"
echo "Tickers succeeded: ${OK}"

if [ ${#FAILED[@]} -gt 0 ]; then
  echo "Failed tickers (${#FAILED[@]}): ${FAILED[*]}"
fi

if [ "${OK}" -eq 0 ] && [ "${COUNT}" -gt 0 ]; then
  echo "ERROR: All tickers failed"
  exit 1
fi

if [ "${COUNT}" -ne 22 ] && [ -z "${override}" ]; then
  echo "WARN: Expected 22 tickers but got ${COUNT}. Check RECOMMENDATION_TICKERS."
fi
