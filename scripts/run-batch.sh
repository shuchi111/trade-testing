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

normalize_empty() {
  case "${1:-}" in
    ""|"empty"|"EMPTY"|"null"|"NULL") printf "" ;;
    *) printf "%s" "$1" ;;
  esac
}

override="$(normalize_empty "${override}")"
trade_date_input="$(normalize_empty "${trade_date_input}")"

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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=market-date.sh
source "${SCRIPT_DIR}/market-date.sh"

if [ -z "${trade_date_input}" ]; then
  DATE="$(market_trade_date)"
else
  DATE="${trade_date_input}"
fi

TICKERS="$(echo "${TICKERS}" | tr -d '[:space:]')"

if [ -z "${TICKERS}" ]; then
  echo "ERROR: No tickers. Set RECOMMENDATION_TICKERS in CircleCI."
  exit 1
fi

shard_index="${TICKER_SHARD_INDEX:-0}"
shard_total="${TICKER_SHARD_TOTAL:-1}"

# Manual override: only shard 0 runs (other shards exit immediately)
if [ -n "${override}" ] && [ "${shard_total}" -gt 1 ] && [ "${shard_index}" -ne 0 ]; then
  echo "SKIP: tickers_override set — only shard 0 runs manual tickers."
  exit 0
fi

IFS=',' read -ra ARR <<< "${TICKERS}"

# Split full batch across shards (CircleCI free tier ≈ 1h job limit)
if [ -z "${override}" ] && [ "${shard_total}" -gt 1 ]; then
  SHARDED=()
  for i in "${!ARR[@]}"; do
    if [ $((i % shard_total)) -eq "${shard_index}" ]; then
      SHARDED+=("${ARR[$i]}")
    fi
  done
  ARR=("${SHARDED[@]}")
  echo "Shard ${shard_index}/${shard_total}: ${#ARR[@]} ticker(s)"
fi

if [ ${#ARR[@]} -eq 0 ]; then
  echo "No tickers for this shard — nothing to do."
  exit 0
fi

TICKERS="$(IFS=','; echo "${ARR[*]}")"

export LLM_PROVIDER="${LLM_PROVIDER:-***REMOVED***}"
export LLM_BACKEND_URL="${LLM_BACKEND_URL:-${ANTHROPIC_BASE_URL:-***REMOVED***}}"
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-${LLM_BACKEND_URL}}"

ticker_delay="${BATCH_TICKER_DELAY_SEC:-0}"

echo "=== Batch run ==="
echo "Current time (IST): $(ist_now)"
echo "LLM provider: ${LLM_PROVIDER}"
echo "LLM backend:  ${LLM_BACKEND_URL}"
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
  if [ "${ticker_delay}" -gt 0 ] && [ "${COUNT}" -lt ${#ARR[@]} ]; then
    echo "Waiting ${ticker_delay}s before next ticker..."
    sleep "${ticker_delay}"
  fi
done

echo
echo "=== Batch shard finished ==="
echo "Tickers attempted: ${COUNT}"
echo "Tickers succeeded: ${OK}"

if [ ${#FAILED[@]} -gt 0 ]; then
  echo "Failed tickers (${#FAILED[@]}): ${FAILED[*]}"
fi

if [ "${OK}" -eq 0 ] && [ "${COUNT}" -gt 0 ]; then
  echo "ERROR: All tickers failed"
  exit 1
fi

if [ "${COUNT}" -ne 22 ] && [ -z "${override}" ] && [ "${shard_total}" -le 1 ]; then
  echo "WARN: Expected 22 tickers but got ${COUNT}. Check RECOMMENDATION_TICKERS."
fi
