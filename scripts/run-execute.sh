#!/usr/bin/env bash
# Paper-trade executor only (after all batch shards finish).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=market-date.sh
source "${SCRIPT_DIR}/market-date.sh"

trade_date_input="${TRADE_DATE_INPUT:-}"
if [ -z "${trade_date_input}" ]; then
  DATE="$(market_trade_date)"
else
  DATE="${trade_date_input}"
fi

echo "=== Execute paper trades ==="
echo "Current time (IST): $(ist_now)"
echo "Trade date: ${DATE}"

python agent/execute_ai_trades.py --all --trade-date "${DATE}"

echo "=== Execute finished ==="
