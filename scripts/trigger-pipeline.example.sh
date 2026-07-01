#!/usr/bin/env bash
# Trigger full batch (all 30 tickers) on CircleCI.
#
#   export CIRCLECI_TOKEN=your-personal-api-token
#   export CIRCLE_PROJECT_SLUG=gh/YOUR_USER/swing-trader-circleci-cron
#   bash scripts/trigger-pipeline.example.sh

set -euo pipefail

MODE="${1:-batch}"

: "${CIRCLECI_TOKEN:?Set CIRCLECI_TOKEN}"
: "${CIRCLE_PROJECT_SLUG:?Set CIRCLE_PROJECT_SLUG e.g. gh/user/swing-trader-circleci-cron}"

case "${MODE}" in
  batch)
    PARAMS='{"mode":"batch","tickers_override":"","trade_date":""}'
    ;;
  price_refresh)
    PARAMS='{"mode":"price_refresh","tickers_override":"","trade_date":""}'
    ;;
  both)
    PARAMS='{"mode":"both","tickers_override":"","trade_date":""}'
    ;;
  *)
    echo "Usage: $0 [batch|price_refresh|both]"
    exit 1
    ;;
esac

curl -sS -X POST \
  "https://circleci.com/api/v2/project/${CIRCLE_PROJECT_SLUG}/pipeline" \
  -H "Circle-Token: ${CIRCLECI_TOKEN}" \
  -H "Content-Type: application/json" \
  -d "{\"branch\":\"main\",\"parameters\":${PARAMS}}"

echo
