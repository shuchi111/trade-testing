#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [ ! -f "agent/refresh_stale_recommendations.py" ]; then
  echo "ERROR: agent/refresh_stale_recommendations.py not found"
  exit 1
fi

echo "=== Price refresh run ==="
echo "PRICE_REFRESH_RATIO=${PRICE_REFRESH_RATIO:-0.03}"
echo "MAX_CACHE_AGE_DAYS=${MAX_CACHE_AGE_DAYS:-10}"
echo

python agent/refresh_stale_recommendations.py

echo "=== Price refresh finished ==="
