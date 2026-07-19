#!/usr/bin/env bash
# Persist lessons from closed paper trades after execute.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "=== Reflect / learn from closed trades ==="
python agent/write_reflection_memory.py --lookback-days 45 --limit 50
echo "=== Reflection finished ==="
