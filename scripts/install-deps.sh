#!/usr/bin/env bash
set -euo pipefail

REQ="agent/requirements.txt"

if [ ! -f "${REQ}" ]; then
  echo "ERROR: ${REQ} not found"
  exit 1
fi

python -m pip install --upgrade pip
pip install -r "${REQ}"
echo "Dependencies installed from ${REQ}"
