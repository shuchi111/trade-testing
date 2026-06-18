#!/usr/bin/env bash
# Preflight: same Z.ai Anthropic gateway as Postman/curl (x-api-key + /v1/messages).
set -euo pipefail

api_key="${Z_API_KEY:-${GLM_API_KEY:-${ANTHROPIC_AUTH_TOKEN:-}}}"
if [ -z "${api_key}" ]; then
  echo "ERROR: Set Z_API_KEY, GLM_API_KEY, or ANTHROPIC_AUTH_TOKEN for LLM preflight."
  exit 1
fi

provider="${LLM_PROVIDER:-anthropic}"
backend="${LLM_BACKEND_URL:-${ANTHROPIC_BASE_URL:-https://api.z.ai/api/anthropic}}"
backend="${backend%/}"

if [ "${provider}" != "anthropic" ] || [[ "${backend}" == *"/paas/"* ]]; then
  echo "WARN: LLM_PROVIDER=${provider} backend=${backend}"
  echo "WARN: Using Z.ai Anthropic gateway (https://api.z.ai/api/anthropic/v1/messages)."
  provider="anthropic"
  backend="https://api.z.ai/api/anthropic"
fi

if [[ "${backend}" == */v1 ]]; then
  url="${backend}/messages"
else
  url="${backend}/v1/messages"
fi

model="${DEEP_THINK_LLM:-glm-5.2}"
body="$(printf '{"model":"%s","max_tokens":16,"messages":[{"role":"user","content":"Reply with exactly: OK"}]}' "${model}")"
max_attempts="${LLM_PREFLIGHT_MAX_ATTEMPTS:-6}"

echo "LLM preflight: POST ${url} (provider=${provider}, model=${model})"

response_file="$(mktemp)"
trap 'rm -f "${response_file}"' EXIT

attempt=0
while [ "${attempt}" -lt "${max_attempts}" ]; do
  attempt=$((attempt + 1))
  http_code="$(curl -sS -o "${response_file}" -w "%{http_code}" \
    -X POST "${url}" \
    -H "Content-Type: application/json" \
    -H "x-api-key: ${api_key}" \
    -d "${body}")"

  if [ "${http_code}" = "200" ]; then
    echo "LLM preflight passed (HTTP ${http_code}, attempt ${attempt})."
    exit 0
  fi

  if [ "${attempt}" -lt "${max_attempts}" ] && { [ "${http_code}" = "529" ] || [ "${http_code}" = "503" ] || [ "${http_code}" = "429" ]; }; then
    wait_secs=$((20 * attempt))
    echo "WARN: LLM preflight HTTP ${http_code} (Z.ai overloaded) — retry ${attempt}/${max_attempts} in ${wait_secs}s..."
    cat "${response_file}" || true
    echo
    sleep "${wait_secs}"
    continue
  fi

  echo "ERROR: LLM preflight failed (HTTP ${http_code}, attempt ${attempt})."
  cat "${response_file}" || true
  echo
  exit 1
done
