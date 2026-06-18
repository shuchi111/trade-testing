#!/usr/bin/env bash
# Preflight: same Z.ai Anthropic gateway as Postman/curl (x-api-key + /v1/messages).
set -euo pipefail

api_key="${Z_API_KEY:-${GLM_API_KEY:-${ANTHROPIC_AUTH_TOKEN:-}}}"
if [ -z "${api_key}" ]; then
  echo "ERROR: Set Z_API_KEY, GLM_API_KEY, or ANTHROPIC_AUTH_TOKEN for LLM preflight."
  exit 1
fi

provider="${LLM_PROVIDER:-***REMOVED***}"
backend="${LLM_BACKEND_URL:-${ANTHROPIC_BASE_URL:-***REMOVED***}}"
backend="${backend%/}"

if [ "${provider}" != "***REMOVED***" ] || [[ "${backend}" == *"/paas/"* ]]; then
  echo "WARN: LLM_PROVIDER=${provider} backend=${backend}"
  echo "WARN: Using Z.ai Anthropic gateway (***REMOVED***/v1/messages)."
  provider="***REMOVED***"
  backend="***REMOVED***"
fi

if [[ "${backend}" == */v1 ]]; then
  url="${backend}/messages"
else
  url="${backend}/v1/messages"
fi

model="${DEEP_THINK_LLM:-glm-5.2}"
body="$(printf '{"model":"%s","max_tokens":16,"messages":[{"role":"user","content":"Reply with exactly: OK"}]}' "${model}")"

echo "LLM preflight: POST ${url} (provider=${provider}, model=${model})"

response_file="$(mktemp)"
trap 'rm -f "${response_file}"' EXIT

http_code="$(curl -sS -o "${response_file}" -w "%{http_code}" \
  -X POST "${url}" \
  -H "Content-Type: application/json" \
  -H "x-api-key: ${api_key}" \
  -d "${body}")"

if [ "${http_code}" = "200" ]; then
  echo "LLM preflight passed (HTTP ${http_code})."
  exit 0
fi

echo "ERROR: LLM preflight failed (HTTP ${http_code})."
cat "${response_file}" || true
echo
exit 1
