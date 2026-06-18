#!/usr/bin/env bash
# NSE-style trade date: calendar day in IST; Sat/Sun roll back to Friday.
# Does not skip exchange holidays (add a holiday calendar later if needed).
set -euo pipefail

export TZ='Asia/Kolkata'

ist_today() {
  date +'%Y-%m-%d'
}

ist_now() {
  date +'%Y-%m-%d %H:%M:%S %Z'
}

market_trade_date() {
  local raw dow
  raw="$(ist_today)"
  dow="$(date +%u)" # 1=Mon .. 7=Sun
  case "$dow" in
    6) date -d "$raw -1 day" +'%Y-%m-%d' ;;
    7) date -d "$raw -2 day" +'%Y-%m-%d' ;;
    *) echo "$raw" ;;
  esac
}

is_ist_weekday() {
  local dow
  dow="$(date +%u)"
  [ "$dow" -le 5 ]
}
