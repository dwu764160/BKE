#!/usr/bin/env bash
# notify-approval.sh
# Notification hook — fires when Claude Code needs user interaction/approval
# (tool permission prompts, waiting for input, etc.)

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
NOTIFY="$PROJECT_DIR/scripts/notify.sh"
LOG_FILE="$PROJECT_DIR/.claude/notification_log.txt"

mkdir -p "$(dirname "$LOG_FILE")"

MSG=""
[[ -n "${CLAUDE_NOTIFICATION:-}" ]] && MSG="$CLAUDE_NOTIFICATION"
[[ -z "$MSG" && -n "${NOTIFICATION:-}" ]] && MSG="$NOTIFICATION"
[[ -z "$MSG" && -n "${CLAUDE_MESSAGE:-}" ]] && MSG="$CLAUDE_MESSAGE"
[[ -z "$MSG" ]] && MSG="Claude Code is waiting for your approval or input"

{
  echo "[$(date -u +%H:%M:%SZ)] Notification event"
  echo "  Message: $MSG"
  echo "  Env vars: CLAUDE_NOTIFICATION='${CLAUDE_NOTIFICATION:-}' NOTIFICATION='${NOTIFICATION:-}' CLAUDE_MESSAGE='${CLAUDE_MESSAGE:-}'"
  echo ""
} >> "$LOG_FILE"

bash "$NOTIFY" \
  "⏸️  $MSG" \
  "Claude Code — Waiting for You" "high" "bell,hourglass,claude" 2>/dev/null || {
  echo "[$(date -u +%H:%M:%SZ)] notify.sh failed" >> "$LOG_FILE"
}

echo "🔔 Notification sent to ntfy.sh — check your phone"
