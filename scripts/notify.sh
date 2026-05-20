#!/usr/bin/env bash
# notify.sh — sends a push notification to ntfy.sh
# Usage: notify.sh "message" [title] [priority] [tags]
#   priority: min | low | default | high | urgent
#   tags:     comma-separated ntfy tag names (e.g. "white_check_mark,claude")
#
# Setup: set NTFY_CHANNEL_URL to your ntfy.sh topic, e.g.:
#   export NTFY_CHANNEL_URL="https://ntfy.sh/my-project-alerts"
# Or edit the default below.

# ── Configure your channel ────────────────────────────────────────────────────
NTFY_URL="${NTFY_CHANNEL_URL:-https://ntfy.sh/YOUR_CHANNEL_NAME}"
# Replace YOUR_CHANNEL_NAME with your ntfy.sh topic, or set NTFY_CHANNEL_URL env var.
# Free topics at https://ntfy.sh — pick something unique to avoid others subscribing.

PROJECT_TITLE="${NTFY_PROJECT_TITLE:-Claude Code}"

MSG="${1:-Claude notification}"
TITLE="${2:-$PROJECT_TITLE}"
PRIORITY="${3:-default}"
TAGS="${4:-bell}"

curl -s \
  -H "Title: $TITLE" \
  -H "Priority: $PRIORITY" \
  -H "Tags: $TAGS" \
  -d "$MSG" \
  "$NTFY_URL" > /dev/null 2>&1 || true
