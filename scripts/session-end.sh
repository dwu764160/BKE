#!/usr/bin/env bash
# session-end.sh — Stop hook (minimal, post-strip 2026-05-30).
#
# The entire improvement loop is now: write a `### Debug Entry` in
# .claude/debugging_log.md when you make a real mistake. This hook only does
# four things:
#   1. verification check  (pytest — this Python repo's equivalent of a tsc build check)
#   2. append a session boundary to the debugging log
#   3. run the memory-update script (new Debug Entries -> persistent memory)
#   4. send a push notification
#
# Removed: per-file verification matrix, error-threshold auto-flagging,
# pattern scanning, skill smoke tests, pending-improvements generation.

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_FILE="$PROJECT_DIR/.claude/debugging_log.md"
cd "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR/.claude"

# ── 1. Verification check (pytest = repo build/type check) ────────────────────
CHECK_OUT="$(timeout 180 python3 -m pytest -q tests/ 2>&1 | tail -20 || true)"
if echo "$CHECK_OUT" | grep -qiE "failed|error"; then
  CHECK_STATUS="❌ check errors"
else
  CHECK_STATUS="✅ check clean"
fi

# ── 2. Append a session boundary to the debugging log ─────────────────────────
{
  echo ""
  echo "---"
  echo "## Session boundary — $(date -u +'%Y-%m-%d %H:%M:%SZ') — $CHECK_STATUS"
  echo ""
} >> "$LOG_FILE"

# ── 3. Memory update (Debug Entries -> persistent memory) ─────────────────────
UPDATE_MEM="$PROJECT_DIR/scripts/update-skill-memory.sh"
[[ -x "$UPDATE_MEM" ]] && bash "$UPDATE_MEM" 2>/dev/null || true

# ── 4. Push notification ──────────────────────────────────────────────────────
NOTIFY="$PROJECT_DIR/scripts/notify.sh"
[[ -x "$NOTIFY" ]] && bash "$NOTIFY" \
  "Session ended — $CHECK_STATUS" "Claude Code" "default" "claude" 2>/dev/null || true

echo "SESSION END: $CHECK_STATUS"
