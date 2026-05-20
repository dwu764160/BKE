#!/usr/bin/env bash
# post-edit-check.sh
# PostToolUse hook — fires after every Edit or Write tool call.
# Runs configured verification checks for the edited file.
# Rate-limited: skips if the same root+check was run less than N seconds ago.

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ERRORS_TMP="$PROJECT_DIR/.claude/session_errors.tmp"
LAST_CHECK_FILE="$PROJECT_DIR/.claude/last_check.tmp"
CONFIG_FILE="$PROJECT_DIR/.claude/config"

POST_EDIT_RATE_LIMIT_SECS=30
CHECK_TIMEOUT_SECS=45
CHECK_OUTPUT_LINES=40
[[ -f "$CONFIG_FILE" ]] && source "$CONFIG_FILE"
RATE_LIMIT_SECS="${POST_EDIT_RATE_LIMIT_SECS}"
if [[ -z "${CHECK_DEFINITIONS+x}" ]]; then
  CHECK_DEFINITIONS=()
fi

mkdir -p "$PROJECT_DIR/.claude"

# ─── 1. Extract edited file path from CLAUDE_TOOL_INPUT ──────────────────────
FILE=""
if [[ -n "${CLAUDE_TOOL_INPUT:-}" ]]; then
  FILE=$(python3 -c "
import json, sys
try:
    d = json.loads(sys.argv[1])
    print(d.get('file_path', ''))
except Exception:
    print('')
" "$CLAUDE_TOOL_INPUT" 2>/dev/null || echo "")
fi

[[ -z "$FILE" ]] && exit 0

find_root() {
  local marker="$1"
  local search_dir="$2"
  if [[ -z "$marker" || "$marker" == "project" ]]; then
    echo "$PROJECT_DIR"
    return
  fi
  while [[ "$search_dir" != "/" && "$search_dir" != "." ]]; do
    if [[ -f "$search_dir/$marker" ]]; then
      echo "$search_dir"
      return
    fi
    [[ "$search_dir" == "$PROJECT_DIR" ]] && break
    search_dir="$(dirname "$search_dir")"
  done
  echo ""
}

count_errors() {
  local codes="$1"
  if [[ -z "$codes" ]]; then
    echo 1
    return
  fi
  local count
  count=$(echo "$codes" | tr ' ' '\n' | grep -E '.+' | sort -u | wc -l | tr -d ' ')
  [[ "$count" -eq 0 ]] && count=1
  echo "$count"
}

[[ ${#CHECK_DEFINITIONS[@]} -eq 0 ]] && exit 0

MATCHED=false
for def in "${CHECK_DEFINITIONS[@]}"; do
  IFS='|' read -r -a CHECK_PARTS <<< "$def"
  if [[ ${#CHECK_PARTS[@]} -ne 5 ]]; then
    echo "⚠️  POST-EDIT: invalid check definition (expected 5 fields): $def"
    continue
  fi
  CHECK_NAME="${CHECK_PARTS[0]}"
  FILE_REGEX="${CHECK_PARTS[1]}"
  ROOT_MARKER="${CHECK_PARTS[2]}"
  COMMAND="${CHECK_PARTS[3]}"
  ERROR_REGEX="${CHECK_PARTS[4]}"
  [[ -z "$CHECK_NAME" || -z "$FILE_REGEX" || -z "$COMMAND" ]] && continue
  if [[ "$FILE" =~ $FILE_REGEX ]]; then
    MATCHED=true
    ROOT="$(find_root "$ROOT_MARKER" "$(dirname "$FILE")")"
    if [[ -z "$ROOT" ]]; then
      if [[ -n "$ROOT_MARKER" && "$ROOT_MARKER" != "project" ]]; then
        echo "ℹ️  POST-EDIT: $CHECK_NAME skipped (no $ROOT_MARKER found for $(basename "$FILE"))"
      fi
      continue
    fi

    CHECK_KEY_RAW="${CHECK_NAME}|${ROOT}"
    CHECK_KEY="$(printf '%s' "$CHECK_KEY_RAW" | sha256sum | cut -d' ' -f1)"

    NOW=$(date +%s)
    LAST_CHECK=0
    if [[ -f "$LAST_CHECK_FILE" ]]; then
      LAST_CHECK=$(grep -F "$CHECK_KEY=" "$LAST_CHECK_FILE" 2>/dev/null | cut -d= -f2 || echo 0)
    fi
    ELAPSED=$(( NOW - LAST_CHECK ))
    if [[ $ELAPSED -lt $RATE_LIMIT_SECS ]]; then
      continue
    fi

    if grep -Fq "$CHECK_KEY=" "$LAST_CHECK_FILE" 2>/dev/null; then
      sed -i "s|^$CHECK_KEY=.*|$CHECK_KEY=$NOW|" "$LAST_CHECK_FILE"
    else
      echo "$CHECK_KEY=$NOW" >> "$LAST_CHECK_FILE"
    fi

    RESULT=$(cd "$ROOT" && timeout "$CHECK_TIMEOUT_SECS" bash -c "$COMMAND" 2>&1 | head -"${CHECK_OUTPUT_LINES}" || true)
    [[ -z "$RESULT" ]] && continue

    ERROR_CODES=""
    if [[ -n "$ERROR_REGEX" ]]; then
      ERROR_CODES=$(echo "$RESULT" | grep -oE "$ERROR_REGEX" | sort -u | tr '\n' ' ')
    fi

    {
      echo "### Check error — $CHECK_NAME — $(date -u +%H:%M:%SZ)"
      echo "**Root:** $ROOT"
      echo "**Command:** $COMMAND"
      echo '```'
      echo "$RESULT"
      echo '```'
      if [[ -n "$ERROR_CODES" ]]; then
        for code in $ERROR_CODES; do
          echo "ERROR_CODE: $code"
        done
      else
        echo "CHECK_ERROR: $CHECK_NAME"
      fi
      echo ""
    } >> "$ERRORS_TMP"

    ERROR_COUNT=$(count_errors "$ERROR_CODES")
    bash "$PROJECT_DIR/scripts/notify.sh" \
      "$CHECK_NAME: $ERROR_COUNT error(s) after editing $(basename "$FILE")" \
      "Claude Code — Verification Error" "high" "x,claude" 2>/dev/null || true

    echo "⚠️  POST-EDIT: $ERROR_COUNT error(s) in $CHECK_NAME — see .claude/session_errors.tmp"
  fi
done

if ! $MATCHED; then
  exit 0
fi
