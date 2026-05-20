#!/usr/bin/env bash
# session-end.sh
# Runs after each Claude Code session (Stop hook).
# 1. Runs configured verification checks for modified files
# 2. Appends a structured session boundary to the debugging log
# 3. Flags skill-improvement-loop if error threshold is exceeded

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_FILE="$PROJECT_DIR/.claude/debugging_log.md"
ERRORS_TMP="$PROJECT_DIR/.claude/session_errors.tmp"
CONFIG_FILE="$PROJECT_DIR/.claude/config"

ERROR_THRESHOLD_DEFAULT=2
CHECK_TIMEOUT_SECS_DEFAULT=45
CHECK_OUTPUT_LINES_DEFAULT=40
[[ -f "$CONFIG_FILE" ]] && source "$CONFIG_FILE"
if [[ -z "${ERROR_THRESHOLD+x}" && -n "${TS_ERROR_THRESHOLD+x}" ]]; then
  ERROR_THRESHOLD="$TS_ERROR_THRESHOLD"
fi
ERROR_THRESHOLD="${ERROR_THRESHOLD:-$ERROR_THRESHOLD_DEFAULT}"
CHECK_TIMEOUT_SECS="${CHECK_TIMEOUT_SECS:-$CHECK_TIMEOUT_SECS_DEFAULT}"
CHECK_OUTPUT_LINES="${CHECK_OUTPUT_LINES:-$CHECK_OUTPUT_LINES_DEFAULT}"
THRESHOLD_TRIGGER="$ERROR_THRESHOLD"
if [[ -z "${CHECK_DEFINITIONS+x}" ]]; then
  CHECK_DEFINITIONS=()
fi

mkdir -p "$PROJECT_DIR/.claude"

SESSION_DATE="$(date -u +%Y-%m-%d)"
SESSION_TIME="$(date -u +%H:%M:%SZ)"
CHECK_ERRORS=""
CHECK_BLOCKS=""
CHECKS_RUN=0
SKILL_FLAG=false

# ─── 1. Verification checks ──────────────────────────────────────────────────
MODIFIED_FILES="$(cd "$PROJECT_DIR" && git diff --name-only HEAD 2>/dev/null || true)"

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

declare -A CHECK_SEEN=()

if [[ -n "$MODIFIED_FILES" && ${#CHECK_DEFINITIONS[@]} -gt 0 ]]; then
  while IFS= read -r changed_file; do
    [[ -z "$changed_file" ]] && continue
    full_path="$PROJECT_DIR/$changed_file"
    for def in "${CHECK_DEFINITIONS[@]}"; do
      IFS='|' read -r -a CHECK_PARTS <<< "$def"
      if [[ ${#CHECK_PARTS[@]} -ne 5 ]]; then
        echo "⚠️  SESSION END: invalid check definition (expected 5 fields): $def"
        continue
      fi
      CHECK_NAME="${CHECK_PARTS[0]}"
      FILE_REGEX="${CHECK_PARTS[1]}"
      ROOT_MARKER="${CHECK_PARTS[2]}"
      COMMAND="${CHECK_PARTS[3]}"
      ERROR_REGEX="${CHECK_PARTS[4]}"
      [[ -z "$CHECK_NAME" || -z "$FILE_REGEX" || -z "$COMMAND" ]] && continue
      if [[ "$full_path" =~ $FILE_REGEX ]]; then
        ROOT="$(find_root "$ROOT_MARKER" "$(dirname "$full_path")")"
        if [[ -z "$ROOT" ]]; then
          if [[ -n "$ROOT_MARKER" && "$ROOT_MARKER" != "project" ]]; then
            echo "ℹ️  SESSION END: $CHECK_NAME skipped (no $ROOT_MARKER found for $changed_file)"
          fi
          continue
        fi
        CHECK_KEY_RAW="${CHECK_NAME}|${ROOT}"
        CHECK_KEY="$(printf '%s' "$CHECK_KEY_RAW" | sha256sum | cut -d' ' -f1)"
        [[ -n "${CHECK_SEEN[$CHECK_KEY]:-}" ]] && continue
        CHECK_SEEN["$CHECK_KEY"]=1

        CHECKS_RUN=$(( CHECKS_RUN + 1 ))
        RESULT=$(cd "$ROOT" && timeout "$CHECK_TIMEOUT_SECS" bash -c "$COMMAND" 2>&1 | head -"${CHECK_OUTPUT_LINES}" || true)
        ERROR_CODES=""
        if [[ -n "$ERROR_REGEX" && -n "$RESULT" ]]; then
          ERROR_CODES=$(echo "$RESULT" | grep -oE "$ERROR_REGEX" | sort -u | tr '\n' ' ')
        fi

        if [[ -n "$RESULT" ]]; then
          ERROR_COUNT=$(count_errors "$ERROR_CODES")
          CHECK_ERRORS="$CHECK_ERRORS
### Verification — $CHECK_NAME ($(basename "$ROOT"))
**Root:** $ROOT
**Command:** $COMMAND
**Result:** ❌ Errors found ($ERROR_COUNT)
\`\`\`
$RESULT
\`\`\`"
          if [[ -n "$ERROR_CODES" ]]; then
            for code in $ERROR_CODES; do
              CHECK_ERRORS="$CHECK_ERRORS
ERROR_CODE: $code"
            done
          else
            CHECK_ERRORS="$CHECK_ERRORS
CHECK_ERROR: $CHECK_NAME"
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
        else
          CHECK_BLOCKS="$CHECK_BLOCKS
### Verification — $CHECK_NAME ($(basename "$ROOT"))
**Root:** $ROOT
**Command:** $COMMAND
**Result:** ✅ No errors"
        fi
      fi
    done
  done <<< "$MODIFIED_FILES"
fi

# ─── 2. Count errors and decide flag ─────────────────────────────────────────
ERROR_COUNT=0
if [[ -n "$CHECK_ERRORS" ]]; then
  ERROR_COUNT="$(echo "$CHECK_ERRORS" | grep -c 'ERROR_CODE:' || true)"
  [[ "$ERROR_COUNT" -eq 0 ]] && ERROR_COUNT="$(echo "$CHECK_ERRORS" | grep -c 'CHECK_ERROR:' || true)"
fi

PREV_ERROR_COUNT=0
if [[ -f "$ERRORS_TMP" ]]; then
  PREV_ERROR_COUNT="$(grep -c 'ERROR_CODE:' "$ERRORS_TMP" 2>/dev/null || true)"
  [[ "$PREV_ERROR_COUNT" -eq 0 ]] && PREV_ERROR_COUNT="$(grep -c 'CHECK_ERROR:' "$ERRORS_TMP" 2>/dev/null || true)"
fi

TOTAL_ERRORS=$(( ERROR_COUNT + PREV_ERROR_COUNT ))
[[ $TOTAL_ERRORS -ge $THRESHOLD_TRIGGER ]] && SKILL_FLAG=true

# ─── 3. Append to debugging log ──────────────────────────────────────────────
{
  echo ""
  echo "---"
  echo "## Session End — $SESSION_DATE $SESSION_TIME"
  echo ""

  if [[ -n "$MODIFIED_FILES" ]]; then
    echo "**Modified files:**"
    echo "$MODIFIED_FILES" | sed 's/^/- /'
    echo ""
  fi

  if [[ $CHECKS_RUN -eq 0 ]]; then
    echo "**Verification Result:** ⚠️ No checks ran (no matching files or checks configured)"
  else
    if [[ -n "$CHECK_ERRORS" ]]; then
      echo "**Verification Result:** ❌ Errors found ($ERROR_COUNT error(s))"
      echo "$CHECK_ERRORS"
    else
      echo "**Verification Result:** ✅ No verification errors"
    fi
    if [[ -n "$CHECK_BLOCKS" ]]; then
      echo "$CHECK_BLOCKS"
    fi
  fi

  if [[ -f "$ERRORS_TMP" && -s "$ERRORS_TMP" ]]; then
    echo ""
    echo "**Accumulated session errors:**"
    cat "$ERRORS_TMP"
  fi

  if $SKILL_FLAG; then
    echo ""
    echo "> ⚠️ **AUTO-FLAG:** $TOTAL_ERRORS error(s) this session exceeded threshold ($THRESHOLD_TRIGGER)."
    echo "> Run \`skill-improvement-loop\` before next task — score active skills and update any with trigger gaps."
  fi

  echo ""
} >> "$LOG_FILE"

# ─── 4. Send ntfy.sh push notification ───────────────────────────────────────
NOTIFY="$PROJECT_DIR/scripts/notify.sh"
if [[ -x "$NOTIFY" ]]; then
  if $SKILL_FLAG; then
    bash "$NOTIFY" \
      "Done — $TOTAL_ERRORS verification error(s). Run skill-improvement-loop before next task." \
      "Claude Code" "high" "warning,claude"
  elif [[ -n "$CHECK_ERRORS" ]]; then
    bash "$NOTIFY" \
      "Done — verification errors found. Check .claude/debugging_log.md" \
      "Claude Code" "default" "warning,claude"
  else
    bash "$NOTIFY" \
      "Done — no errors." \
      "Claude Code" "default" "white_check_mark,claude"
  fi
fi

# ─── 5. Update skill memory from structured debug entries ────────────────────
UPDATE_MEM="$PROJECT_DIR/scripts/update-skill-memory.sh"
if [[ -x "$UPDATE_MEM" ]]; then
  bash "$UPDATE_MEM" 2>/dev/null || true
fi

# ─── 6. Run pattern analysis ──────────────────────────────────────────────────
ANALYZE="$PROJECT_DIR/scripts/analyze-patterns.sh"
if [[ -x "$ANALYZE" ]]; then
  bash "$ANALYZE" 2>/dev/null || true
fi

if $SKILL_FLAG; then
  echo "⚠️  SESSION END: $TOTAL_ERRORS verification error(s) — skill-improvement-loop flagged"
elif [[ -n "$CHECK_ERRORS" ]]; then
  echo "⚠️  SESSION END: Verification errors found — check .claude/debugging_log.md"
elif [[ $CHECKS_RUN -eq 0 ]]; then
  echo "⚠️  SESSION END: No verification checks ran"
else
  echo "✅ SESSION END: No verification errors"
fi
