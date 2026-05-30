#!/usr/bin/env bash
# update-skill-memory.sh
# THE memory-update script: parses new `### Debug Entry` blocks from
# .claude/debugging_log.md and folds their signal into persistent memory:
#   - memory/skill_effectiveness.md   (Miss Count column, from **Skill gap:** lines)
#   - memory/debugging_patterns.md    (root-cause entries, from **Root cause:** lines)
#
# Incremental: a cursor (.claude/skill_memory_cursor.tmp) tracks the last log line
# processed, so each session only reads NEW entries — it never rescans history.
# Called by session-end.sh. This is the only surviving piece of the old loop.

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_FILE="$PROJECT_DIR/.claude/debugging_log.md"
SKILL_MEM="$PROJECT_DIR/memory/skill_effectiveness.md"
PATTERN_MEM="$PROJECT_DIR/memory/debugging_patterns.md"
CURSOR_FILE="$PROJECT_DIR/.claude/skill_memory_cursor.tmp"

[[ ! -f "$LOG_FILE" ]] && exit 0

# ── 1. Find unprocessed Debug Entry blocks (incremental via cursor) ───────────
LAST_LINE=0
[[ -f "$CURSOR_FILE" ]] && LAST_LINE=$(cat "$CURSOR_FILE" 2>/dev/null || echo 0)
TOTAL_LINES=$(wc -l < "$LOG_FILE")

NEW_ENTRIES=$(awk -v start="$LAST_LINE" '
  NR <= start { next }
  /^### Debug Entry/ { in_entry=1; block="" }
  in_entry { block = block "\n" $0 }
  in_entry && /^### / && !/^### Debug Entry/ { in_entry=0; print block }
  END { if (in_entry) print block }
' "$LOG_FILE" 2>/dev/null || echo "")

if [[ -z "$NEW_ENTRIES" ]]; then
  echo "$TOTAL_LINES" > "$CURSOR_FILE"
  exit 0
fi

# ── 2. Parse skill gaps and root causes from the new entries ──────────────────
declare -A SKILL_GAPS=()
declare -A ROOT_CAUSES=()
declare -A ROOT_CAUSE_TEXT=()

while IFS= read -r block; do
  [[ -z "$block" ]] && continue

  gap_line=$(echo "$block" | grep -m1 '\*\*Skill gap:\*\*' | sed 's/\*\*Skill gap:\*\*[[:space:]]*//')
  if [[ -n "$gap_line" ]]; then
    skill_name=$(echo "$gap_line" | grep -oE '[a-z]+-[a-z][a-z-]+[a-z]' | head -1 || true)
    [[ -n "$skill_name" ]] && SKILL_GAPS["$skill_name"]=$(( ${SKILL_GAPS[$skill_name]:-0} + 1 ))
  fi

  cause_line=$(echo "$block" | grep -m1 '\*\*Root cause:\*\*' | sed 's/\*\*Root cause:\*\*[[:space:]]*//')
  if [[ -n "$cause_line" ]]; then
    cause_key=$(echo "$cause_line" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g' | sed 's/-\+/-/g' | cut -c1-60)
    ROOT_CAUSES["$cause_key"]=$(( ${ROOT_CAUSES[$cause_key]:-0} + 1 ))
    ROOT_CAUSE_TEXT["$cause_key"]="$cause_line"
  fi
done <<< "$NEW_ENTRIES"

# ── 3. Fold skill gaps into skill_effectiveness.md miss counts ────────────────
if [[ ${#SKILL_GAPS[@]} -gt 0 && -f "$SKILL_MEM" ]]; then
  for skill in "${!SKILL_GAPS[@]}"; do
    new_misses="${SKILL_GAPS[$skill]}"
    if grep -q "| $skill " "$SKILL_MEM" 2>/dev/null; then
      current=$(awk -F'|' -v s="$skill" '$2 ~ s { gsub(/ /,"",$5); print $5+0; exit }' "$SKILL_MEM" 2>/dev/null || echo 0)
      updated=$(( current + new_misses ))
      sed -i "s/| $skill \\(.*\\)| $current \\(.*\\)|/| $skill \\1| $updated \\2|/" "$SKILL_MEM" 2>/dev/null || true
    fi
    echo "  skill_effectiveness.md: $skill miss +$new_misses"
  done
fi

# ── 4. Record recurring root causes in debugging_patterns.md ──────────────────
if [[ ${#ROOT_CAUSES[@]} -gt 0 && -f "$PATTERN_MEM" ]]; then
  today="$(date -u +%Y-%m-%d)"
  for cause_key in "${!ROOT_CAUSES[@]}"; do
    count="${ROOT_CAUSES[$cause_key]}"
    [[ $count -lt 2 ]] && continue
    cause_text="${ROOT_CAUSE_TEXT[$cause_key]}"
    if grep -q "$cause_key" "$PATTERN_MEM" 2>/dev/null; then
      sed -i "s/\*\*Last seen:\*\* .*/\*\*Last seen:\*\* $today/" "$PATTERN_MEM" 2>/dev/null || true
      echo "  debugging_patterns.md: updated '$cause_text' (x$count)"
    else
      {
        echo ""
        echo "### Pattern: $(echo "$cause_text" | cut -c1-60)"
        echo "- **First seen:** $today"
        echo "- **Last seen:** $today"
        echo "- **Occurrences:** $count"
        echo "- **Description:** $cause_text"
      } >> "$PATTERN_MEM"
      echo "  debugging_patterns.md: new pattern '$cause_text'"
    fi
  done
fi

# ── 5. Advance cursor ─────────────────────────────────────────────────────────
echo "$TOTAL_LINES" > "$CURSOR_FILE"
echo "✅ memory updated from $(echo "$NEW_ENTRIES" | grep -c '### Debug Entry' || echo 0) new Debug Entry(s)"
