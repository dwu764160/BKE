#!/usr/bin/env bash
# update-skill-memory.sh
# Parses structured Debug Entry blocks from .claude/debugging_log.md and
# auto-updates both memory files:
#   - memory/skill_effectiveness.md   (Miss Count column)
#   - memory/debugging_patterns.md    (Pattern Registry + entries)
#
# Called by session-end.sh after each session.
# Only processes entries that haven't been processed yet (tracks position via cursor).

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_FILE="$PROJECT_DIR/.claude/debugging_log.md"
SKILL_MEM="$PROJECT_DIR/memory/skill_effectiveness.md"
PATTERN_MEM="$PROJECT_DIR/memory/debugging_patterns.md"
CURSOR_FILE="$PROJECT_DIR/.claude/skill_memory_cursor.tmp"
CONFIG_FILE="$PROJECT_DIR/.claude/config"

SKILL_GAP_ESCALATE_MIN=2
[[ -f "$CONFIG_FILE" ]] && source "$CONFIG_FILE"

[[ ! -f "$LOG_FILE" ]] && exit 0

# ─── 1. Find unprocessed debug entries ────────────────────────────────────────
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

[[ -z "$NEW_ENTRIES" ]] && {
  echo "$TOTAL_LINES" > "$CURSOR_FILE"
  exit 0
}

# ─── 2. Parse skill gaps and root causes from entries ─────────────────────────
declare -A SKILL_GAPS=()
declare -A ROOT_CAUSES=()
declare -A ROOT_CAUSE_TEXT=()

while IFS= read -r block; do
  [[ -z "$block" ]] && continue

  gap_line=$(echo "$block" | grep -m1 '\*\*Skill gap:\*\*' | sed 's/\*\*Skill gap:\*\*[[:space:]]*//')
  if [[ -n "$gap_line" ]]; then
    skill_name=$(echo "$gap_line" | grep -oE '[a-z]+-[a-z][a-z-]+[a-z]' | head -1 || true)
    if [[ -n "$skill_name" ]]; then
      SKILL_GAPS["$skill_name"]=$(( ${SKILL_GAPS[$skill_name]:-0} + 1 ))
    fi
  fi

  cause_line=$(echo "$block" | grep -m1 '\*\*Root cause:\*\*' | sed 's/\*\*Root cause:\*\*[[:space:]]*//')
  if [[ -n "$cause_line" ]]; then
    cause_key=$(echo "$cause_line" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g' | sed 's/-\+/-/g' | cut -c1-60)
    ROOT_CAUSES["$cause_key"]=$(( ${ROOT_CAUSES[$cause_key]:-0} + 1 ))
    ROOT_CAUSE_TEXT["$cause_key"]="$cause_line"
  fi
done <<< "$NEW_ENTRIES"

# ─── 3. Update skill_effectiveness.md miss counts ─────────────────────────────
if [[ ${#SKILL_GAPS[@]} -gt 0 && -f "$SKILL_MEM" ]]; then
  for skill in "${!SKILL_GAPS[@]}"; do
    new_misses="${SKILL_GAPS[$skill]}"
    if grep -q "| $skill " "$SKILL_MEM" 2>/dev/null; then
      current=$(awk -F'|' -v s="$skill" '$2 ~ s { gsub(/ /,"",$5); print $5+0; exit }' "$SKILL_MEM" 2>/dev/null || echo 0)
      updated=$(( current + new_misses ))
      sed -i "s/| $skill \\(.*\\)| $current \\(.*\\)|/| $skill \\1| $updated \\2|/" "$SKILL_MEM" 2>/dev/null || true
    fi
    echo "  Updated skill_effectiveness.md: $skill miss count +$new_misses"
  done
  {
    echo ""
    printf "<!-- %s -->\n" "$(date -u +%Y-%m-%d)"
  } >> "$SKILL_MEM"
fi

# ─── 4. Update debugging_patterns.md for recurring root causes ────────────────
if [[ ${#ROOT_CAUSES[@]} -gt 0 && -f "$PATTERN_MEM" ]]; then
  for cause_key in "${!ROOT_CAUSES[@]}"; do
    count="${ROOT_CAUSES[$cause_key]}"
    [[ $count -lt 2 ]] && continue

    cause_text="${ROOT_CAUSE_TEXT[$cause_key]}"
    today="$(date -u +%Y-%m-%d)"

    if grep -q "$cause_key" "$PATTERN_MEM" 2>/dev/null; then
      sed -i "s/\*\*Last seen:\*\* .*/\*\*Last seen:\*\* $today/" "$PATTERN_MEM" 2>/dev/null || true
      echo "  Updated pattern: $cause_text (count: $count)"
    else
      {
        echo ""
        echo "### Pattern: $(echo "$cause_text" | cut -c1-60)"
        echo "- **First seen:** $today"
        echo "- **Last seen:** $today"
        echo "- **Occurrences:** $count"
        echo "- **Description:** $cause_text"
        echo "- **Root cause:** (see debugging_log.md entries)"
        echo "- **Associated skill:** (fill in from Skill gap entries)"
        echo "- **Skill gap:** (fill in from debugging_log.md)"
        echo "- **Resolution applied:** (fill in after fix)"
        echo "- **Skill updated:** no"
      } >> "$PATTERN_MEM"
      echo "  New pattern added: $cause_text"
    fi
  done
fi

# ─── 5. Check for escalation (skill gap >= threshold) ─────────────────────────
ESCALATED=""
for skill in "${!SKILL_GAPS[@]}"; do
  count="${SKILL_GAPS[$skill]}"
  if [[ $count -ge $SKILL_GAP_ESCALATE_MIN ]]; then
    ESCALATED="$ESCALATED $skill($count)"
  fi
done

if [[ -n "$ESCALATED" ]]; then
  PRIORITY="${SKILL_GAP_NOTIFY_PRIORITY:-urgent}"
  bash "$PROJECT_DIR/scripts/notify.sh" \
    "URGENT: Skill gap threshold hit —$ESCALATED. Run skill-improvement-loop NOW." \
    "Claude Code — Skill Gap" "$PRIORITY" "rotating_light,claude" 2>/dev/null || true
  echo "⚠️  SKILL MEMORY: Gap escalated for$ESCALATED — urgent improvement needed"
fi

# ─── 6. Update cursor ─────────────────────────────────────────────────────────
echo "$TOTAL_LINES" > "$CURSOR_FILE"
echo "✅ SKILL MEMORY: Updated from $(echo "$NEW_ENTRIES" | grep -c '### Debug Entry' || echo 0) new entry(s)"
