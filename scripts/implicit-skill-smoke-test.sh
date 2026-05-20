#!/usr/bin/env bash
# implicit-skill-smoke-test.sh
# Validates that skill descriptions + When-to-Use sections would auto-load
# the correct skills for realistic task prompts — without explicit skill names.
#
# Usage:
#   ./scripts/implicit-skill-smoke-test.sh            # run full golden test suite
#   ./scripts/implicit-skill-smoke-test.sh "my prompt"  # test a single custom prompt
#   ./scripts/implicit-skill-smoke-test.sh --report    # only show failures + effectiveness

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SKILLS_DIR="$ROOT_DIR/.github/skills"
REPORT_ONLY=false
[[ "${1:-}" == "--report" ]] && REPORT_ONLY=true && shift

if [[ ! -d "$SKILLS_DIR" ]]; then
  echo "ERROR: skills directory not found at $SKILLS_DIR" >&2; exit 1
fi

# ─── Scoring ────────────────────────────────────────────────────────────────
# For each skill, build a trigger vocabulary from:
#   - frontmatter description (weight 3)
#   - "## When to Use" section body (weight 2)
# Score a prompt by summing weights of matching unique tokens.
# Threshold >= 3 = skill "loads"

LOAD_THRESHOLD=3

tokenize() {
  local input
  if [[ $# -gt 0 ]]; then
    input="$1"
  else
    input="$(cat)"
  fi
  echo "$input" | tr '[:upper:]' '[:lower:]' \
    | sed 's/[^a-z0-9 ]/ /g' \
    | tr -s ' ' '\n' \
    | grep -Ev '^(the|a|an|is|in|to|for|of|on|at|by|up|do|if|or|be|we|it|no|via|and|not|use|any|all|when|with|that|this|from|into|each|more|also|then|does|but|are|has|was|its|per)$' \
    | grep -E '.{3,}' \
    | sort -u
}

score_skill() {
  local skill_file="$1"
  local prompt_tokens="$2"
  local score=0

  local desc
  desc="$(awk '/^description:/{sub(/^description: */, ""); gsub(/^"|"$/, ""); print; exit}' "$skill_file")"
  local desc_tokens
  desc_tokens="$(tokenize "$desc")"

  local body_raw
  body_raw="$(awk 'found && /^## /{exit} /^## When to Use/{found=1; next} found' "$skill_file")"
  local body_tokens
  body_tokens="$(tokenize "$body_raw")"

  while IFS= read -r tok; do
    [[ -z "$tok" ]] && continue
    if echo "$desc_tokens" | grep -qF "$tok"; then
      score=$((score + 3))
    elif echo "$body_tokens" | grep -qF "$tok"; then
      score=$((score + 2))
    fi
  done <<< "$prompt_tokens"

  echo "$score"
}

# ─── Golden Test Suite ───────────────────────────────────────────────────────
declare -a TEST_PROMPTS
declare -A TEST_EXPECTED

add_test() { TEST_PROMPTS+=("$1"); TEST_EXPECTED["$1"]="$2"; }

# scope-creep-guard
add_test "we need to check we're not touching out-of-scope files in this task" \
  "scope-creep-guard"
add_test "planning a refactor but want to confirm we stay in the declared phase boundary" \
  "scope-creep-guard"

# documentation-cohesion
add_test "update the architecture planning document to reflect the new authentication flow" \
  "documentation-cohesion"
add_test "integrate the new requirement into the existing spec without bolting it on at the end" \
  "documentation-cohesion"

# manual-testing-guides
add_test "write a manual test guide for the user login and token refresh flow" \
  "manual-testing-guides"
add_test "create a runbook with sunny and rainy path drills for the deployment health checks" \
  "manual-testing-guides"

# skill-map-governance
add_test "add a new skill for the payment processing domain and register it in the skill map" \
  "skill-map-governance"
add_test "remove the deprecated auth-flow skill and update the catalog" \
  "skill-map-governance"

# verification-gate
add_test "verify all changed files are consistent and no conflicting guidance was introduced" \
  "verification-gate"
add_test "check the refactored module does not break any existing contracts before completing" \
  "verification-gate"

# workflow-logging
add_test "log the workflow changes made in this session to the progress log" \
  "workflow-logging"
add_test "update the progress log after making instruction changes" \
  "workflow-logging"

# remote-commit-logging
add_test "set up automatic logging of pushed commits to a branch-sectioned commit log" \
  "remote-commit-logging"
add_test "maintain the pre-push hook automation for appending commit history" \
  "remote-commit-logging"

# detailed-chat-output
add_test "this is a multi-step migration that needs clear traceability and structured output" \
  "detailed-chat-output"
add_test "document each verification step of this high-risk refactor with explicit outcome reporting" \
  "detailed-chat-output"

# self-improvement-loop
add_test "we keep making the same mistake attaching event listeners — workflow docs may be stale" \
  "self-improvement-loop"
add_test "recurring errors suggest avoidable rework — improve the instructions to prevent recurrence" \
  "self-improvement-loop"

# skill-improvement-loop
add_test "the skill for error handling did not auto-load when it should have" \
  "skill-improvement-loop"
add_test "evaluate skill effectiveness and rewrite any SKILL.md files with trigger gaps" \
  "skill-improvement-loop"

# repo-workflow
add_test "discover which instruction files are relevant for the CI pipeline task" \
  "repo-workflow"
add_test "update the workflow instructions and customization files for the deployment process" \
  "repo-workflow"

# ─── Runtime ─────────────────────────────────────────────────────────────────
declare -a ALL_SKILLS=()
declare -A SKILL_NAME=()
while IFS= read -r f; do
  name="$(awk '/^name:/{print $2; exit}' "$f")"
  [[ -n "$name" ]] && ALL_SKILLS+=("$f") && SKILL_NAME["$f"]="$name"
done < <(find "$SKILLS_DIR" -mindepth 2 -maxdepth 2 -name SKILL.md | sort)

PASS=0
FAIL=0
declare -A SKILL_MISS_COUNT=()
declare -A SKILL_HIT_COUNT=()

for f in "${ALL_SKILLS[@]}"; do
  SKILL_MISS_COUNT["${SKILL_NAME[$f]}"]=0
  SKILL_HIT_COUNT["${SKILL_NAME[$f]}"]=0
done

declare -a RUN_PROMPTS=()
if [[ $# -gt 0 && "${1:-}" != "--report" ]]; then
  RUN_PROMPTS=("$@")
else
  RUN_PROMPTS=("${TEST_PROMPTS[@]}")
fi

for prompt in "${RUN_PROMPTS[@]}"; do
  expected_raw="${TEST_EXPECTED[$prompt]:-CUSTOM}"
  prompt_tokens="$(tokenize "$prompt")"

  declare -A cur_scores=()
  declare -A cur_loads=()
  for f in "${ALL_SKILLS[@]}"; do
    s="$(score_skill "$f" "$prompt_tokens")"
    cur_scores["$f"]=$s
    [[ $s -ge $LOAD_THRESHOLD ]] && cur_loads["$f"]=1 || cur_loads["$f"]=0
  done

  pass=true
  misses=""
  if [[ "$expected_raw" != "CUSTOM" ]]; then
    IFS=',' read -ra exp_arr <<< "$expected_raw"
    for exp in "${exp_arr[@]}"; do
      found=false
      for f in "${ALL_SKILLS[@]}"; do
        if [[ "${SKILL_NAME[$f]}" == "$exp" && ${cur_loads[$f]} -eq 1 ]]; then
          found=true; break
        fi
      done
      if ! $found; then
        pass=false
        misses="$misses $exp"
        SKILL_MISS_COUNT["$exp"]=$(( ${SKILL_MISS_COUNT[$exp]:-0} + 1 ))
      else
        SKILL_HIT_COUNT["$exp"]=$(( ${SKILL_HIT_COUNT[$exp]:-0} + 1 ))
      fi
    done
  fi

  if $REPORT_ONLY && $pass; then
    [[ "$expected_raw" != "CUSTOM" ]] && PASS=$((PASS+1))
    continue
  fi

  if $pass; then
    echo "✅ PASS | $prompt"
    PASS=$((PASS+1))
  else
    echo "❌ FAIL | $prompt"
    echo "   Expected but missed:$misses"
    FAIL=$((FAIL+1))
  fi

  if ! $REPORT_ONLY || ! $pass; then
    echo "   Top loaded skills:"
    for f in "${ALL_SKILLS[@]}"; do
      printf "     %d  %s\n" "${cur_scores[$f]}" "${SKILL_NAME[$f]}"
    done | sort -rn | head -6
    echo
  fi
done

echo "════════════════════════════════════════"
echo "  SKILL EFFECTIVENESS REPORT"
echo "════════════════════════════════════════"
printf "%-42s  %s / %s\n" "Skill" "Hits" "Opportunities"
echo "────────────────────────────────────────"
for f in "${ALL_SKILLS[@]}"; do
  name="${SKILL_NAME[$f]}"
  hits="${SKILL_HIT_COUNT[$name]:-0}"
  misses_count="${SKILL_MISS_COUNT[$name]:-0}"
  total=$(( hits + misses_count ))
  [[ $total -eq 0 ]] && continue
  rate=$(( hits * 100 / total ))
  bar=$(printf '█%.0s' $(seq 1 $((rate / 10))))
  printf "%-42s  %2d / %2d  (%3d%%)  %s\n" "$name" "$hits" "$total" "$rate" "$bar"
done | sort -t'(' -k2 -rn

echo "────────────────────────────────────────"
total_tests=$(( PASS + FAIL ))
echo "Suite: $PASS/$total_tests passed"
[[ $FAIL -gt 0 ]] && echo "ACTION NEEDED: $FAIL skill(s) have trigger gaps — run skill-improvement-loop"
echo "════════════════════════════════════════"

[[ $FAIL -gt 0 ]] && exit 1 || exit 0
