#!/usr/bin/env bash
# =============================================================================
# scripts/validate_pipeline.sh
# Run all validation scripts against both data/ (original) and data_temp_reprod/
# (reproduced), then compare results and rank best-to-worst.
# =============================================================================
set -uo pipefail

WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WORKSPACE"

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

TMP="/tmp/bke_validate_$$"
mkdir -p "$TMP"/{orig,reprod,sed}

SCRIPTS=(
    validate_advanced_metrics
    validate_gamelogs
    validate_official_stats
    validate_possessions
    validate_rapm
    validate_tracking_data
    validate_ws_broad
)

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║            BKE VALIDATION PIPELINE                         ║"
echo "║            $(date '+%Y-%m-%d %H:%M:%S')                            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ─────────────────────────────────────────────────────────────────────
# PHASE 1: Run validation on ORIGINAL data (data/)
# ─────────────────────────────────────────────────────────────────────
echo "┌──────────────────────────────────────────────────────────────┐"
echo "│  PHASE 1: Validating ORIGINAL dataset (data/)               │"
echo "└──────────────────────────────────────────────────────────────┘"
echo ""

for s in "${SCRIPTS[@]}"; do
    echo "  ▶ Running ${s}.py ..."
    python3 "tests/${s}.py" > "$TMP/orig/${s}.log" 2>&1 || true
    echo "    Done ($(wc -l < "$TMP/orig/${s}.log") lines of output)"
done

echo ""

# ─────────────────────────────────────────────────────────────────────
# PHASE 2: Create sed-patched copies and run on REPRODUCED data
# ─────────────────────────────────────────────────────────────────────
echo "┌──────────────────────────────────────────────────────────────┐"
echo "│  PHASE 2: Validating REPRODUCED dataset (data_temp_reprod/) │"
echo "└──────────────────────────────────────────────────────────────┘"
echo ""

# Sed-patch: replace "data/ with "data_temp_reprod/ in all scripts
for s in "${SCRIPTS[@]}"; do
    sed 's|"data/|"data_temp_reprod/|g' "tests/${s}.py" > "$TMP/sed/${s}.py"
done

for s in "${SCRIPTS[@]}"; do
    echo "  ▶ Running ${s}.py (reproduced) ..."
    PYTHONPATH="$WORKSPACE" python3 "$TMP/sed/${s}.py" > "$TMP/reprod/${s}.log" 2>&1 || true
    echo "    Done ($(wc -l < "$TMP/reprod/${s}.log") lines of output)"
done

echo ""

# ─────────────────────────────────────────────────────────────────────
# PHASE 3: Compare & Rank (Python analysis)
# ─────────────────────────────────────────────────────────────────────
echo "┌──────────────────────────────────────────────────────────────┐"
echo "│  PHASE 3: Comparison & Rankings                             │"
echo "└──────────────────────────────────────────────────────────────┘"
echo ""

python3 - "$TMP" << 'PYEOF'
import sys, os, re, textwrap
from pathlib import Path

tmp = Path(sys.argv[1])

SCRIPTS = [
    "validate_advanced_metrics",
    "validate_gamelogs",
    "validate_official_stats",
    "validate_possessions",
    "validate_rapm",
    "validate_tracking_data",
    "validate_ws_broad",
]

# Friendly labels for display
LABELS = {
    "validate_advanced_metrics": "Advanced Metrics (BPM/WS/VORP)",
    "validate_gamelogs":        "Game Logs (Player & Team)",
    "validate_official_stats":  "Official Advanced Stats",
    "validate_possessions":     "Possession Data",
    "validate_rapm":            "RAPM / xRAPM",
    "validate_tracking_data":   "Tracking Data Coverage",
    "validate_ws_broad":        "Win Shares (Broad)",
}


def score_log(path):
    """Parse a validation log and return (pass_count, warn_count, fail_count, score).

    Scoring rules:
      - Each  ✅  → +2
      - Each  ⚠  (⚠️) → -1
      - Each  ❌  → -3
      - Text-only PASS / WARNING / FAIL / ERROR on lines without emojis → same weights
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return 0, 0, 0, -100  # missing = severe error

    p = text.count("\u2705")          # ✅
    w = text.count("\u26A0")          # ⚠ (also catches ⚠️ since ⚠️ = ⚠ + \uFE0F)
    f = text.count("\u274C")          # ❌

    # Supplement with text markers on lines that lack emojis
    for line in text.splitlines():
        has_emoji = ("\u2705" in line or "\u26A0" in line or "\u274C" in line)
        if has_emoji:
            continue
        if re.search(r"\bPASS\b", line):
            p += 1
        if re.search(r"\bWARNING\b", line, re.I):
            w += 1
        if re.search(r"\b(FAIL|ERROR)\b", line, re.I):
            f += 1

    score = p * 2 - w * 1 - f * 3
    return p, w, f, score


def extract_key_metrics(path):
    """Try to pull numeric quality indicators from a log file."""
    metrics = {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return metrics

    # MAE values  (e.g.  "WS   MAE:  0.42")
    for m in re.finditer(r"(\w+)\s+MAE:\s*([\d.]+)", text):
        metrics[f"{m.group(1)}_MAE"] = float(m.group(2))

    # Pearson r  (e.g.  "Pearson r:  0.812")
    for m in re.finditer(r"Pearson r:\s*([\d.]+)", text):
        metrics["Pearson_r"] = float(m.group(1))

    # Spearman ρ
    for m in re.finditer(r"Spearman [ρρ]:\s*([\d.]+)", text):
        metrics["Spearman_rho"] = float(m.group(1))

    # Global ORTG
    for m in re.finditer(r"Global ORTG:\s*([\d.]+)", text):
        metrics["ORTG"] = float(m.group(1))

    # Top-10 overlap
    for m in re.finditer(r"Top-10 overlap:\s*(\d+)/10", text):
        metrics["Top10_overlap"] = int(m.group(1))

    # Average Error %  (validate_ws_broad)
    for m in re.finditer(r"AVERAGE ERROR.*?(\d+\.\d+)%", text):
        metrics["Avg_WS_Err%"] = float(m.group(1))

    return metrics


# ── Gather scores ──────────────────────────────────────────────────
orig_data = {}
reprod_data = {}
orig_metrics = {}
reprod_metrics = {}

for s in SCRIPTS:
    orig_data[s]    = score_log(tmp / "orig" / f"{s}.log")
    reprod_data[s]  = score_log(tmp / "reprod" / f"{s}.log")
    orig_metrics[s]   = extract_key_metrics(tmp / "orig" / f"{s}.log")
    reprod_metrics[s] = extract_key_metrics(tmp / "reprod" / f"{s}.log")


# ── Comparison Table ───────────────────────────────────────────────
hdr = f"{'VALIDATION AREA':<36}│{'ORIGINAL':^27}│{'REPRODUCED':^27}│{'Δ':^10}"
sep = "─" * 36 + "┼" + "─" * 27 + "┼" + "─" * 27 + "┼" + "─" * 10
print(hdr)
print(sep)

for s in SCRIPTS:
    label = LABELS.get(s, s)
    op, ow, of, osc = orig_data[s]
    rp, rw, rf, rsc = reprod_data[s]
    delta = rsc - osc
    tag = " ✅" if delta > 0 else " ⚠️" if delta < 0 else " ="
    print(f"{label:<36}│ {op:>3}✅ {ow:>3}⚠  {of:>3}❌  {osc:>4} │ {rp:>3}✅ {rw:>3}⚠  {rf:>3}❌  {rsc:>4} │ {delta:>+5}{tag}")

print(sep)
orig_total  = sum(v[3] for v in orig_data.values())
reprod_total = sum(v[3] for v in reprod_data.values())
delta_total = reprod_total - orig_total
tag_t = " ✅" if delta_total > 0 else " ⚠️" if delta_total < 0 else " ="
print(f"{'TOTAL':<36}│ {'':>3}  {'':>3}   {'':>3}   {orig_total:>4} │ {'':>3}  {'':>3}   {'':>3}   {reprod_total:>4} │ {delta_total:>+5}{tag_t}")

# ── Key Numeric Metrics Comparison ──────────────────────────────────
print("\n")
print("KEY NUMERIC METRICS")
print("─" * 70)
print(f"{'Metric':<30} {'Original':>15} {'Reproduced':>15} {'Better?':>8}")
print("─" * 70)

all_keys = set()
for s in SCRIPTS:
    all_keys.update(orig_metrics[s].keys())
    all_keys.update(reprod_metrics[s].keys())

for key in sorted(all_keys):
    # find which script has this key
    ov = rv = None
    for s in SCRIPTS:
        if key in orig_metrics[s]:
            ov = orig_metrics[s][key]
        if key in reprod_metrics[s]:
            rv = reprod_metrics[s][key]

    ov_s = f"{ov:.3f}" if ov is not None else "N/A"
    rv_s = f"{rv:.3f}" if rv is not None else "N/A"

    # Determine which is better (lower MAE/Err = better, higher corr = better)
    better = ""
    if ov is not None and rv is not None:
        if "MAE" in key or "Err" in key:
            better = "Orig" if ov < rv else "Reprod" if rv < ov else "Tied"
        else:
            better = "Orig" if ov > rv else "Reprod" if rv > ov else "Tied"

    print(f"{key:<30} {ov_s:>15} {rv_s:>15} {better:>8}")


# ── Rankings ────────────────────────────────────────────────────────
print("\n")
print("┌──────────────────────────────────────────────────────────┐")
print("│  ORIGINAL (data/) — Ranked Best → Worst                  │")
print("└──────────────────────────────────────────────────────────┘")
for rank, (s, (p, w, f, sc)) in enumerate(
    sorted(orig_data.items(), key=lambda x: -x[1][3]), 1
):
    label = LABELS.get(s, s)
    print(f"  {rank}. [{sc:>4}]  {label:<36} ({p}✅  {w}⚠  {f}❌)")

print()
print("┌──────────────────────────────────────────────────────────┐")
print("│  REPRODUCED (data_temp_reprod/) — Ranked Best → Worst    │")
print("└──────────────────────────────────────────────────────────┘")
for rank, (s, (p, w, f, sc)) in enumerate(
    sorted(reprod_data.items(), key=lambda x: -x[1][3]), 1
):
    label = LABELS.get(s, s)
    print(f"  {rank}. [{sc:>4}]  {label:<36} ({p}✅  {w}⚠  {f}❌)")


# ── Overall Verdict ─────────────────────────────────────────────────
print()
print("═" * 62)
if delta_total > 0:
    print(f"  VERDICT: Reproduced data IMPROVED overall  (+{delta_total} points) ✅")
elif delta_total < 0:
    print(f"  VERDICT: Original data scores HIGHER  ({delta_total} points) ⚠️")
else:
    print(f"  VERDICT: Both datasets have EQUAL quality scores  =")

print(f"  Original total: {orig_total}  |  Reproduced total: {reprod_total}")
print("═" * 62)
print()
print(f"  Full logs saved at: {tmp}/")
print(f"    Original:   {tmp}/orig/<script>.log")
print(f"    Reproduced: {tmp}/reprod/<script>.log")
print()
PYEOF

echo ""
echo "All done."
