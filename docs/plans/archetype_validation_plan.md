# Archetype Validation Plan

> **Status:** Brainstormed, not yet implemented. Decisions finalized 2026-05-19.
> **Scope:** Validate that offensive (11) and defensive (5) archetype classifications are
> stable, internally consistent, and meaningful. Add safeguards where they aren't.
> **Does not change:** Archetype definitions or basketball philosophy.

---

## Why This Plan Exists

The archetype system is downstream of nothing and upstream of everything:
- **Position-conditional z-scoring** uses archetype for cohort selection (Dims 4, 5, 6, 8)
- **Layer 2 role utilization** measures efficiency against archetype baseline
- **Layer 3 elevation** measures how much a player exceeds archetype expectations
- **Team structure modifiers** count playmakers, spacers, rim protectors by archetype
- **Interaction matrix** (121 pair values) is keyed entirely by archetype pairs
- **Minute model features** include archetype probability embeddings

If archetype assignments are noisy or year-over-year unstable, every downstream model inherits
that noise. The investigation found:
- 100+ feature classification with percentile-based thresholds → fragile to small shifts
- Heuristic-driven assignment order (BDC checked before Ballhandler, etc.) → swapping checks reclassifies 5-10% of players
- No external ground truth validation
- No stability metrics published

This plan establishes whether archetypes are trustworthy inputs to the rest of the system,
and adds safeguards where they aren't.

---

## Investigation Tracks (Run in Parallel)

### Track 1: Year-over-Year Stability

**Question:** When a player appears in consecutive seasons, how often does their archetype change?

**Methodology:**
For each player who appears in seasons N and N+1:
- Record archetype_N and archetype_N+1 (both offensive and defensive)
- Build 11×11 transition matrix (offensive) and 5×5 transition matrix (defensive)
- Diagonal entries: % of players who kept the same archetype
- Off-diagonal: which archetypes commonly transition to which

**Expected healthy result:**
- Diagonal ≥ 75% for offensive archetypes (most players don't fundamentally change role year-over-year)
- Diagonal ≥ 85% for defensive archetypes (defense is more positional, less volatile)
- Off-diagonal transitions should be "adjacent" (BDC ↔ Heliocentric Guard is fine; BDC ↔ Off-Ball Finisher is suspicious)

**Red flag patterns:**
- Diagonal < 60% → archetypes are not stable, downstream features are noisy
- Non-adjacent transitions (e.g., Interior Scorer → Movement Shooter) common → classifier is forcing assignments without confidence

**Output:** `reports/archetype_stability.json` with transition matrices + diagonal rates

---

### Track 2: Threshold Sensitivity Analysis

**Question:** How fragile is the classifier to small threshold perturbations?

**Methodology:**
Bootstrap the percentile thresholds by ±2 percentile points and re-run classification:
```python
# Baseline: P80 for BDC gate
# Perturbed: P78, P82
# Run classifier with each variant
# Measure: what % of players get reclassified
```

For each archetype, compute:
- Fragility score = % of assignments that change under ±2pp perturbation
- Players most affected (those near boundaries)

**Expected healthy result:**
- Fragility < 10% per archetype
- Affected players cluster at boundaries (acceptable — they're genuinely borderline)

**Red flag pattern:**
- Fragility > 20% → classifier is overfit to specific threshold values
- Specific archetypes show high fragility → those thresholds need reconsideration

**Output:** `reports/archetype_sensitivity.json` with per-archetype fragility scores

---

### Track 3: Internal Coherence

**Question:** Do players within the same archetype actually look similar?

**Methodology:**
For each archetype, compute the within-archetype distribution of key role features:
- Heliocentric Guards should have similar high usage, high assist rate, on-ball creation
- Off-Ball Finishers should have similar low usage, high rim rate, low 3PT volume
- Etc.

For each archetype, compute:
- Within-archetype standard deviation of usage, AST%, 3PT rate, rim rate
- Compare to overall qualified-player standard deviation
- Ratio = within_archetype_std / overall_std

**Expected healthy result:**
- Within-archetype std should be < 0.5x overall std for the archetype's defining features
- (Heliocentric Guards should have tight usage distribution)

**Red flag pattern:**
- Within-archetype std ≈ overall std → archetype isn't capturing a coherent role
- High variance specifically on the archetype's defining feature → classification is grouping unlike players

**Output:** `reports/archetype_coherence.json` per-archetype feature distributions

---

### Track 4: Manual Spot-Check Sample

**Question:** Do the classifications match basketball common sense?

**Methodology:**
Sample 30 random players per archetype (across all 3 seasons), print:
- Player name, season, team
- Assigned archetype
- Key features (usage, AST%, 3PT rate, position, height)

Have a basketball-literate reviewer (you) flag any obvious misclassifications.

**Output:** `reports/archetype_manual_sample.csv` for review, plus annotated results

---

## Remediation (Based on Investigation Findings)

### If Stability Track Fails (diagonal < 70%):

**Add temporal smoothing to archetype assignment**

For players with prior-season classification, blend current-season scores with prior:
```python
# Per archetype:
smoothed_score = 0.7 * current_season_score + 0.3 * prior_season_score
# Then classify based on smoothed scores
```

This reduces year-to-year flipping caused by small statistical noise while still allowing
genuine role changes (a player who clearly transitions roles will accumulate prior-season
score in the new archetype over time).

For rookies (no prior season): use current-season-only classification with a confidence flag.

### If Sensitivity Track Fails (fragility > 20%):

**Add archetype confidence scores**

Currently, archetype is a categorical assignment. Replace with:
- Primary archetype (current behavior)
- Confidence score (0-1, based on how close to threshold boundaries the player is)

Downstream code can:
- Use hard assignment as before (preserves current behavior)
- Optionally weight features by confidence (improves robustness)
- Flag low-confidence players (< 0.6) for review

Confidence is computed from distance to thresholds:
```python
confidence = min(
    (player_score - threshold_for_assigned) / (threshold_for_assigned - threshold_for_next_archetype),
    1.0
)
```

### If Coherence Track Fails (within-archetype std ≈ overall std):

**Investigate the specific archetype's threshold logic**

This is more invasive — it means the archetype isn't capturing what we think it is.
Revisit the threshold definitions for that specific archetype based on which features
are showing high variance. This is iterative and requires basketball judgment.

### If Manual Sample Reveals Systematic Errors:

**Refine threshold logic for the specific failure mode**

Example: if many "Wing Stoppers" are actually offensive wings with mediocre defense,
the defensive archetype gates need to be tighter on actual defensive impact, not just
defensive position.

---

## Validation Output Schema

After investigation, archetypes are tagged with:

```python
{
    "player_id": str,
    "season": str,
    "primary_archetype": str,
    "archetype_confidence": float,  # 0-1
    "prior_archetype": str | None,  # for stability tracking
    "is_stable": bool,  # True if prior == current OR confidence > 0.7
    "smoothed": bool,  # True if temporal smoothing was applied
}
```

---

## Files Touched

| File | Change |
|---|---|
| `src/data_compute/compute_player_archetypes.py` | Add confidence score output; optionally temporal smoothing |
| `src/data_compute/compute_defensive_archetypes_v2.py` | Same |
| `src/modeling/validate_archetypes.py` | **New** — runs all 4 tracks, outputs JSON reports |
| `src/player_eval/build_player_impact_profiles.py` | Propagate archetype_confidence into profile schema |

---

## Verification

1. Run `python3 src/modeling/validate_archetypes.py` → produces 4 JSON reports
2. Read `reports/archetype_stability.json` — diagonal rates per archetype
3. Read `reports/archetype_sensitivity.json` — fragility scores
4. Read `reports/archetype_coherence.json` — within-archetype distributions
5. Review `reports/archetype_manual_sample.csv` — sanity check
6. Make remediation decision per archetype based on findings

---

## Dependencies & Sequencing

**Can start:** After Plan 1 (data pipeline audit) completes — needs clean RAPM and BPM data
because archetype features depend on them.

**Must complete before:** Plan 3 Phase B (interaction matrix OLS), because validating the
matrix requires knowing the inputs (archetype pair counts) are stable.

**Output feeds:** Minute model rebuild (Plan 2), team feature aggregation, all downstream
modeling.
