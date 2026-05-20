# Minute Model Rebuild Plan

> **Status:** Brainstormed, not yet implemented. Decisions finalized 2026-05-19.
> **Scope:** Fix same-season leakage, rebuild training regime, simplify architecture.
> **Preserves:** BKE metric philosophy, archetype structure, RAPM backbone.

---

## Problem Statement

The current minute model (`src/simulation/train_minute_model.py`) trains on **same-season features
to predict same-season MPG**. The dominant feature (`pec_defensive_shrinkage_lambda`, importance=0.476)
is literally `possessions_played_this_season / (possessions_played_this_season + 2000)` — a direct
function of the target variable. All other top features (usage, BKE, ORAPM, DRAPM, behavioral rates)
are also same-season values.

For **forecast deployment** the model is applied to prior-season feature values, but it was trained
on the in-season distribution — a severe distribution mismatch between training and deployment.

---

## Decisions

| Question | Decision |
|---|---|
| Training regime | Rebuild temporal: season-N features → season-N+1 MPG target |
| Architecture | Simplify to Ridge Regression (~15-20 features) |
| Rookie handling | Separate lookup table (draft_round + draft_pick → expected MPG) |
| `project_next_season.py` | Replace manual minute projection logic with temporal model output |
| Manual adjustments in `project_next_season.py` | Retain ONLY if they improve holdout MAE on walk-forward set |

---

## Training Data Construction

**New function:** `build_temporal_training_set(aggregate_df)` in `src/simulation/train_minute_model.py`

For each player who appears in consecutive seasons N and N+1:
- **Features** = season-N values (prior MPG, prior BKE, prior ORAPM, prior DRAPM, age at N+1,
  salary, prior usage rate, prior 3PT rate, prior AST rate, prior archetype embedding,
  draft position, roster competition from team context at N, prior GP fraction)
- **Target** = season-N+1 actual MPG

This is a join of consecutive player-season pairs, not same-season training. Excludes rookies
(no prior NBA season) → handled by separate rookie model.

---

## Feature Set (~15-20 features)

| Feature | Source | Rationale |
|---|---|---|
| `prior_mpg` | lag_mpg (already in aggregate) | Dominant signal — carry r=0.83-0.85 |
| `prior_bke` | lag_impact_bke | Impact quality: good enough to keep role? |
| `prior_orapm` | lag_impact_orapm | Offensive value |
| `prior_drapm` | lag_impact_drapm | Defensive value |
| `age` | current season (time-invariant) | Peak vs. decline |
| `prior_salary` | lag_salary | Organizational commitment |
| `prior_usage` | lag_behavioral_usage | Role type (ball-dominant vs. spot-up) |
| `prior_3pt_rate` | lag_behavioral_three_point_rate | Behavioral fingerprint |
| `prior_ast_rate` | lag_behavioral_assist_rate | Behavioral fingerprint |
| `draft_position` | draft_history (static) | Talent ceiling for young players |
| `experience_years` | derived from draft year | Seniority |
| `team_ctx_bke_rank` | team context (prior season) | Roster competition for minutes |
| `prior_gp_fraction` | lag availability score | Availability history |
| `prior_archetype_embedding` (top 3 dims) | lag archetype probs | Role continuity |

**Remove entirely:** `pec_defensive_shrinkage_lambda` (same-season possession count),
all `pec_impact_*` (current-season), all `pec_behavioral_*` (current-season), all
`pec_off_prob_*`/`pec_def_prob_*` (current-season archetype).

---

## Model Architecture

**Replace GBR** (70 features, prone to overfitting on thin data) **with Ridge Regression**.

Rationale: With the leakage removed, the dominant signal is prior MPG (r≈0.84). A 70-feature
GBR on ~1000 training samples (3 seasons × ~300 returning players per transition) is overparameterized.
Ridge with proper alpha cross-validation is more robust and interpretable.

```python
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=GroupKFold(n_splits=3))
# GroupKFold groups by season — prevents within-season leakage in CV
```

**Validation:**
- GroupKFold by season (same as current) for cross-validation
- Latest season as temporal holdout test (same as current)
- Target: holdout MAE < 2.36 MPG (current model's best) with honest temporal split

---

## Rookie Model (Separate)

Lookup table: `draft_round × draft_pick → expected first-year MPG`

Fit on all available rookie seasons in the dataset. Applied when `experience_years == 0`.

Example structure:
- Round 1 picks 1-5: expected ~24 MPG
- Round 1 picks 6-15: expected ~20 MPG
- Round 1 picks 16-30: expected ~16 MPG
- Round 2 picks: expected ~8 MPG

This replaces the current `ROOKIE_DEFAULT_USAGE = 0.18` flat estimate in `project_next_season.py`.

---

## Integration with `project_next_season.py`

Current `project_next_season.py` has its own parallel minute projection with:
- Hardcoded age curve breakpoints (22, 25, 27, 29, 31, 33) and deltas (+0.5% to -3.5%)
- `ENABLE_MINUTES_IMPACT_ADJUSTMENT = True` (impact → MPG slope: 0.03 per 0.1 BKE)
- `ENABLE_MINUTES_SALARY_ADJUSTMENT = True` (2.5x salary multiplier)
- `ENABLE_MINUTES_TEAM_COMPETITION_ADJUSTMENT = True` (max -0.40 penalty)

**Replace** the hardcoded projection with temporal model predictions as the base.

**Retain adjustments only if validated:**
- Run holdout evaluation with and without each adjustment factor
- Keep only adjustments that improve holdout MAE by >0.1 MPG on the walk-forward test set
- If an adjustment hurts or is neutral: remove it (it was a manually tuned prior, not a learned signal)

---

## Files to Modify

| File | Change |
|---|---|
| `src/simulation/train_minute_model.py` | New `build_temporal_training_set()`, replace GBR with Ridge, rebuild feature set |
| `src/player_eval/project_next_season.py` | Replace minute projection with temporal model output |
| `src/profile_aggregate/build_profile_aggregate.py` | Verify lag features are correctly constructed for all prior-season values needed |

---

## Verification

1. `python3 src/simulation/train_minute_model.py` → temporal holdout MAE and feature list logged
2. Distribution check: predicted MPG distribution should match actual MPG distribution (no systematic over/under-prediction)
3. Year-over-year: correlation between predicted and actual MPG should be ≥0.80 on temporal holdout
4. Compare against old model: run both on the 2024-25 holdout season, compare MAE

---

## Dependencies Before Implementation

- Phase 2 (data backfill) will add more training transitions and improve this model's fit
- Phase 3 (young player prior) will improve the RAPM inputs fed as features into this model
- Neither is strictly required to begin implementation — the rebuild can proceed with current 3-season data
