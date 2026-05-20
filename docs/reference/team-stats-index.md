# Team Stats Index

**Source of truth for all columns in `data/processed/player_eval/team_feature_aggregation.parquet`.**

This file is the canonical manifest of every column in the team feature aggregate.
It must be kept in sync with `team_feature_aggregation.parquet` at all times.
When columns are added, removed, or renamed, update this file in the same change.

**Current shape:** 92 rows × 49 columns (3 seasons × 30-31 teams)
**Key:** type `obj` = string/categorical, `f64` = float, `i64` = integer

---

## Column Groups Overview

| Group | Columns | Description |
|---|---|---|
| Identity | 4 | Season, team, roster size, minutes |
| Offensive Model | 13 | Talent + adjustments that build off_mean |
| Defensive Model | 10 | Talent + adjustments that build def_mean |
| Team Net Rating | 2 | Raw and projected net rating |
| Star Concentration | 6 | Top-1/2/3 impact and minute share |
| Variance / Volatility | 10 | Game-to-game score volatility components |
| Archetype Distributions | 3 | Per-team offensive/defensive archetype breakdowns |

---

## 1. Identity

| Column | Type | Description |
|---|---|---|
| `season` | obj | NBA season string, e.g. `"2024-25"` |
| `team_abbreviation` | obj | 3-letter team abbreviation (primary key with season) |
| `n_players` | i64 | Number of players with qualifying minutes on roster |
| `team_total_minutes` | f64 | Total player-minutes accumulated by roster |

---

## 2. Offensive Model Components

These columns decompose how `off_mean` is assembled. The net is:
`off_mean = off_talent_base + off_interaction_term + off_structure_term`

All component values are in impact units (pts/100 above average, approximately).

| Column | Type | Description |
|---|---|---|
| `off_talent_base` | f64 | Minute-weighted sum of individual offensive BKE scores |
| `off_interaction_term` | f64 | Pairwise archetype synergy/conflict bonus/penalty |
| `off_structure_term` | f64 | Structure adjustments (playmaking, spacing, transition) |
| `off_mean` | f64 | **Canonical offensive talent score** (sum of above 3) |
| `off_tov_penalty` | f64 | Penalty for high team turnover rate |
| `off_ftr_bonus` | f64 | Bonus for high free-throw drawing |
| `off_playmaking_adj` | f64 | Bonus for having elite playmakers in lineup |
| `off_spacing_adj` | f64 | Bonus for shooter spacing (season-relative threshold) |
| `off_transition_bonus` | f64 | Bonus for fast-break frequency and efficiency |
| `off_n_playmakers` | i64 | Count of players meeting playmaker threshold |
| `off_n_shooters` | i64 | Count of players meeting spacing/shooter threshold |
| `off_team_transition_freq` | f64 | Team transition possession frequency |
| `off_team_transition_success` | f64 | Team transition PPP relative to league |

---

## 3. Defensive Model Components

These columns decompose how `def_mean` is assembled.

| Column | Type | Description |
|---|---|---|
| `def_talent_base` | f64 | Minute-weighted sum of individual defensive BKE scores |
| `def_adjustments` | f64 | Total structural defensive adjustments |
| `def_mean` | f64 | **Canonical defensive talent score** (talent_base + adjustments) |
| `def_rim_protector_penalty` | f64 | Penalty for lacking a rim protector |
| `def_poa_defender_penalty` | f64 | Penalty for lacking a point-of-attack defender |
| `def_both_missing_penalty` | f64 | Additional penalty when both rim + POA are absent |
| `def_anchor_bonus` | f64 | Bonus for having an elite defensive anchor |
| `def_diversity_bonus` | f64 | Bonus for having diverse defensive archetype coverage |
| `def_liability_penalty` | f64 | Penalty for having defensive liabilities in rotation |
| `def_n_liabilities` | i64 | Count of players flagged as defensive liabilities |
| `def_unique_archetypes` | i64 | Number of distinct defensive archetypes on roster |

---

## 4. Team Net Rating

Final projected team strength outputs used by the game model.

| Column | Type | Description |
|---|---|---|
| `team_net_rating_raw` | f64 | Pre-calibration net rating: `TEAM_SCALE * (off_mean - def_mean)` |
| `team_net_rating_projected` | f64 | Post-calibration projected net rating (used in simulation) |

> **Note:** `TEAM_SCALE` (currently 20.0) is a single multiplier that converts talent units to
> estimated points per game net rating. It is not yet calibrated via regression — see plan
> `docs/basketball_intuition_plan.md` Phase B for calibration roadmap.

---

## 5. Star Concentration

Captures how top-heavy the talent distribution is on a roster.

| Column | Type | Description |
|---|---|---|
| `star_top1_impact` | f64 | Impact score of the top player by BKE |
| `star_top2_impact` | f64 | Cumulative impact of top-2 players |
| `star_top3_impact` | f64 | Cumulative impact of top-3 players |
| `star_top1_share` | f64 | Top player's minute share (0–1) |
| `star_top2_share` | f64 | Cumulative minute share of top-2 |
| `star_concentration_penalty` | f64 | Penalty for over-reliance on a single star (top1_share too high) |

---

## 6. Variance / Volatility Model

Game-to-game score variance components. Used by the game model sigma computation.

| Column | Type | Description |
|---|---|---|
| `vol_base` | f64 | Baseline volatility (sigma anchor, in pts) |
| `vol_3pa` | f64 | Volatility adjustment for 3-point volume (negative = more volatile) |
| `vol_creation` | f64 | Volatility adjustment for self-creation ability |
| `vol_transition` | f64 | Volatility adjustment for transition play frequency |
| `vol_depth` | f64 | Volatility adjustment for roster depth |
| `n_sig_players` | i64 | Number of players with significant minute share |
| `vol_floor_used` | f64 | Effective volatility floor applied |
| `vol_ceiling_used` | f64 | Effective volatility ceiling applied |
| `vol_total` | f64 | **Final game-level volatility (sigma)** used in game model |

---

## 7. Archetype Distribution Objects

These are JSON-serialized dicts embedded in the parquet. Used for lineup analysis and debugging — not consumed directly by the simulation model.

| Column | Type | Description |
|---|---|---|
| `off_archetype_distribution` | obj | Dict mapping offensive archetype label → player count on roster |
| `def_archetype_distribution` | obj | Dict mapping defensive archetype label → player count |
| `interaction_details` | obj | List of dicts describing each pairwise archetype interaction applied to `off_interaction_term` |
| `player_summaries` | obj | List of per-player dicts: `{player_id, player_name, mpg, minute_share, impact_obke, impact_dbke, off_archetype, def_archetype, usage, ast_rate, tov_rate, three_rate, efg, transition_freq}` |

---

## Derived Columns Added at Simulation Time

These columns do not live in the parquet but are computed on-the-fly during game model execution.

| Column | Source | Description |
|---|---|---|
| `hca` | `simulation_config.get_team_hca(team_abbr)` | Per-team home court advantage (pts) |
| `pace` | `simulation_config.get_team_pace(season, team_abbr)` | Per-team-season possessions per 48 min |
| `is_b2b` | Schedule data | Back-to-back game flag (0 or 1) |
| `days_rest` | Schedule data | Days since last game (0–7) |
| `b2b_adj` | Fitted coefficients | B2B margin adjustment |
| `rest_adj` | Fitted coefficients | Rest-day margin adjustment |

---

## Sync Requirements

This file must be updated whenever:
- New offensive or defensive structure terms are added to `team_feature_aggregation.py`
- The volatility model gains or loses components
- The `player_summaries` schema changes (new per-player fields)
- The interaction matrix structure changes
- New seasons are added (row count changes but schema stays same)

The fastest sync check:
```bash
python3 -c "
import pandas as pd
df = pd.read_parquet('data/processed/player_eval/team_feature_aggregation.parquet')
print(f'{len(df.columns)} columns, {len(df)} rows')
for c in df.columns:
    print(f'  {c}  [{df[c].dtype}]')
"
```

---

## Column Stability Expectations

| Stability | Columns |
|---|---|
| **Stable** (change requires plan) | `season`, `team_abbreviation`, `off_mean`, `def_mean`, `team_net_rating_projected`, `vol_total` |
| **Likely to change** (calibration roadmap) | `team_net_rating_raw` (TEAM_SCALE will be replaced by OLS fit), `off_spacing_adj` (threshold becomes season-relative), `off_playmaking_adj` (threshold under review) |
| **Future additions** | `hca_fitted` (per-team HCA from reports/rest_hca_coefficients.json), `pace_fitted` (from reports/team_pace.json) — both currently computed at simulation time |
