# Player Evaluation Core Summary

## Append-Only Rule
- Never delete sections, only update content of the sections.
- If it is a new version update, track the version number and date every time you update a section (only when user explicitly ask for version update)

---

## Entry: 2026-03-03 — PEC v1 Step 1 & Step 2 (initial implementation)

> Superseded by the 2026-03-04 entry below. Retained for history.

- Initial Step 1 built `PlayerImpactProfile` with 32 fields from 6 sources.
- Initial Step 2 trained minute-share model (target: share of team minutes, not raw MPG).
- Holdout MAE 0.0119 (share), R² 0.6155; 96% of feature importance was volume stats (minutes, possessions).
- XGBoost fallback to GradientBoostingRegressor due to missing runtime dependency.

---

## Entry: 2026-03-04 — PEC v1 Major Revision (Steps 1 & 2 rewrite, Profile Aggregate, Frontend Viewer)

### Philosophy & Design Principles

**Archetype = what you are; Impact = how well you do it.**

The Player Evaluation Core enforces a strict separation between *role identity* (archetype assignment, behavioral fingerprint) and *value* (impact metrics, BKE scores). This mirrors the repo's founding principle: a player's role is determined by behavior patterns, never by how effective they are. Effectiveness is layered on *after* role assignment as an independent overlay.

**Why predict MPG?** Minutes per game is the single strongest revealed-preference signal from NBA coaching staffs. Coaches allocate minutes based on a holistic (often subconscious) evaluation of player value that integrates skill, fit, matchups, development, and politics. A model that accurately predicts MPG from non-volume features has effectively learned what coaches value — which features of a player's profile translate into playing time. The residuals (actual − predicted MPG) reveal players who are over/under-utilized relative to their statistical profile.

**No volume leakage.** The MPG model explicitly excludes total minutes, games played, and possessions from the feature set. Including them would create circularity (predicting minutes from minutes). Instead, the model uses only behavioral rates, impact metrics, archetype probabilities, and contextual features — forcing it to learn *why* players get minutes, not just *how much* they already play.

### Step 1: Player Impact Profiles — What was built

`src/player_eval/build_player_impact_profiles.py` constructs a `PlayerImpactProfile` dataclass with **47 scalar fields + 3 complex fields** (offensive archetype probs, defensive archetype probs, playtype embedding) for every player-season in the pipeline.

#### Data sources merged (9 total):
1. `data/processed/modeling_inputs_all.parquet` — box-score stats, shooting splits, RAPM
2. `data/processed/player_archetypes.parquet` — 11-dimensional offensive archetype embedding
3. `data/processed/defensive_archetypes_v2.parquet` — 7-dimensional defensive archetype scores + hustle percentile
4. `data/processed/bke/BKE_Scores_v27.json` — OBKE, DBKE, BKE composite scores
5. `reports/bke_v29_player_diagnostic_report.json` — diagnostic stability metrics
6. `reports/dbke_v30_defense_shrinkage.json` — defensive shrinkage lambda
7. `data/processed/metrics_linear.parquet` — Win Shares, BPM, VORP
8. `data/historical/players.parquet` — height, weight, draft year, experience
9. `data/historical/player_salaries_*.parquet` — per-season salary data

#### Field categories:

**Impact metrics (11 fields):**
- `impact_orapm`, `impact_drapm` — Regularized Adjusted Plus-Minus (offensive/defensive)
- `impact_bke`, `impact_obke`, `impact_dbke` — Basketball KPI Engine score (composite, offensive, defensive)
- `impact_stability` — confidence-weighted stability index: `base_stability * (1 - shrinkage_lambda)` where base comes from v2.9 diagnostics
- `impact_ws`, `impact_bpm`, `impact_vorp` — Win Shares, Box Plus-Minus, Value Over Replacement Player
- `impact_portable_talent` — scheme-independent talent proxy from BKE Layer 1 (Bayesian-shrunk)
- `impact_total_impact` — full BKE pipeline composite (all layers applied)

**Behavioral fingerprint (11 fields):**
- `behavioral_usage` — USG% (proportion of team possessions used while on court)
- `behavioral_assist_rate` — AST% (assists per teammate FGM opportunity)
- `behavioral_turnover_rate` — TOV% (turnovers per play)
- `behavioral_three_point_rate` — FG3A/FGA (what fraction of shots are threes; *not* per-36)
- `behavioral_rim_rate` — rim attempt frequency from tracking data
- `behavioral_efg` — eFG% (effective field goal percentage, weights threes at 1.5x)
- `behavioral_orb_rate`, `behavioral_drb_rate` — offensive/defensive rebound rates (clipped 0-1)
- `behavioral_free_throw_rate` — FTA/FGA (free throw attempt rate, can exceed 1.0 — e.g., Embiid)
- `behavioral_hustle_pctl` — hustle percentile from defensive archetype engine (0-1 normalized)
- `behavioral_foul_rate` — personal fouls per 36 minutes

**Archetype probabilities:**
- `off_prob_*` (11 dimensions) — softmax-normalized offensive archetype embedding: `ball_dominant_creator`, `all_around_scorer`, `ballhandler`, `interior_scorer`, `perimeter_scorer`, `connector`, `pnr_rolling_big`, `pnr_popping_big`, `off_ball_finisher`, `off_ball_movement_shooter`, `off_ball_stationary_shooter`
- `def_prob_*` (7 dimensions) — min-max + re-normalized defensive scores: `poa_score`, `wing_score`, `chaser_score`, `versatile_score`, `rim_score`, `drop_big_score`, `mobile_big_score`
- Softmax temperature tau=0.5 for offensive archetypes (controls sharpness: lower tau = more peaked distribution)

**Playtype embedding (11 dimensions):**
- ISO, PnR ball-handler, post-up, cut, PnR roll-man, handoff, off-screen, spot-up, transition, offensive rebound, miscellaneous

**Context/bio (7 fields):**
- `age`, `height_inches`, `weight_lbs`, `experience_years`, `salary`, `mpg`, `position_proxy`
- `on_off_diff` — team's net rating differential when player is on court vs off (from BKE decomposition)
- `scheme_stability_index` — how much the player's value depends on team scheme context

#### Mathematical concepts in Step 1:

1. **Softmax normalization** — converts raw archetype embedding values to a probability distribution that sums to 1.0: `p_i = exp(x_i/tau) / sum(exp(x_j/tau))`. The temperature parameter tau controls the entropy of the distribution. At tau=0.5, players with dominant archetypes get sharper profiles (~60-70% in top archetype), while versatile players get flatter distributions.

2. **Min-max + re-normalization** — for defensive scores (which are not embedding dimensions), we first clip negatives to 0, then divide by the sum. This ensures each player's defensive profile is a valid probability simplex while preserving relative magnitudes.

3. **Stability formula** — `stability = base * (1 - lambda)` where base comes from the v2.9 diagnostic engine (measuring cross-season consistency of the player's BKE components) and lambda is the v3.0 shrinkage parameter (how much the defensive signal needed regularization). A player with high base consistency and low shrinkage gets stability near 1.0.

4. **Traded player deduplication** — when a player appears on multiple teams in one season, we keep only the row with the most minutes. This prevents split-season artifacts from inflating the dataset.

#### Step 1 validation results:
- 1978 player-season rows across 2022-23, 2023-24, 2024-25
- **ZERO** bounds violations (was 12 before fix — 10 in free_throw_rate, 2 in orb_rate)
- All offensive archetype probability vectors sum to 1.0 +/- 1e-6
- All defensive archetype probability vectors sum to 1.0 +/- 1e-6
- 100% impact coverage (BKE, WS, BPM, VORP)
- 91% bio coverage (height/weight/experience)
- 100% salary coverage

---

### Profile Aggregate — Foundational Data Layer

`src/profile_aggregate/build_profile_aggregate.py` merges **all 15 pipeline data sources** into a single comprehensive file: `aggregate/player_profile_aggregate.parquet`.

#### Purpose:
The aggregate is the canonical "everything we know about a player-season" file. It exists so that any downstream analysis (Step 2, future steps, exports) can read from one file instead of doing ad-hoc joins across 15 sources. It is the *spine* of all player evaluation.

#### Data sources merged:
1. **BKE Decomposition** (spine) — 379 columns of decomposed BKE components
2. **Complete box stats** — traditional and advanced box-score statistics
3. **Offensive archetypes** — 11-dim archetype embeddings + cluster labels
4. **Defensive archetypes v2** — 7 defensive scores + hustle/switchability percentiles
5. **Position estimates** — PG/SG/SF/PF/C probability shares
6. **Linear metrics** — WS, BPM, VORP, per-minute metrics
7. **RAPM** — ORAPM/DRAPM/RAPM values + residual stats
8. **xRAPM** — extended RAPM variants (v1, v2)
9. **DARKO** — external advanced metric projections
10. **BKE JSON scores** — final OBKE/DBKE/BKE + percentiles
11. **Player diagnostics** — v2.9 diagnostic domain scores
12. **Player bio/meta** — height, weight, draft year, country
13. **Salary data** — per-season contract salary
14. **Game log aggregates** — per-player game-level stats (MPG mean/std/median/max/min, DNP count, low-minute count)
15. **Step 1 profiles** — all PEC Step 1 fields (prefixed with `pec_`)

#### Merge strategy:
- BKE decomposition is the **spine** — all other sources are left-joined onto it
- Smart merge with conflict handling: when a column exists in both the spine and an incoming source, the incoming column is renamed with a source-specific suffix (e.g., `_stats`, `_rapm`)
- Traded player deduplication: keep max-minute row per (player_id, season)
- Game log aggregation: raw per-game logs are aggregated to player-season level (mean, std, median, max, min of MPG; DNP and low-minute counts)

#### Output:
- `aggregate/player_profile_aggregate.parquet` — 1971 rows x 908 columns, 771 unique players, 3 seasons

---

### Step 2: MPG Prediction Model — What was built

`src/player_eval/train_minute_model.py` trains a **Gradient Boosted Decision Tree (GBDT)** model to predict minutes per game (MPG) from non-volume features.

#### ML Concepts Used

1. **Gradient Boosted Decision Trees (GBDT)**
   - An ensemble of shallow decision trees trained sequentially, where each new tree corrects the residual errors of all previous trees.
   - At each step, the algorithm fits a new tree to the *negative gradient* of the loss function (MSE in our case), which equals the residual (actual - predicted).
   - Trees are combined additively: `F(x) = F0 + eta*h1(x) + eta*h2(x) + ...` where eta is the learning rate (0.03) and hk are individual trees.
   - Each tree has limited depth (max_depth=4), making individual trees weak learners that capture simple interaction patterns.
   - GBDT naturally handles non-linear relationships, feature interactions, and missing data patterns.

2. **Regularization (preventing overfitting)**
   - **Learning rate** (eta=0.03): shrinks each tree's contribution, requiring more trees to converge but producing smoother, more generalizable fits.
   - **Subsample** (0.85): each tree is fit on a random 85% of training rows (stochastic gradient boosting), adding noise that prevents overfitting to specific training examples.
   - **Max features** (0.7): each tree split considers only 70% of available features (random feature subsampling), decorrelating trees.
   - **Min samples split** (20): a node must have >=20 samples to be split, preventing micro-overfitting on small groups.
   - **Min samples leaf** (10): every terminal leaf must contain >=10 samples, ensuring predictions are based on adequate support.
   - **Early stopping** (n_iter_no_change=30, validation_fraction=0.15): training halts if validation loss doesn't improve for 30 consecutive rounds, automatically selecting the optimal number of trees.

3. **GroupKFold Cross-Validation**
   - Standard k-fold CV would randomly mix seasons, allowing the model to see 2023-24 data while predicting 2023-24 players in another fold — temporal data leakage.
   - GroupKFold assigns all rows from the same season to the same fold, ensuring the model never trains and tests on the same season simultaneously.
   - 3 folds (one per season): trains on 2 seasons, tests on the held-out 3rd.

4. **Temporal Holdout**
   - Beyond grouped CV, the most recent season (2024-25) is held out as a pure temporal test set.
   - The model is trained on 2022-23 + 2023-24, then evaluated on 2024-25.
   - This simulates the real-world use case: predicting *future* minutes from *past* data.

5. **Feature Engineering — No Volume Leakage**
   - Volume statistics (MIN, GP, POSS) are explicitly excluded from features.
   - This forces the model to learn *why* players get minutes (skill, role, impact) rather than tautologically predicting minutes from minutes.
   - Lag features use *prior-season* values only: `lag_mpg`, `lag_bke`, `lag_orapm`, `lag_usage`, `lag_salary`.

6. **Median Imputation**
   - Missing feature values are filled with the column median.
   - Median is robust to outliers (unlike mean) — important for NBA data where superstar stats can heavily skew distributions.

7. **Team Context Features**
   - `team_ctx_roster_size`: number of players on the team-season
   - `team_ctx_avg_bke`: team average BKE (captures team quality)
   - `team_ctx_bke_rank`: player's BKE rank within the team
   - `team_ctx_avg_usage`: team average usage rate
   - These are engineered *per team-season* to capture the competitive context.

8. **Optional Team Normalization**
   - After prediction, per-team MPG sums are scaled to approximate 240 (5 starters x 48 min).
   - This enforces the constraint that a team's total minutes are fixed.
   - Not applied by default (raw predictions are typically more useful for evaluation).

#### Training Pipeline (exact steps):

```
1. Load aggregate (1971 rows x 908 cols)
2. Filter: GP >= 5, MIN > 0 -> 1587 qualifying rows
3. Compute target: MPG = MIN / GP
4. Build lag features: for each player, retrieve prior-season values
5. Build team context: per-team-season aggregates
6. Select 70 features across 7 categories:
   - Impact (11): pec_impact_{bke,obke,dbke,orapm,drapm,stability,ws,bpm,vorp,portable_talent,total_impact}
   - Behavioral (11): pec_behavioral_{usage,assist_rate,turnover_rate,...}
   - Offensive archetypes (11): pec_off_prob_{ball_dominant_creator,...}
   - Defensive archetypes (7): pec_def_prob_{poa_score,...}
   - Playtypes (11): pec_playtype_{0..10}
   - Context (10): pec_age, pec_height_inches, pec_weight_lbs, pec_experience_years, pec_salary, pec_on_off_diff, pec_scheme_stability_index, compression_flag, shrinkage_lambda, minute_share_potential
   - Lag (5): lag_mpg, lag_bke, lag_orapm, lag_usage, lag_salary
   - Team context (4): team_ctx_roster_size, team_ctx_avg_bke, team_ctx_bke_rank, team_ctx_avg_usage
7. Median impute all NaN values
8. GroupKFold CV (3 folds by season)
9. Temporal holdout (train: 2022-23+2023-24, test: 2024-25)
10. Full fit on all qualifying data
11. Export: model pickle + predictions parquet + validation JSON
```

#### Step 2 validation results (v2):
- **1587 training rows**, 70 features
- **Holdout (2024-25):** MAE=2.36 MPG, RMSE=3.16, R2=0.884, Spearman=0.916
- **CV folds:** Fold1 MAE=2.29/R2=0.895, Fold2 MAE=2.36/R2=0.884, Fold3 MAE=2.92/R2=0.806
- **Full fit:** MAE=1.43, R2=0.952
- **Feature importance (top 8):**
  - `pec_defensive_shrinkage_lambda` (47.6%) — measures how much defensive data was regularized; proxy for playing time reliability
  - `pec_behavioral_hustle_pctl` (16.3%) — hustle effort percentile
  - `pec_impact_ws` (11.3%) — Win Shares
  - `pec_salary` (3.3%) — contract value (strong minute signal)
  - `lag_mpg` (2.9%) — prior-season minutes per game
  - `pec_behavioral_usage` (2.5%) — usage rate
  - `team_ctx_avg_bke` (1.5%) — team quality
  - `pec_impact_bke` (1.5%) — BKE score

#### Interpretation of feature importance:
- `shrinkage_lambda` dominating at 47.6% is meaningful, not circular: it captures how much a player's defensive signal needed regularization. High-minute starters have stable defensive data -> low lambda -> low shrinkage. Low-minute players have noisy defensive data -> high lambda -> high shrinkage. The model learned this association as a soft proxy for playing time reliability.
- The remaining 52.4% is spread across hustle (effort), Win Shares (production), salary (economic value signal), lag MPG (persistence), usage (offensive involvement), and team context — a diverse and basketball-meaningful feature mix.

---

### Frontend Viewer — What was built

`app/player_eval_viewer.py` generates a standalone interactive HTML file (`app/player_eval.html`, ~2.5MB with embedded data) for visualizing Steps 1 and 2.

#### Features:
- **Player Cards view:** sortable grid of player cards showing BKE badge, ORAPM/DRAPM/WS/USG%, offensive/defensive archetype probability strips, MPG bar with predicted vs actual
- **Team View:** tabular view grouped by team, showing actual vs predicted MPG, residuals, sorted by team rotation order
- **Detail Modal:** clicking any player opens a full-screen modal with:
  - Complete impact metrics (BKE, OBKE, DBKE, ORAPM, DRAPM, WS, BPM, VORP, portable talent, total impact, stability, on/off diff)
  - Minutes prediction comparison (actual, predicted, residual)
  - Full behavioral profile (usage, eFG%, AST%, TOV%, 3PA rate, rim freq, ORB%, DRB%, FT rate, hustle pctl, fouls/36)
  - Offensive archetype mix (11-bar horizontal chart)
  - Defensive archetype mix (7-bar horizontal chart)
  - Playtype distribution (11-bar horizontal chart)
  - Bio info (height, weight, experience, salary)
- **Filters:** season dropdown, team dropdown, name search
- **Sort options:** BKE, MPG, predicted MPG, ORAPM, WS, USG%
- **Dark theme** matching existing viewers (GitHub-style design tokens)

---

### Deviations from original plan and why

1. **Target changed from minute-share to MPG:** Minute-share normalization forced the model to predict a simplex-constrained value (team shares sum to 1.0). This made MAE misleadingly low and concentrated feature importance in volume stats. Switching to raw MPG (a natural, interpretable target) produced more diverse feature importance and more useful residuals.

2. **Volume features excluded:** The v1 model had 96% of feature importance in minutes + possessions — a circular tautology. The v2 model excludes these entirely, forcing the model to learn from *behavioral and impact* features.

3. **Profile Aggregate added as prerequisite:** Instead of having Step 2 read directly from Step 1 profiles, we introduced a comprehensive aggregate layer merging all 15 pipeline sources. This gives Step 2 access to lag features, team context, and richer metadata without Step 1 needing to carry every column.

4. **Feature count expanded from 56 to 70:** Added lag features (prior-season MPG/BKE/ORAPM/usage/salary) and team context features (roster size, team avg BKE, within-team BKE rank, team avg usage). These capture temporal consistency and competitive context.

5. **Step 1 fields expanded from 32 to 47+:** Added OBKE/DBKE, WS/BPM/VORP, portable_talent, total_impact, bio fields, salary, scheme_stability, and corrected several basketball-sense errors in behavioral metrics.

### Output artifacts (v2)
- `data/processed/player_eval/player_impact_profiles.parquet` — 1978 rows, 47+ fields
- `data/processed/player_eval/player_profiles_season.pkl` — pickle of per-season profile dicts
- `reports/player_eval_step1_validation.json` — Step 1 validation metrics
- `data/processed/player_eval/minute_model_v2.pkl` — trained GBDT model
- `data/processed/player_eval/minute_model_predictions_v2.parquet` — per-player MPG predictions
- `reports/player_eval_step2_minute_model_validation.json` — Step 2 validation report
- `aggregate/player_profile_aggregate.parquet` — 1971 rows x 908 columns
- `reports/profile_aggregate_validation.json` — aggregate data quality report
- `app/player_eval.html` — interactive frontend viewer (~2.5MB)
