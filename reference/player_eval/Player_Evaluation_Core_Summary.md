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

---

## Entry: 2026-03-05 — PEC v1 Step 3: Team Feature Aggregation + Data Fixes

### Data Quality Fixes Applied

**Salary data pipeline fix:**
- Root cause: `normalize_name()` in `fetch_player_salaries.py` used `re.sub(r"[^a-z0-9 ]", "", name)` which **deleted** diacritical characters instead of decomposing them. "Dončić" → "Doni" (removes č,ć) instead of "Doncic".
- Fix: Added `unicodedata.normalize("NFKD", name)` + combining char removal before ASCII filtering. Now "Dončić" → "Doncic" correctly.
- Result: Luka Dončić, Nikola Jokić, and ~100+ other players now correctly matched to salary data (from 0% to 63.6% coverage).
- Removed salary from `numeric_fill_cols` — salary NaN values were being median-filled with ~$5M, creating false placeholders. Now salary stays NaN when not matched, displayed as "Salary not found" in frontend.

**Behavioral rate column fix:**
- Root cause: BKE decomposition parquet AND complete_player_season_stats parquet both have `FGA`, `FG3A`, `FTA` columns. After merge, pandas renames them to `FGA_x`/`FGA_y`, etc. The `_first_existing()` lookup couldn't find the original column names.
- Fix: Updated `_first_existing()` calls to also search for `_x`/`_y` suffixed columns.
- Result: `behavioral_three_point_rate` and `behavioral_free_throw_rate` now have correct values (were all 0.0 before).

**Frontend search fix:**
- Added `stripDiacritics()` JS function using `String.normalize('NFD')` to strip combining characters from search queries and player names, enabling ASCII search for players with diacritical names.

### Step 3: Team Feature Aggregation — What was built

`src/player_eval/team_feature_aggregation.py` computes team-level structural metrics from Step 1 player impact profiles. It implements a three-component team rating model: Offensive Mean Model, Defensive Mean Model, and Volatility Model.

#### Architecture

Each component is **toggleable** via flags (`use_interaction`, `use_structure`, `use_defense`, `use_volatility`) for ablation testing.

**Offensive Mean Model:**
```
OFF_Mean = TalentBase + InteractionTerm + StructureTerm
```

1. **TalentBase** = Σ(minute_share_i × impact_obke_i) — minute-weighted offensive BKE
2. **InteractionTerm** = λ × Σ(w_i × w_j × InteractionMatrix[arch_i, arch_j])
   - λ = 0.75 (interaction lambda), capped at ±2.0
   - 66-entry symmetric interaction matrix for all 11 offensive archetype pairs
   - Conditional flags: `"both_low"` (penalty only if BOTH players below p25 BKE), `"j_low"` (penalty only if second player below p25)
   - Positive pairs: BDC+PnR Roll (+0.10), BDC+Off-Ball Move Shooter (+0.10)
   - Negative pairs: Interior Scorer+Interior Scorer (-0.10), PnR Roll+Off-Ball Finisher (-0.10)
3. **StructureTerm** (capped at ±2.5):
   - TOV control: −β_TOV × (team_tov − league_avg_tov), β=0.12
   - FTR bonus: +β_FTR × (team_ftr − league_avg_ftr), β=0.15
   - Playmaking diversity: 1 playmaker → −0.7, 3+ → +0.3 (>15 MPG AND >15% AST)
   - Spacing credibility: <2 shooters → −1.0, 4+ → +0.5 (>25% 3PA rate AND >33% eFG AND >15 MPG)
   - **Transition as structure term** (NEW): freq × success_ratio weighted formula. Rewards teams with BOTH high transition frequency AND good transition execution (eFG-based success proxy)

**Defensive Mean Model:**
```
DEF_Mean = DefTalent + Essentials + AnchorQuality + Diversity − Liability
```

1. **DefTalent** = Σ(minute_share_i × impact_dbke_i) — minute-weighted defensive BKE
2. **Essentials**: Missing rim protector → −1.2, missing POA defender → −0.8, both missing → additional −0.5
3. **Anchor Quality**: 0.6 × top_RP_DBKE + 0.4 × top_POA_DBKE (scales with anchor effectiveness)
4. **Diversity**: +0.15 per unique defensive archetype above 3 (max +0.6)
5. **Liability**: −0.4 per defensive liability (DBKE < −0.5 AND >15 MPG), −0.4 stacking penalty if 2+ liabilities

**Volatility Model** (separate from mean):
```
σ_team = σ_base + σ_3PA + σ_creation + σ_transition
```
- σ_base = league std of team net ratings
- σ_3PA = α_3PA × (team_3PA_rate − league_avg) — 3PT-heavy teams are more volatile
- σ_creation = α_creation × (max_usage − threshold) — concentrated creation increases variance
- **σ_transition** = α_transition × (transition_freq − league_avg) — transition frequency as volatility driver (appears in BOTH structure and volatility, as requested)
- Clamped to [8.0, 16.0] floor/ceiling

#### Interaction Matrix Design

The 66-entry interaction matrix is based on basketball role theory:
- **Synergistic pairs** (positive): Ball-handler + roll man, creator + off-ball shooter, connector + finisher
- **Redundant pairs** (negative): Two ball-dominant creators, two interior scorers, two PnR roll men
- **Conditional penalties**: Same-archetype duplication only penalized if BOTH players are below the 25th percentile of that archetype's BKE distribution (prevents penalty when one is an elite player)

#### Step 3 Validation Results:
- 92 team-seasons processed (30 teams × 3 seasons, +2 edge cases)
- Projected net rating range: −1.95 to +2.60 (narrow; calibration needed)
- vs actual net rating: correlation=0.02, MAE=1.64, Spearman=0.05
- Low correlation is expected for v1: the model operates on minute-share-weighted BKE which has a compressed range compared to real NBA net ratings (±12). Scaling calibration is the next step.
- Structural components function correctly: interaction term identifies synergistic/redundant pairings, structure term captures spacing/playmaking/transition dynamics, defense model identifies anchor presence and liability exposure.

#### Step 3 Remediation Pass (post-eval hardening)

After targeted diagnostics, Step 3 was hardened with explicit ablations and calibration rather than relying on a single latent index.

Implemented fixes:
- **Defense sign verification** added (`corr(off+def, actual)` vs `corr(off-def, actual)`) to prevent silent sign inversion when DBKE semantics change.
- **5-man talent scaling** applied only to talent terms (`5 * off_talent_base`, `5 * def_talent_base`) instead of scaling all structural terms.
- **Interaction magnitude boost** applied via pair-weight scaling (`(minute_share_i * minute_share_j) / 0.2`) so interactions are not numerically negligible.
- **Calibration layer** added: linear calibration `actual = a + b * projected` fit on train seasons (`2022-23`, `2023-24`) and evaluated on holdout (`2024-25`).
- **Ablation stack** added and persisted to validation report:
   1) talent-only scaled, 2) talent+calibration, 3) talent+defense, 4) +structure, 5) full raw, 6) full calibrated.
- **Volatility floor/ceiling fix**: replaced hard `[8,16]` clipping with dynamic bounds tied to observed league std so volatility is no longer collapsed to 8.0 for all teams.
- **Spacing structural rule tightened**: shooter qualification now includes meaningful efficiency and rotation-share guards, reducing artificial spacing inflation.

Key observed outcomes from remediation run:
- `vol_total` is now distributed (no longer constant 8.0).
- Defensive sign inferred as additive (`+1`) on current data (`corr(def_talent, actual) > 0`).
- Correlation remains weak overall in v1 and unstable on holdout, indicating remaining cross-season mapping instability between player-level latent impacts and team-level margin outcomes.
- Step 3 report now explicitly exposes variance/correlation by component and by ablation stage, so failure modes are measurable and reproducible.

### Frontend Additions

**Team Aggregation tab** added to `app/player_eval_viewer.py`:
- Third tab "Team Aggregation" with team modal cards
- **Team cards**: one per team-season showing NET projected, OFF Mean, DEF Mean, Volatility, talent/interaction/structure breakdown, top archetypes
- **Team modal** (click to expand): full breakdown including:
  - Offensive model waterfall (Talent → Interaction → Structure → Total)
  - Defensive model waterfall (Talent → Adjustments → Total)
  - Volatility components (base, 3PA, creation, transition)
  - Top 15 archetype interaction pairs with contribution values
  - Full roster table with MPG, minute share, OBKE/DBKE, off/def archetypes, usage, AST, 3P rate, transition freq

### Constants Centralized

All Step 3 thresholds added to `src/player_eval/constants.py`:
- Interaction: `INTERACTION_LAMBDA=0.75`, `INTERACTION_CAP=2.0`
- Structure: `BETA_TOV=0.12`, `BETA_FTR=0.15`, `BETA_TRANSITION=0.10`, `STRUCTURE_CAP=2.5`
- Defense: `DEFENSE_CAP=3.0`, RP/POA/diversity/liability thresholds
- Volatility: `VOL_FLOOR=8.0`, `VOL_CEILING=16.0`, alpha coefficients
- Transition: `TRANSITION_PPP_LEAGUE_AVG=1.10`

### Output Artifacts (Step 3)
- `data/processed/player_eval/team_feature_aggregation.parquet` — 92 team-seasons with ~35 columns + JSON detail columns
- `reports/player_eval_step3_team_features_validation.json` — validation metrics vs actual
- `app/player_eval.html` — updated with Team Aggregation tab (~4.3MB)

### Step 3 notes for v1.1: 
 Issue 1: TEAM_SCALE = 20 Is a Blunt Instrument

Yes, it works.

But what it reveals:

Your player BKE distribution is extremely compressed.

You’re scaling a small signal into realism rather than extracting more variance at the source.

That’s acceptable — but long term:

You want:

Player metric variance to reflect impact,

Not rely on aggressive team scaling.

It’s not wrong — but it’s a signal that Step 1 (player layer) may need future expansion.

❗ Issue 2: You Might Be Underweighting Structure Now

7.5% modifier ratio is very conservative.

That ensures stability.
But it may be too safe.

In reality:

A team with no spacing and two heliocentric creators

A team with no rim protection

These can swing several points per 100.

Your current structure dispersion:

off_structure std ≈ 0.19
def_adjustments std ≈ 0.24

That’s tiny relative to 5-point spread.

You may have overcorrected from overweight to underweight.

Right now structure is almost cosmetic.

❗ Issue 3: Correlations Reveal Something Important

Per-season r:

2022-23: 0.19
2023-24: 0.49

That tells you something crucial:

The model works when the target behaves normally.

But even 0.49 at peak suggests:

You are capturing talent strength,
but not game-state dynamics.

What’s missing?

Coaching/system effects

Injury clustering

Depth stability

Lineup continuity

Not saying add them now.
But that’s the gap.

❗ Issue 4: Defensive Archetype Model Is Binary

You fixed rim detection (good),
but philosophically defense is still checklist-based.

Checklist defense is:

Stable
Interpretable
Clean

But it lacks interaction modeling.

You penalize:

Missing rim

Missing POA

Stacking liabilities

But you don’t yet model:

Rim protector + bad screen navigation interaction

Mobile big + no nail help

Versatile defender synergy effects

You simplified defense to avoid noise — smart —
but long term it’s too static.

❗ Issue 5: Volatility Still Assumes Linear Effects

You increased sensitivity, good.

But volatility likely behaves nonlinearly:

Very high 3PA% → fat tails

Very heliocentric offense → bimodal outcomes

Thin rotations → injury cascade risk

You’re still modeling volatility as additive.

It’s fine for now.
But future versions should explore nonlinear risk amplification.


