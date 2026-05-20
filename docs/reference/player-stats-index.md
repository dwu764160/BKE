# Player Stats Index

**Source of truth for all columns in `aggregate/player_profile_aggregate.parquet`.**

This file is the canonical manifest of every column available in the player profile aggregate.
It must be kept in sync with `aggregate/player_profile_aggregate.parquet` at all times.
When columns are added, removed, or renamed in the aggregate, update this file in the same change.

**Current shape:** ~1,971 rows × 933 columns (3 seasons: 2022-23, 2023-24, 2024-25)
**Key:** type `obj` = string/categorical, `f64` = float, `i64` = integer, `bool` = boolean

---

## Column Naming Conventions

The aggregate joins multiple source tables. Prefix conventions indicate origin:

| Prefix | Origin |
|---|---|
| *(none)* | Canonical / first join — use these |
| `box_` | NBA official box score advanced stats (duplicate for validation) |
| `arche_` | Offensive archetype table join (duplicate of no-prefix equivalents) |
| `defarche_` | Defensive archetype table join (duplicate of no-prefix equivalents) |
| `pos_` | Position estimate table join (duplicate of no-prefix equivalents) |
| `ml_` | Minute model / additional lookup join |
| `pec_` | Player Eval Composite — canonical input features used by simulation |
| `bke_` | BKE metric output (composite score, dimension details) |
| `dim_` | BKE 9-dimension z-scores |
| `agg_` | Aggregate-level outputs (final minute share etc.) |
| `diag_` | Diagnostic / internal model metadata |
| `rapm_` | RAPM model metadata |
| `gl_` | Game log derived statistics |
| `bio_` | Bio/biographical data |
| `s1_` | Simulation step1 join metadata |
| `pcv_` | Play-type Context Value (surplus + percentile rollup) |

---

## 1. Core Identity Keys

These columns uniquely identify a player-season row.

| Column | Type | Description |
|---|---|---|
| `season` | obj | NBA season string, e.g. `"2024-25"` |
| `player_id` | obj | NBA player ID (primary key) |
| `player_name` | obj | Full player name |
| `team_id` | f64 | NBA team ID |
| `team_abbreviation` | obj | 3-letter team abbreviation |
| `age` | f64 | Player age at season start |
| `nickname` | obj | Short first name |

---

## 2. RAPM (Regularized Adjusted Plus-Minus)

Core possession-level impact estimates. `orapm`/`drapm` are the primary offensive/defensive signals.

| Column | Type | Description |
|---|---|---|
| `rapm` | f64 | Total RAPM (pts/100 poss above average) |
| `orapm` | f64 | Offensive RAPM |
| `drapm` | f64 | Defensive RAPM |
| `possessions_played` | f64 | Total offensive possessions used in RAPM |
| `rapm_type` | obj | `pooled_split`, `single_season`, etc. |
| `poss_off` | f64 | Offensive possessions (raw) |
| `poss_def` | f64 | Defensive possessions (raw) |
| `x_rapm` | f64 | Prior-weighted xRAPM (box + tracking prior) |
| `career_prior` | f64 | Career RAPM prior used in regularization |
| `onoff_prior` | f64 | On-off prior |
| `is_collinear` | obj | Collinearity flag from RAPM solve |
| `xrapm_alpha` | f64 | Ridge alpha used for xRAPM |
| `drapm_bayesian` | f64 | Posterior-shrunk DRAPM |
| `drapm_raw_preshrink` | f64 | DRAPM before defensive Bayesian shrinkage |
| `n_seasons_pooled` | i64 | Seasons included in pooled RAPM |
| `alpha` | i64 | Ridge alpha for primary RAPM solve |
| `alpha_pooled` | f64 | Ridge alpha for pooled RAPM |
| `intercept` | f64 | RAPM model intercept (league average baseline) |
| `rapm_z` | f64 | RAPM z-score vs. league |
| `orapm_z` | f64 | ORAPM z-score vs. league |
| `drapm_z` | f64 | DRAPM z-score vs. league |
| `rapm_league_pctl` | f64 | RAPM league percentile |
| `orapm_league_pctl` | f64 | ORAPM league percentile |
| `drapm_league_pctl` | f64 | DRAPM league percentile |
| `rapm_position_bucket_pctl` | f64 | RAPM percentile within position group |
| `orapm_position_bucket_pctl` | f64 | ORAPM percentile within position group |
| `drapm_position_bucket_pctl` | f64 | DRAPM percentile within position group |
| `rapm_primary_archetype_pctl` | f64 | RAPM percentile within archetype cohort |
| `orapm_primary_archetype_pctl` | f64 | ORAPM percentile within archetype cohort |
| `drapm_primary_archetype_pctl` | f64 | DRAPM percentile within archetype cohort |
| `baseline_rapm_mean` | f64 | Cohort mean RAPM (archetype + position baseline) |
| `baseline_rapm_median` | f64 | Cohort median RAPM |
| `baseline_rapm_std` | f64 | Cohort std RAPM |
| `baseline_orapm_mean` | f64 | Cohort mean ORAPM |
| `baseline_orapm_median` | f64 | Cohort median ORAPM |
| `baseline_orapm_std` | f64 | Cohort std ORAPM |
| `baseline_drapm_mean` | f64 | Cohort mean DRAPM |
| `baseline_drapm_median` | f64 | Cohort median DRAPM |
| `baseline_drapm_std` | f64 | Cohort std DRAPM |
| `baseline_surplus_mean` | f64 | Cohort mean playtype surplus |
| `baseline_surplus_median` | f64 | Cohort median playtype surplus |
| `baseline_ts_mean` | f64 | Cohort mean TS% |
| `elevation_rapm` | f64 | RAPM above cohort baseline |
| `elevation_orapm` | f64 | ORAPM above cohort baseline |
| `elevation_drapm` | f64 | DRAPM above cohort baseline |
| `elevation_surplus` | f64 | Playtype surplus above cohort baseline |
| `elevation_efficiency` | f64 | TS% - cohort baseline TS% |
| `elevation_score_raw` | f64 | Composite elevation above cohort |
| `elevation_score` | f64 | Elevation score as league percentile |
| `elevation_z` | f64 | Elevation z-score |
| `elevation_tier` | obj | `Elite`, `Above Average`, `Average`, `Below Average` |
| `elevation_score_raw_primary_archetype_pctl` | f64 | Elevation pctl within archetype cohort |
| `elevation_score_raw_league_pctl` | f64 | Elevation pctl vs. league |

---

## 3. Box Score Totals

Season accumulation stats from NBA API (box score).

| Column | Type | Description |
|---|---|---|
| `gp` | f64 | Games played |
| `min` | f64 | Total minutes |
| `mpg` | f64 | Minutes per game |
| `pts` | f64 | Total points |
| `ast` | f64 | Total assists |
| `reb` | f64 | Total rebounds |
| `orb` | f64 | Offensive rebounds |
| `drb` | f64 | Defensive rebounds |
| `stl` | f64 | Total steals |
| `blk` | f64 | Total blocks |
| `tov` | f64 | Total turnovers |
| `fgm` | f64 | Field goals made |
| `fga` | f64 | Field goals attempted |
| `fg3_m` | f64 | Three-pointers made |
| `fg3_a` | f64 | Three-pointers attempted |
| `ftm` | f64 | Free throws made |
| `fta` | f64 | Free throws attempted |
| `pf` | f64 | Personal fouls |
| `pfd` | f64 | Personal fouls drawn |
| `blka` | f64 | Shots blocked against player |
| `plus_minus` | f64 | Total plus/minus |
| `w` | f64 | Team wins while player active |
| `l` | f64 | Team losses while player active |
| `w_pct` | f64 | Team win pct while player active |
| `nba_fantasy_pts` | f64 | NBA fantasy point total |
| `dd2` | f64 | Double-doubles |
| `td3` | f64 | Triple-doubles |
| `team_count` | f64 | Number of teams played for that season |

---

## 4. Box Score Rates and Efficiency

| Column | Type | Description |
|---|---|---|
| `ts_pct` | f64 | True shooting % |
| `efg_pct` | f64 | Effective FG% |
| `usg_rate` | f64 | Usage rate (possession-based, per 100) |
| `ast_pct` | f64 | Assist % (of teammate FGM while on court) |
| `tov_pct` | f64 | Turnover % (per 100 possessions) |
| `ortg` | f64 | On/off offensive rating |
| `drtg` | f64 | On/off defensive rating |
| `net_rtg` | f64 | Net rating |
| `fg3_pct` | f64 | Three-point % |
| `fg_pct` | f64 | Field goal % (from box source) |
| `ft_pct` | f64 | Free throw % |
| `ft_rate` | f64 | FTA / FGA |
| `pts_per36` | f64 | Points per 36 minutes |
| `ast_per36` | f64 | Assists per 36 minutes |
| `reb_per36` | f64 | Rebounds per 36 minutes |
| `tov_per36` | f64 | Turnovers per 36 minutes |
| `stl_per36` | f64 | Steals per 36 minutes |
| `blk_per36` | f64 | Blocks per 36 minutes |
| `fg3_a_per36` | f64 | Three-point attempts per 36 minutes |
| `ppg` | f64 | Points per game |
| `oreb_pct` | f64 | Offensive rebound rate (poss-based) |
| `dreb_pct` | f64 | Defensive rebound rate |
| `ts_zscore` | f64 | TS% z-score vs. league |
| `fg3_pct_adj` | f64 | TS-adjusted 3P% |
| `ts_pct_adj` | f64 | Sample-size adjusted TS% |

---

## 5. NBA Advanced Box (Official Hustle/Tracking)

NBA API advanced efficiency and on/off columns.

| Column | Type | Description |
|---|---|---|
| `e_off_rating` | f64 | Estimated offensive rating |
| `off_rating` | f64 | On-court offensive rating |
| `e_def_rating` | f64 | Estimated defensive rating |
| `def_rating` | f64 | On-court defensive rating |
| `e_net_rating` | f64 | Estimated net rating |
| `net_rating` | f64 | On-court net rating |
| `ast_to` | f64 | AST/TOV ratio |
| `ast_ratio` | f64 | AST ratio (per 100 touches) |
| `reb_pct` | f64 | Total rebound rate |
| `usg_pct` | f64 | Usage % (box) |
| `e_usg_pct` | f64 | Estimated usage % |
| `e_tov_pct` | f64 | Estimated turnover % |
| `tm_tov_pct` | f64 | Team turnover % while player on court |
| `pace` | f64 | Pace (possessions / 48 min) while on court |
| `e_pace` | f64 | Estimated pace |
| `pie` | f64 | Player Impact Estimate |
| `poss` | f64 | Possessions played (box) |

---

## 6. B-Ref Advanced (BPM / Win Shares)

Basketball-Reference box plus/minus and win share columns.

| Column | Type | Description |
|---|---|---|
| `ws` | f64 | Win Shares |
| `ows` | f64 | Offensive Win Shares |
| `dws` | f64 | Defensive Win Shares |
| `bpm` | f64 | Box Plus/Minus |
| `obpm` | f64 | Offensive Box Plus/Minus |
| `dbpm` | f64 | Defensive Box Plus/Minus |
| `vorp` | f64 | Value Over Replacement Player |
| `gmsc_avg` | f64 | Average Game Score |
| `pprod` | f64 | Point production (BRef formula) |
| `tot_poss` | f64 | Total possessions (BRef) |
| `q_ast` | f64 | Assist quality factor (BRef) |
| `marginal_off` | f64 | Marginal offensive contribution |
| `marginal_def` | f64 | Marginal defensive contribution |
| `position` | f64 | Positional scalar (BRef formula) |
| `off_role` | f64 | Offensive role factor (BRef formula) |

---

## 7. DARKO (External — Currently Empty)

DARKO daily player metric columns. All `N/A` until DARKO data is ingested.
See `src/data_fetch/ingest_darko.py` for ingestion path.

| Column | Type | Description |
|---|---|---|
| `darko_dpm` | obj | DARKO Daily Plus/Minus |
| `darko_odpm` | obj | DARKO Offensive DPM |
| `darko_ddpm` | obj | DARKO Defensive DPM |
| `box_ddpm` | f64 | Box-based DPM defensive |
| `box_odpm` | f64 | Box-based DPM offensive |
| `d_dpm` | f64 | DARKO d_dpm (alternative) |
| `dpm` | f64 | DARKO DPM total |
| `o_dpm` | f64 | DARKO o_dpm |
| `on_off_ddpm` | f64 | DARKO on/off DDPM |
| `on_off_odpm` | f64 | DARKO on/off ODPM |
| `darko_player_name` | obj | Player name from DARKO join |

---

## 8. Tracking Data (NBA API)

Per-game averages from NBA tracking endpoints. May be missing for pre-2022 seasons.

### Ball Handling / Touches
| Column | Type | Description |
|---|---|---|
| `touches` | f64 | Touches per game |
| `front_ct_touches` | f64 | Frontcourt touches per game |
| `time_of_poss` | f64 | Time of possession per game (sec) |
| `avg_sec_per_touch` | f64 | Average seconds per touch |
| `avg_drib_per_touch` | f64 | Average dribbles per touch |
| `dribbles_per_touch` | f64 | Dribbles per touch (alt calc) |
| `time_of_poss_per36` | f64 | Time of possession per 36 min |
| `passes_made` | f64 | Passes made per game |
| `passes_made_per36` | f64 | Passes made per 36 min |

### Drives
| Column | Type | Description |
|---|---|---|
| `drives` | f64 | Drives per game |
| `drives_per36` | f64 | Drives per 36 minutes |
| `drive_pts` | f64 | Points from drives per game |
| `drive_fg_pct` | f64 | FG% on drives |
| `drive_ast` | f64 | Assists from drives per game |
| `drive_tov` | f64 | Turnovers from drives per game |
| `drive_tov_rate` | f64 | TOV / (drives + drive_tov) |
| `drive_ast_ratio` | f64 | Drive assists as share of drive FGA |

### Shot Zones
| Column | Type | Description |
|---|---|---|
| `at_rim_freq` | f64 | At-rim FGA share (within 4 ft) |
| `at_rim_fg_pct` | f64 | At-rim FG% |
| `at_rim_plus_paint_freq` | f64 | At-rim + paint-non-RA FGA share |
| `paint_freq` | f64 | Paint FGA share (non-RA) |
| `paint_fga` | f64 | Paint FGA per game |
| `midrange_freq` | f64 | Midrange FGA share |
| `midrange_fg_pct` | f64 | Midrange FG% |
| `corner3_freq` | f64 | Corner 3 FGA share |
| `ab3_freq` | f64 | Above-the-break 3 FGA share |
| `ra_fga` | f64 | Restricted area FGA per game |
| `mr_fga` | f64 | Midrange FGA per game |
| `total_fga` | f64 | Total FGA per game (tracking) |
| `fg2_a_rate` | f64 | 2PA / FGA |
| `interior_ratio` | f64 | Interior shot share (2P) |

### Catch & Shoot
| Column | Type | Description |
|---|---|---|
| `catch_shoot_fgm` | f64 | Catch-and-shoot FGM per game |
| `catch_shoot_fga` | f64 | Catch-and-shoot FGA per game |
| `catch_shoot_fg_pct` | f64 | Catch-and-shoot FG% |
| `catch_shoot_fg3_m` | f64 | Catch-and-shoot 3PM per game |
| `catch_shoot_fg3_a` | f64 | Catch-and-shoot 3PA per game |
| `catch_shoot_fg3_pct` | f64 | Catch-and-shoot 3P% |
| `catch_shoot_pts` | f64 | Catch-and-shoot points per game |

### Pull-Up
| Column | Type | Description |
|---|---|---|
| `pull_up_fgm` | f64 | Pull-up FGM per game |
| `pull_up_fga` | f64 | Pull-up FGA per game |
| `pull_up_fg_pct` | f64 | Pull-up FG% |
| `pull_up_fg3_m` | f64 | Pull-up 3PM per game |
| `pull_up_fg3_a` | f64 | Pull-up 3PA per game |
| `pull_up_fg3_pct` | f64 | Pull-up 3P% |
| `pull_up_efg_pct` | f64 | Pull-up EFG% |
| `pull_up_pts` | f64 | Pull-up points per game |
| `pull_up_fga_per36` | f64 | Pull-up FGA per 36 min |

### Playmaking
| Column | Type | Description |
|---|---|---|
| `potential_ast` | f64 | Potential assists per game |
| `potential_ast_per36` | f64 | Potential assists per 36 min |
| `secondary_ast` | f64 | Secondary assists per game |
| `secondary_ast_per36` | f64 | Secondary assists per 36 min |
| `ast_points_created` | f64 | Points created via assist per game |
| `screen_assists` | f64 | Screen assists per game |
| `playmaking_score` | f64 | Composite playmaking index |

### Hustle / Defense Tracking
| Column | Type | Description |
|---|---|---|
| `deflections` | f64 | Deflections per game |
| `contested_shots` | f64 | Contested shots per game |
| `charges_drawn` | f64 | Charges drawn per game |
| `loose_balls_recovered` | f64 | Loose balls recovered per game |
| `def_loose_balls_recovered` | f64 | Defensive loose balls recovered |
| `hustle_score` | f64 | Composite hustle index |

### Movement
| Column | Type | Description |
|---|---|---|
| `dist_miles` | f64 | Miles run per game |
| `avg_speed` | f64 | Average speed (mph) |

### Rebounding
| Column | Type | Description |
|---|---|---|
| `oreb_contest` | f64 | Offensive rebounds contested per game |
| `dreb_contest` | f64 | Defensive rebounds contested per game |
| `reb_contest` | f64 | Total rebounds contested per game |

---

## 9. Playtype (Synergy) — Frequency and Efficiency

9 primary play types + offrebound + misc. Each has 3 columns.

| Pattern | `_poss_pct` | `_ppp` | `_poss` |
|---|---|---|---|
| `isolation` | Isolation freq | ISO PPP | ISO possessions/game |
| `prballhandler` | PnR ball handler freq | PnR BH PPP | PnR BH poss/game |
| `postup` | Post-up freq | Post PPP | Post poss/game |
| `cut` | Cut freq | Cut PPP | Cut poss/game |
| `prrollman` | PnR roll man freq | PnR roll PPP | PnR roll poss/game |
| `handoff` | Hand-off freq | H/O PPP | H/O poss/game |
| `offscreen` | Off-screen freq | OS PPP | OS poss/game |
| `spotup` | Spot-up freq | Spot-up PPP | Spot-up poss/game |
| `transition` | Transition freq | Trans PPP | Trans poss/game |
| `offrebound` | Off-reb putback freq | Putback PPP | Putback poss/game |
| `misc` | Misc playtype freq | Misc PPP | Misc poss/game |

**Derived playtype columns:**
| Column | Type | Description |
|---|---|---|
| `ball_dominant_pct` | f64 | ISO + PnR BH freq (ball dominant share) |
| `on_ball_creation` | f64 | ISO + PnR BH freq (alias) |
| `post_creation` | f64 | Post-up freq |
| `movement_shooter_pct` | f64 | Off-screen freq |
| `spotup_pct` | f64 | Spot-up freq (alias) |
| `transition_pct` | f64 | Transition freq (alias) |
| `cut_pnrrm_pct` | f64 | Cut + PnR roll-man combined |
| `putback_pct` | f64 | Off-rebound putback pct |

---

## 10. Playtype Surplus and PCValue

For each of 9 play types: how much above/below league average PPP vs. usage expectation.

| Pattern | `_surplus` | `_surplus_league_pctl` | `_surplus_z` | `pcv_` |
|---|---|---|---|---|
| `isolation` | ISO surplus | ISO surp pctl | ISO surp z | ISO PCValue |
| `prballhandler` | PnR BH surplus | PnR BH pctl | PnR BH z | PnR BH PCValue |
| `postup` | Post surplus | Post pctl | Post z | Post PCValue |
| `cut` | Cut surplus | Cut pctl | Cut z | Cut PCValue |
| `prrollman` | Roll surplus | Roll pctl | Roll z | Roll PCValue |
| `handoff` | H/O surplus | H/O pctl | H/O z | H/O PCValue |
| `offscreen` | OS surplus | OS pctl | OS z | OS PCValue |
| `spotup` | Spot-up surplus | Spot-up pctl | Spot-up z | Spot-up PCValue |
| `transition` | Trans surplus | Trans pctl | Trans z | Trans PCValue |

**Summary:**
| Column | Type | Description |
|---|---|---|
| `playtype_surplus_total` | f64 | Sum of all playtype surpluses |
| `playtype_surplus_total_league_pctl` | f64 | Total surplus league percentile |
| `playtype_surplus_total_z` | f64 | Total surplus z-score |
| `playtype_efficiency_z` | f64 | Overall playtype efficiency z |
| `pcv_dominant_playtype` | obj | Play type with highest PCValue |
| `pcv_entropy` | f64 | Playtype usage entropy (diversity) |

---

## 11. BKE 9-Dimension Model

The 9 portable talent dimensions. Each exists in multiple variants.

| Dimension | Canonical Column | Description |
|---|---|---|
| Shooting Gravity | `dim_shooting_gravity_z` | Spacing threat z-score |
| Driving Gravity | `dim_driving_gravity_z` | Drive-and-kick threat z-score |
| Playmaking Creation | `dim_playmaking_creation_z` | Assist + creation volume z-score |
| Extra Possession | `dim_extra_possession_z` | OREB + steal-caused possession z |
| Defensive Playmaking | `dim_defensive_playmaking_z` | Steal + defensive assist z |
| Defensive Impact | `dim_defensive_impact_z` | DRAPM + rim protection z |
| Turnover Control | `dim_turnover_control_z` | Low-TOV contribution z |
| Defensive Versatility | `dim_defensive_versatility_z` | Switch ability + coverage breadth z |
| Self Creation | `dim_self_creation_z` | ISO + PnR BH + drive creation z |

**Dimension variants (per above dimension, replace `<dim>`):**

| Suffix | Meaning |
|---|---|
| `dim_<dim>_z` | Bayesian-shrunk z-score (canonical, use this) |
| `dim_<dim>_z_raw` | Unshrunk raw z-score |
| `dim_<dim>_z_bayesian` | Explicit Bayesian posterior (same as `_z`) |
| `dim_<dim>_z_raw_preshrink` | Before defensive shrinkage step |
| `dim_<dim>_z_positional` | Z-score within position group |
| `dim_<dim>_z_archetype` | Z-score within archetype cohort |

**Composite scores:**
| Column | Type | Description |
|---|---|---|
| `offensive_portable_z` | f64 | Offensive dimension composite z |
| `defensive_portable_z` | f64 | Defensive dimension composite z |
| `dimension_model_z` | f64 | Full 9-dim composite z |
| `dimension_model_scale_factor` | f64 | Scale factor for dimension composite |
| `offensive_portable_z_positional` | f64 | Offensive composite z within position |
| `defensive_portable_z_positional` | f64 | Defensive composite z within position |
| `dimension_model_z_positional` | f64 | Full composite z within position |
| `offensive_portable_z_archetype` | f64 | Offensive composite z within archetype |
| `defensive_portable_z_archetype` | f64 | Defensive composite z within archetype |
| `dimension_model_z_archetype` | f64 | Full composite z within archetype |

---

## 12. BKE Scores (Composite)

Final BKE talent scores.

| Column | Type | Description |
|---|---|---|
| `bke_raw_obke` | f64 | Raw offensive BKE score |
| `bke_raw_dbke` | f64 | Raw defensive BKE score |
| `bke_raw_bke` | f64 | Raw total BKE score |
| `bke_transformed_bke` | f64 | BKE after non-linear transform |
| `bke_final_pctl` | f64 | BKE percentile (0–100) |
| `bke_obke_pctl` | f64 | Offensive BKE percentile |
| `bke_dbke_pctl` | f64 | Defensive BKE percentile |
| `bke_rank` | f64 | BKE league rank |
| `portable_talent_z` | f64 | Portable talent z-score |
| `portable_talent_z_adj` | f64 | Stability-weighted portable talent z |
| `portable_talent_raw` | f64 | Portable talent before transform |
| `portable_talent_score` | f64 | Portable talent as 0–100 percentile |
| `portable_talent_cdf_pctl` | f64 | Portable talent CDF percentile |
| `portable_talent_transformed` | f64 | Transformed portable talent score |
| `portable_talent_score_league_pctl` | f64 | Portable talent vs. league |
| `portable_talent_score_position_bucket_pctl` | f64 | Portable talent within position |
| `portable_talent_score_primary_archetype_pctl` | f64 | Portable talent within archetype |

**v30 variant (older BKE scoring approach, kept for continuity):**
| Column | Type | Description |
|---|---|---|
| `bke_v30` | f64 | BKE v30 composite |
| `dbke_raw_v30` | f64 | v30 raw DBKE |
| `dbke_norm_v30` | f64 | v30 normalized DBKE |
| `dbke_asym_v30` | f64 | v30 asymmetric DBKE |
| `dbke_final_v30` | f64 | v30 final DBKE |
| `dbke_scaled_v30` | f64 | v30 scaled DBKE |
| `obke_v30_reference` | f64 | v30 OBKE reference |

**BKE dimension detail objects (JSON dicts):**
| Column | Type | Description |
|---|---|---|
| `bke_dim_shooting_gravity` | obj | Dict: `{raw, z}` for this dimension |
| `bke_dim_driving_gravity` | obj | Dict: `{raw, z}` |
| `bke_dim_playmaking_creation` | obj | Dict: `{raw, z}` |
| `bke_dim_extra_possession` | obj | Dict: `{raw, z}` |
| `bke_dim_defensive_playmaking` | obj | Dict: `{raw, z}` |
| `bke_dim_defensive_impact` | obj | Dict: `{raw, z}` |
| `bke_dim_turnover_control` | obj | Dict: `{raw, z}` |
| `bke_dim_defensive_versatility` | obj | Dict: `{raw, z}` |
| `bke_dim_self_creation` | obj | Dict: `{raw, z}` |

---

## 13. Impact Composite Scores

| Column | Type | Description |
|---|---|---|
| `total_impact_z` | f64 | Combined portable+role z-score |
| `total_impact_raw` | f64 | Raw combined impact |
| `total_impact_transformed` | f64 | Non-linearly transformed impact |
| `total_impact_score` | f64 | Impact as 0–100 league percentile |
| `total_impact_cdf_pctl` | f64 | Impact CDF percentile |
| `total_impact_score_league_pctl` | f64 | Impact vs. league |
| `total_impact_score_position_bucket_pctl` | f64 | Impact within position |
| `total_impact_score_primary_archetype_pctl` | f64 | Impact within archetype |
| `role_dependent_impact_z` | f64 | Role-specific (non-portable) impact z |
| `role_dependent_impact_raw` | f64 | Raw role impact |
| `role_dependent_impact_transformed` | f64 | Transformed role impact |
| `role_dependent_impact_score` | f64 | Role impact 0–100 |
| `role_dependent_impact_score_league_pctl` | f64 | Role impact vs. league |
| `role_dependent_impact_score_position_bucket_pctl` | f64 | Role impact within position |
| `role_dependent_impact_score_primary_archetype_pctl` | f64 | Role impact within archetype |
| `impact_tier` | obj | `Star`, `Rotation`, `Fringe`, etc. |
| `stability_weight` | f64 | Sample-size stability weight applied to scores |

---

## 14. Offensive Archetype

Archetype embedding and classification results.

| Column | Type | Description |
|---|---|---|
| `primary_archetype` | obj | Assigned offensive archetype label |
| `secondary_archetype` | obj | Secondary archetype label |
| `role_confidence` | f64 | Confidence score for archetype assignment (0–1) |
| `role_effectiveness` | f64 | How well player executes their archetype role |
| `ball_dominance_tier` | obj | `Very High`, `High`, `Medium`, `Low` |
| `playmaking_tier` | obj | `Elite`, `High`, `Medium`, `Low` |
| `scoring_tier` | obj | Scoring tier |
| `efficiency_tier` | obj | Efficiency tier |
| `qualified` | bool | Meets minimum minutes threshold for archetype |
| `archetype_cohort_size` | f64 | Number of players in same archetype cohort |

**Archetype embedding probabilities (soft assignments, 11 archetypes):**
| Column | Description |
|---|---|
| `emb_ball_dominant_creator` | Ball-dominant creator probability |
| `emb_all_around_scorer` | All-around scorer probability |
| `emb_ballhandler` | Ballhandler probability |
| `emb_interior_scorer` | Interior scorer probability |
| `emb_perimeter_scorer` | Perimeter scorer probability |
| `emb_connector` | Connector probability |
| `emb_pnr_rolling_big` | PnR rolling big probability |
| `emb_pnr_popping_big` | PnR popping big probability |
| `emb_off_ball_finisher` | Off-ball finisher probability |
| `emb_off_ball_movement_shooter` | Movement shooter probability |
| `emb_off_ball_stationary_shooter` | Stationary shooter probability |
| `emb_entropy` | Shannon entropy of archetype embedding |
| `emb_entropy_norm` | Normalized embedding entropy |
| `emb_dominance` | Probability mass on primary archetype |

**Soft archetype probabilities (calibrated probs from classifier):**
`arch_prob_ball_dominant_creator`, `arch_prob_all_around_scorer`, `arch_prob_ballhandler`, `arch_prob_interior_scorer`, `arch_prob_perimeter_scorer`, `arch_prob_connector`, `arch_prob_pnr_rolling_big`, `arch_prob_pnr_popping_big`, `arch_prob_off_ball_finisher`, `arch_prob_off_ball_movement_shooter`, `arch_prob_off_ball_stationary_shooter`

| Column | Type | Description |
|---|---|---|
| `soft_archetype_entropy` | f64 | Entropy of soft archetype probabilities |
| `soft_archetype_entropy_norm` | f64 | Normalized soft entropy |

---

## 15. Defensive Archetype

| Column | Type | Description |
|---|---|---|
| `defensive_archetype` | obj | Primary defensive archetype label |
| `defensive_secondary` | obj | Secondary defensive archetype |
| `defensive_confidence` | f64 | Confidence in defensive assignment (0–1) |
| `defensive_effectiveness` | f64 | Defensive effectiveness score |
| `defensive_fit` | obj | `Elite`, `Good`, `Average`, `Poor` |
| `assignment_difficulty` | obj | Average difficulty of matchups: `High`, `Medium`, `Low` |
| `size_band` | obj | Physical size band: `Guard`, `Wing`, `Big` |
| `switch_score` | f64 | Switching ability (0–1) |
| `versatility_pctl` | f64 | Defensive versatility percentile |
| `rim_protection_index_pctl` | f64 | Rim protection percentile |
| `engagement_pctl` | f64 | Defensive engagement percentile |
| `engagement_score` | f64 | Raw engagement score |
| `hustle_pctl` | f64 | Hustle percentile |
| `d_results_pctl` | f64 | Opponent-adjusted results percentile |
| `matchup_diversity_pctl` | f64 | Breadth of matchup types percentile |
| `stl_per100_def_poss` | f64 | Steals per 100 defensive possessions |
| `blk_pct` | f64 | Block rate |
| `blk_pct` | f64 | Block pct |
| `stl_pctl` | f64 | Steal percentile |
| `blk_pctl` | f64 | Block percentile |
| `deflections_pctl` | f64 | Deflections percentile |
| `contested_shots_pctl` | f64 | Contested shots percentile |
| `avg_opponent_ppg` | f64 | Average points allowed per game on primary matchup |
| `elite_matchup_pct` | f64 | Pct of time guarding elite scorers |
| `difficulty_pctl` | f64 | Matchup difficulty percentile |
| `def_rim_fg_pct` | f64 | Opponent FG% at rim when player is defender |
| `rim_fga_rate` | f64 | Rim FGA allowed per game |
| `d_fg_diff` | f64 | Opponent FG% differential at position |

**Defensive role score breakdown:**
| Column | Description |
|---|---|
| `poa_score` | Point-of-attack defender score |
| `wing_score` | Wing stopper score |
| `chaser_score` | Off-ball chaser score |
| `versatile_score` | Versatile/switcher score |
| `rim_score` | Rim protector score |
| `drop_big_score` | Dropping big score |
| `mobile_big_score` | Mobile big score |
| `top_role_score` | Highest role score |
| `second_role_score` | Second-highest role score |
| `role_margin` | Gap between top and second role scores |

**Defensive coverage index percentiles:**
| Column | Description |
|---|---|
| `ball_pressure_index_pctl` | Ball pressure activity percentile |
| `screen_navigation_index_pctl` | Screen navigation skill percentile |
| `offball_navigation_index_pctl` | Off-ball navigation percentile |
| `drop_coverage_index_pctl` | Drop coverage tendency percentile |
| `switch_index_pctl` | Switch execution percentile |
| `help_activity_index_pctl` | Help defense activity percentile |
| `liability_index_pctl` | Defensive liability percentile (lower = better) |

---

## 16. Position Estimates

| Column | Type | Description |
|---|---|---|
| `primary_position_estimate` | obj | Canonical position: Guard / Guard-Forward / Forward / Forward-Center / Center |
| `position_band_5` | obj | 5-band position (same as `primary_position_estimate`) |
| `position_band_3` | obj | 3-band position: smalls / wings / bigs |
| `position_bucket` | obj | Broader bucket: Guard / Wing / Big |
| `primary_position` | obj | NBA-reported primary position |
| `position_estimate_method` | obj | Method used: `lineup_height_rank_v1`, etc. |
| `total_position_seconds` | f64 | Total tracked position seconds |
| `position_entropy` | f64 | Entropy of positional distribution (0 = pure) |
| `pct_pg` | f64 | Pct time at PG |
| `pct_sg` | f64 | Pct time at SG |
| `pct_sf` | f64 | Pct time at SF |
| `pct_pf` | f64 | Pct time at PF |
| `pct_c` | f64 | Pct time at C |
| `pct_guards_own` | f64 | Pct time as guard (PG+SG) |
| `pct_forwards_own` | f64 | Pct time as forward (SF+PF) |
| `pct_centers_own` | f64 | Pct time as center |

---

## 17. Bio and Draft

| Column | Type | Description |
|---|---|---|
| `height_inches` | f64 | Height in inches |
| `weight_lbs` | f64 | Weight in pounds |
| `wingspan_inches` | obj | Wingspan in inches (sparse) |
| `experience_years` | f64 | NBA experience in years |
| `bio_primary_position` | obj | NBA-reported position from bio table |
| `draft_class_year` | f64 | Draft year |
| `draft_round` | f64 | Draft round (1 or 2) |
| `draft_pick_in_round` | f64 | Pick number within round |
| `draft_pick_overall` | f64 | Overall draft pick |
| `draft_tier` | obj | `lottery`, `first_round`, `second_round`, `undrafted` |
| `is_drafted` | bool | True if player was drafted |
| `is_undrafted` | bool | True if undrafted |
| `draft_team_abbreviation` | obj | Team that drafted the player |
| `draft_source` | obj | Data source for draft info |
| `salary` | f64 | Current season salary (USD) |

---

## 18. Portability

| Column | Type | Description |
|---|---|---|
| `portability_dimensional_breadth` | f64 | How many dimensions are elite (vs. concentrated) |
| `portability_universal_skill` | f64 | Skills that translate across all team contexts |
| `portability_two_way_balance` | f64 | Offensive/defensive balance |
| `portability_scheme_independence` | f64 | Performance independent of team scheme |
| `portability_archetype_transfer` | f64 | Role flexibility across archetypes |
| `portability_index_raw` | f64 | Raw portability composite |
| `portability_index` | f64 | Portability as 0–1 score |
| `portability_class` | obj | `Scalable Star`, `Portable Role Player`, etc. |
| `portability_ratio` | f64 | Portability ratio (portable_talent / total_impact) |
| `portability_index_league_pctl` | f64 | Portability vs. league |
| `portability_index_position_bucket_pctl` | f64 | Portability within position |
| `portability_index_primary_archetype_pctl` | f64 | Portability within archetype |

---

## 19. Scheme and Lineup Context

| Column | Type | Description |
|---|---|---|
| `on_off_diff` | f64 | On/off point differential |
| `lineup_variance` | f64 | Variance in lineup-level performance |
| `lineup_interaction_coef` | f64 | Coefficient of teammate interaction |
| `on_off_stability` | f64 | Consistency of on/off diff across lineups |
| `teammate_dependency` | f64 | Estimated dependency on specific teammates |
| `scheme_stability_raw` | f64 | Raw scheme stability score |
| `scheme_stability_index` | f64 | Scheme stability as 0–100 |
| `scheme_amplification` | f64 | How much scheme enhances player (0–100) |
| `scheme_stability_raw_z` | f64 | Scheme stability z-score |
| `scheme_stability_z` | f64 | Scheme stability z (alias) |
| `scheme_stability_raw_league_pctl` | f64 | Scheme stability vs. league |
| `scheme_stability_raw_primary_archetype_pctl` | f64 | Scheme stability within archetype |
| `scheme_classification` | obj | `System-Dependent`, `System-Independent`, `System-Enhancer` |

---

## 20. Role Utilization

| Column | Type | Description |
|---|---|---|
| `role_utilization_raw` | f64 | How much player is used in their primary role |
| `role_utilization_efficiency` | f64 | Output per role opportunity |
| `role_utilization_raw_z` | f64 | Role utilization z-score |
| `role_utilization_raw_primary_archetype_pctl` | f64 | Utilization within archetype |

---

## 21. Clutch Stats

| Column | Type | Description |
|---|---|---|
| `clutch_team_id` | obj | Team ID for clutch stats source |
| `clutch_team_abbreviation` | obj | Team abbreviation |
| `clutch_gp` | f64 | Games with clutch minutes |
| `clutch_minutes` | f64 | Total clutch minutes |
| `clutch_pts` | f64 | Clutch points |
| `clutch_ast` | f64 | Clutch assists |
| `clutch_reb` | f64 | Clutch rebounds |
| `clutch_plus_minus` | f64 | Clutch +/- |

---

## 22. Game Log Derived

| Column | Type | Description |
|---|---|---|
| `gl_games_total` | f64 | Total games in game log |
| `gl_mpg_mean` | f64 | Average MPG from game log |
| `gl_mpg_std` | f64 | Std dev of MPG across games |
| `gl_mpg_median` | f64 | Median MPG |
| `gl_mpg_max` | f64 | Max MPG (single game) |
| `gl_mpg_min` | f64 | Min MPG (single game, excl. DNP) |
| `gl_dnp_count` | f64 | Number of DNPs |
| `gl_low_min_count` | f64 | Games with < 10 min |

---

## 23. PEC (Player Eval Composite) — Simulation Inputs

`pec_` columns are the canonical feature set used by the simulation step1 and step2 models.
These are the most important columns for downstream simulation logic.

### Impact Features
| Column | Type | Description |
|---|---|---|
| `pec_impact_orapm` | f64 | ORAPM for simulation |
| `pec_impact_drapm` | f64 | DRAPM for simulation |
| `pec_impact_bke` | f64 | BKE total score |
| `pec_impact_obke` | f64 | BKE offensive score |
| `pec_impact_dbke` | f64 | BKE defensive score |
| `pec_impact_stability` | f64 | Impact stability (0–1) |
| `pec_impact_ws` | f64 | Win Shares |
| `pec_impact_bpm` | f64 | Box Plus/Minus |
| `pec_impact_vorp` | f64 | VORP |
| `pec_impact_portable_talent` | f64 | Portable talent score |
| `pec_impact_total_impact` | f64 | Total impact score |

### Behavioral Features (Simulation Role Assignment)
| Column | Type | Description |
|---|---|---|
| `pec_behavioral_usage` | f64 | Usage rate |
| `pec_behavioral_assist_rate` | f64 | Assist rate |
| `pec_behavioral_turnover_rate` | f64 | Turnover rate |
| `pec_behavioral_three_point_rate` | f64 | 3PA / FGA |
| `pec_behavioral_rim_rate` | f64 | At-rim FGA rate |
| `pec_behavioral_efg` | f64 | Effective FG% |
| `pec_behavioral_orb_rate` | f64 | OREB rate |
| `pec_behavioral_drb_rate` | f64 | DREB rate |
| `pec_behavioral_free_throw_rate` | f64 | FTA / FGA |
| `pec_behavioral_hustle_pctl` | f64 | Hustle percentile |
| `pec_behavioral_foul_rate` | f64 | Personal fouls per 36 |

### Context Features
| Column | Type | Description |
|---|---|---|
| `pec_position_proxy` | obj | Position band for simulation |
| `pec_age` | f64 | Age |
| `pec_height_inches` | f64 | Height |
| `pec_weight_lbs` | f64 | Weight |
| `pec_experience_years` | f64 | NBA experience |
| `pec_minutes` | f64 | Season total minutes |
| `pec_mpg` | f64 | MPG |
| `pec_minute_share_potential` | f64 | Potential minute share (0–1) |
| `pec_possessions` | f64 | Possessions |
| `pec_games` | f64 | Games |
| `pec_on_off_diff` | f64 | On/off differential |
| `pec_scheme_stability_index` | f64 | Scheme stability |
| `pec_compression_flag` | f64 | Minutes compression flag (injured etc.) |
| `pec_defensive_shrinkage_lambda` | f64 | **LEAKAGE RISK**: = n/(n+2000), same-season possessions. See minute_model_rebuild_plan.md |
| `pec_salary` | f64 | Salary |
| `pec_availability_score` | f64 | Health/availability proxy (0–1) |

### Archetype Probability Arrays
| Column | Type | Description |
|---|---|---|
| `pec_playtype_vector` | obj | 11-element playtype frequency array |
| `pec_playtype_0..10` | f64 | Individual playtype frequencies (flat) |
| `pec_offensive_archetype_probs` | obj | Dict of 11 offensive archetype probabilities |
| `pec_defensive_archetype_probs` | obj | Dict of 7 defensive role probabilities |
| `pec_off_primary_archetype` | obj | Primary offensive archetype |
| `pec_off_secondary_archetype` | obj | Secondary offensive archetype |
| `pec_off_role_confidence` | f64 | Offensive archetype confidence |
| `pec_off_role_effectiveness` | f64 | Offensive archetype effectiveness |
| `pec_def_primary_archetype` | obj | Primary defensive archetype |
| `pec_def_secondary_archetype` | obj | Secondary defensive archetype |
| `pec_def_role_confidence` | f64 | Defensive archetype confidence |
| `pec_off_prob_emb_*` | f64 | 11 offensive archetype prob scalars (flat) |
| `pec_def_prob_*` | f64 | 7 defensive role prob scalars (flat) |

---

## 24. Aggregate Output

| Column | Type | Description |
|---|---|---|
| `agg_mpg` | f64 | Final projected MPG (simulation input) |
| `agg_minute_share` | f64 | Final projected minute share (0–1) |

---

## 25. Diagnostic / Internal

These columns are model internals, versioning metadata, or intermediate calculations.
Not intended for downstream consumption.

| Column | Description |
|---|---|
| `diag_*` | Various BKE diagnostic outputs |
| `d_port` | Defensive portable z |
| `d_rapm`, `d_rapm_shrunk`, `d_prior`, `d_stabilized` | DRAPM stabilization pipeline |
| `lambda_shrink` | Shrinkage lambda for DRAPM |
| `arch_scale` | Archetype scaling factor |
| `seasons` | Seasons in pooled estimate |
| `diag_d_stability`, `diag_o_stability` | RAPM year-over-year stability |
| `drapm_vol`, `dbke_vol`, `diag_vol_ratio` | Volatility diagnostics |
| `offensive_portable_mean`, `defensive_portable_mean` | Rolling mean across prior seasons |
| `def_elev_vol`, `defensive_fit_mean`, `entropy_mean` | Smoothing diagnostics |
| `obke_raw`, `dbke_raw`, `bke_raw`, `obke_z`, `dbke_z`, `bke_equal_var` | BKE intermediate |
| `rank_actual`, `rank_equal_var`, `rank_55_45`, `rank_60_40`, `rank_shift_*` | Rank sensitivity checks |

**RAPM model metadata:**
| Column | Description |
|---|---|
| `rapm_2`, `orapm_2`, `drapm_2` | Single-season RAPM (vs. pooled) |
| `rapm_rapm_type` | Model type for `rapm_2` |
| `rapm_possessions_played` | Possessions for model metadata join |
| `rapm_player_name` | Player name from RAPM table |

---

## 26. NBA Box Score Rank Columns

The NBA API returns league ranks for most box/advanced stats.
These are stored as `box_<stat>_rank` columns (e.g., `box_pts_rank`, `box_ast_rank`).
There are approximately 50 rank columns. They are not described individually here — consult the no-prefix stat for meaning and append `_rank` for the rank version.

---

## 27. Duplicate / Prefixed Source Columns

The following groups contain columns that are duplicates of canonical columns above, preserved from join operations. Use canonical (no-prefix) columns wherever possible.

| Prefix Group | Source | Canonical Equivalent |
|---|---|---|
| `box_fgm`, `box_fga`, `box_pts`, etc. | NBA box score table | `fgm`, `fga`, `pts`, etc. |
| `arche_fg_pct`, `arche_fg3_pct`, etc. | Archetype table join | `fg3_pct`, `at_rim_freq`, etc. |
| `defarche_gp`, `defarche_min`, etc. | Defensive archetype table | `gp`, `min`, etc. |
| `pos_gp`, `pos_min`, etc. | Position table join | `gp`, `min`, etc. |
| `ml_gp`, `ml_min` | Minute model table join | `gp`, `min` |
| `player_name_2`, `player_id_2` | Box score join artifact | `player_name`, `player_id` |
| `team_abbreviation_2`, `team_id_2` | Simulation join artifact | `team_abbreviation`, `team_id` |
| `age_2` | Join artifact | `age` |
| `usg_pct_2`, `oreb_pct_2`, `dreb_pct_2`, `ast_pct_2`, `tov_pct_2` | Box score advanced join | `usg_pct`, `oreb_pct`, `dreb_pct` |
| `ts_pct_2`, `e_fg_pct` | Join artifact | `ts_pct`, `efg_pct` |
| `pts_per36_2`, `ast_per36_2`, `reb_per36_2` | Archetype join | `pts_per36`, `ast_per36`, `reb_per36` |
| `fgm_pg`, `fga_pg` | Computed per-game | `fgm / gp`, `fga / gp` |
| `arche_player_name`, `arche_team_abbreviation` | Join artifact | `player_name`, `team_abbreviation` |
| `s1_player_name` | Sim step1 join | `player_name` |
| `player_name_v30` | v30 BKE join | `player_name` |
| `diag_player_name`, `diag_primary_archetype` | Diagnostic join | `player_name`, `primary_archetype` |
| `rapm_player_name` | RAPM join | `player_name` |
| `darko_player_name`, `team_name`, `tm_id` | DARKO join (empty) | (DARKO not ingested) |
| `arche_mpg`, `arche_tov_per36`, etc. | Archetype table rates | `mpg`, `tov_per36`, etc. |

---

## Sync Requirements

This file must be updated whenever:
- New data sources are joined into the aggregate (adds columns)
- Existing join keys are renamed (column prefix changes)
- New BKE metric versions are released (new `bke_v*` columns)
- `pec_` feature set changes (affects simulation input spec)
- Position band columns are added or changed

The fastest sync check:
```bash
python3 -c "
import pandas as pd
df = pd.read_parquet('aggregate/player_profile_aggregate.parquet')
print(f'{len(df.columns)} columns, {len(df)} rows')
print(df.dtypes.to_string())
"
```
