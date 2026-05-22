# Archetype Validation Plan

> **Status:** COMPLETE 2026-05-21. All sub-tasks 4.0–4.6 done. 4.5 deferred to Phase 2.
> **Scope:** Validate, document, and improve offensive and defensive archetype classifications
> before Phase 2 (minute model rebuild) uses archetypes as features.
> **Does not change:** Core archetype definitions or basketball philosophy.
> **Position-agnostic design is intentional** — see `docs/reference/basketball-intuitions.md §4`.

---

## Why This Plan Exists

The archetype system is upstream of everything:
- **Position-conditional z-scoring** uses archetype for cohort selection (Dims 4, 5, 6, 8)
- **Layer 2 role utilization** measures efficiency against archetype baseline
- **Layer 3 elevation** measures how much a player exceeds archetype expectations
- **Team structure modifiers** count playmakers, spacers, rim protectors by archetype
- **Interaction matrix** (121 pair values) is keyed entirely by archetype pairs
- **Minute model features** include archetype probability embeddings

If archetype assignments are noisy or year-over-year unstable, every downstream model inherits
that noise. Known issues going into this phase:
- 32% "Insufficient Minutes" (threshold was 500 min — lowered to 200)
- Hard binary classification gates → 3,498 players at max certainty (1.0 entropy)
- No secondary archetype documentation
- Archetypes only backfilled for 2022-23 to 2024-25 (3 seasons)
- No external ground truth validation
- No stability metrics published

---

## Sub-Task Ordering

```
4.0  Archetype backfill → all 8 seasons           [COMPLETE]
4.1  Threshold lower → 200 min                    [COMPLETE — code change applied]
4.2  Secondary archetype documentation             [COMPLETE — offensive tags finalized, defensive tags eliminated]
4.3  Validation tracks (4 parallel)               [COMPLETE — reports/archetype_stability/sensitivity/coherence.json, archetype_manual_sample.csv]
4.4  Player tier system                            [COMPLETE — player_tier column in player_profile_aggregate.parquet]
4.5  Soft probability adoption                     [DEFERRED — design only, no new model; Phase 2 minute model rebuild]
4.6  Archetype pair validation + matrix OLS        [COMPLETE — 0 pairs survive; INTERACTION_MATRIX eliminated (use_interaction=False)]
```

---

## 4.0 — Archetype Backfill (Prerequisite, Done)

Pre-2022 seasons (2017-18 through 2021-22) had `None` for all archetype columns.
Ran `compute_player_archetypes.py` and `compute_defensive_archetypes_v2.py` for all 8 seasons.
All downstream artifacts rebuilt (impact profiles + aggregate).

---

## 4.1 — Minutes Threshold Change (Done)

Changed in `src/data_compute/compute_player_archetypes.py`:

| Parameter | Before | After |
|---|---|---|
| `MIN_MINUTES` | 500 | 200 |
| `MIN_GP` | 20 | 10 |
| `MIN_MPG` | 15.0 | 8.0 |

**Note:** League-average reference pool for BPM and z-scores remains ≥500 min to prevent
fringe-player noise from corrupting the reference distribution. The 200 min threshold only
controls the "Insufficient Minutes" classification gate.

---

## 4.2 — Secondary Archetype Documentation

Secondary tags are human-readable explainers for primary archetype assignments. They:
- Clarify *what kind* of a given primary archetype a player is
- Have **no computational effect** — z-scores, interaction matrix, and model features all
  use primary archetype only
- Are not assigned to every player — only where a meaningful sub-role exists

### Offensive Secondary Tags (12 tags)

| Tag | Applied to | What it means |
|---|---|---|
| **Heliocentric** | Ball Dominant Creator | Runs entire offense through themselves at extreme rate; defense schemes around stopping them specifically |
| **Post Creator** | Ball Dominant Creator, Interior Scorer | Creates primarily from post position; high post-up possession% |
| **Gravity Engine** | Ball Dominant Creator, Perimeter Scorer | Elite shooter whose mere presence forces defensive attention; opens teammates even when not shooting |
| **Downhill Driver** | Ball Dominant Creator | Creates primarily by attacking the rim; high drives, high fouls drawn, less perimeter creation |
| **Pick-and-Roll Architect** | Ballhandler | Offense primarily runs through PnR initiations; playmaking happens in the act of rolling or kicking |
| **Volume Scorer** | All-Around Scorer, Ballhandler | Shot attempts/min well above archetype average; creates own shot at high frequency |
| **Midrange Specialist** | All-Around Scorer, Interior Scorer, Perimeter Scorer | Mid-range heavy shot profile; elbow pull-ups and floaters are primary creation modes |
| **Lob Threat** | PnR Rolling Big, Off-Ball Finisher | Finishes primarily through alley-oop/lob plays; rim gravity creates spacing for others |
| **Transition Runner** | PnR Rolling Big, Off-Ball Finisher, Off-Ball Movement Shooter | Gets up the floor in transition quickly; significant fast-break possession share |
| **Playmaking Big** | PnR Rolling Big, PnR Popping Big, Connector | Passes out of rolls/pops for assist opportunities; decision-maker in the short roll |
| **Elite Shooter** | Off-Ball Stationary Shooter, Off-Ball Movement Shooter | Top-tier 3PT efficiency (≥38%) on significant volume; automatic from distance |
| **Paint Presence** | Interior Scorer | Inside-the-arc shot heavy; primarily two-point focused with minimal three-point activity |

### Defensive Secondary Tags (10 tags)

| Tag | Applied to | What it means |
|---|---|---|
| **Primary Stopper** | POA Defender, Wing Stopper, Versatile Defender | Assigned to opponent's best offensive player; defensive responsibility is scheme-defining |
| **Help Specialist** | Any defensive archetype | Value comes from help rotations and off-ball positioning; not dominant on-ball |
| **Switchable** | Versatile Defender, Mobile Big | Credibly defends 2+ position groups without causing mismatches |
| **Ball Hawk** | POA Defender, Off-Ball Chaser | High steal rate; gambles for turnovers, creates transition opportunities |
| **Active Hands** | Any defensive archetype | High deflections and tips; disrupts passes and creates loose balls without gambling for steals |
| **Screen Navigator** | POA Defender, Off-Ball Chaser | Excels at fighting through or going over screens; stays attached to shooters off the ball |
| **Hustle Defender** | Any defensive archetype | Charges drawn, dives, high effort stats; defensive value through activity, not athleticism |
| **Interior Anchor** | Mobile Big, Dropping Big | Rim-area defensive authority without being primarily a shot-blocker; organizes paint defense |
| **Defensive Liability** | Any defensive archetype | Clear exploitable weakness; opponents target this player in isolation or pick coverage |
| **Shot Blocker** | Rim Protector, Mobile Big | High block rate as a secondary trait in non-primary rim-protector archetypes |

### Tags Removed from Prior (Vibe-Coded) Version

| Old Tag | Reason Removed |
|---|---|
| `Heliocentric Guard` | Renamed `Heliocentric` — position-agnostic design |
| `Post Hub` | Renamed `Post Creator` — clearer scope |
| `High Volume` | Renamed `Volume Scorer` — more descriptive |
| `Midrange Scorer` | Renamed `Midrange Specialist` — avoids confusion with primary archetype names |
| `Rim Finisher` | Renamed `Lob Threat` — more specific to the behavior |
| `Inside-the-Arc` | Renamed `Paint Presence` — clearer meaning |
| `Transition Player` | Renamed `Transition Runner` — more descriptive |
| `Primary Scorer` | Removed — redundant with BDC definition |
| `Offensive Hub` | Merged into `Playmaking Big` — unclear distinction |
| `Hockey Assist Specialist` | Removed — 0 instances, too niche to maintain |
| `Perimeter Defender` | Removed from offensive list — wrong taxonomy |
| `Rebounder` | Removed from offensive list — wrong taxonomy |
| `Primary` (defensive) | Renamed `Primary Stopper` |
| `Liability` (defensive) | Renamed `Defensive Liability` |
| `Helper` + `Help` (defensive) | Merged into `Help Specialist` |
| `Hustler` (defensive) | Renamed `Hustle Defender` |
| `Interior` (defensive) | Renamed `Interior Anchor` |

---

## 4.3 — Validation Tracks (Run in Parallel)

### Track 1: Year-over-Year Stability

**Question:** When a player appears in consecutive seasons, how often does their archetype change?

**Methodology:**
For each player in seasons N and N+1 (7 transition pairs after 8-season backfill):
- Build 11×11 transition matrix (offensive) and 9×9 (defensive)
- Diagonal = % of players who kept the same archetype

**Gate:** Diagonal ≥ 75% offensive, ≥ 85% defensive.
**Red flag:** Diagonal < 60% → archetypes are noisy features for the minute model.

**Output:** `reports/archetype_stability.json`

---

### Track 2: Threshold Sensitivity

**Question:** How fragile is classification to small threshold shifts?

**Methodology:** Re-run classifier with percentile thresholds ±2pp.
**Gate:** < 10% fragility per archetype.
**Output:** `reports/archetype_sensitivity.json`

---

### Track 3: Internal Coherence

**Question:** Do players in the same archetype actually look similar?

**Methodology:** Within-archetype std / overall std on defining features.
**Gate:** Ratio < 0.5x on each archetype's defining feature.
**Output:** `reports/archetype_coherence.json`

---

### Track 4: Manual Spot-Check

**Methodology:** Sample 30 players per archetype across 8 seasons. Review: does
classification match basketball common sense? Flag any systematic errors.
**Output:** `reports/archetype_manual_sample.csv`

---

## 4.4 — Player Tier System

A semantic label summarizing current-season talent level for each player-season row.
Attached to `player_profile_aggregate.parquet` as a new `player_tier` column.

**Design principles:**
- **BKE-anchored** (not raw RAPM or BPM — avoids double-counting upstream inputs)
- **Within-season percentile** thresholds so tiers are relative, not absolute
- **Minutes threshold** for minimum qualification (below threshold → "Fringe")
- **Current season only** — no lookback smoothing
- **Semantic, not numeric** — once the tier is set, downstream uses the label

| Tier | Criteria | Approx. count/season |
|---|---|---|
| **Superstar** | BKE ≥ 95th pctile AND ≥ 800 min | ~25 players |
| **All-Star** | BKE ≥ 82nd pctile AND ≥ 600 min | ~50 players |
| **Starter** | BKE ≥ 55th pctile AND ≥ 600 min | ~150 players |
| **Rotation Player** | BKE ≥ 30th pctile AND ≥ 300 min | ~200 players |
| **Reserve** | ≥ 200 min (classified) AND below Rotation threshold | ~200 players |
| **Fringe** | < 200 min (Insufficient Minutes class) | ~200+ players |

**Why BKE and not BPM/RAPM:** BKE is our composite portable talent score — it already
synthesizes RAPM (25%), playtype efficiency (20%), and 9 dimensions (55%). Using BKE for
tiers avoids feeding the same raw signal into multiple downstream features separately.

**Implementation:** New function in `build_profile_aggregate.py` or
`build_player_impact_profiles.py` that assigns tier after all BKE scores are computed.

---

## 4.5 — Soft Probability Adoption

The PEC columns already produce 11-dimensional archetype probability vectors
(`pec_off_prob_emb_*` columns in the aggregate). Currently downstream code uses only
`argmax` (hard primary label).

**What changes:**
- Primary archetype label stays the same (argmax doesn't change)
- Phase 2 minute model uses the full probability vector as features instead of one-hot encoding
- Cohort z-scoring can optionally become weighted (player partially belongs to two cohorts)

**What does NOT change:**
- Existing archetype assignments — no player gets reclassified
- The interaction matrix (still keyed on primary archetype)
- The secondary tags

**Implementation:** Deferred to Phase 2 — the minute model rebuild is where this pays off.

---

## 4.6 — Archetype Pair Validation (Last)

**Status (2026-05-21):** v1 ran on team-season aggregates (240 obs, Ridge). Pivot to v2.

### v1 — Team-season Ridge (rejected as final, kept as record)
- 240 obs (8 seasons × 30 teams). NET_RTG outcome. Ridge α=26.83.
- R²: 0.771 (talent-only) → 0.832 (full). Only 3/66 pairs had tight bootstrap CIs.
- Conclusion: underpowered for 66 features. Most pairs are noise after talent control.
- Archived: `src/modeling/fit_archetype_interactions_v1_team_season.py`.
- Findings JSON: `reports/archetype_interaction_fit.json`.

### v2 — Lineup-level Lasso with forward stepwise verify (COMPLETE)

**Outcome:** INTERACTION_MATRIX eliminated entirely. After fitting on 4,914 lineup-stints
across 8 seasons with talent control, **zero pairs survived** Lasso significance selection.
Decision per user (2026-05-21): if no significant pairs emerge after expanding the
dataset, eliminate the matrix completely.

**Why pivoted:** team-season has only 240 obs vs. 62,812 lineup-stints (8 seasons, after
re-running compute_advanced_metrics.py against the full historical possessions backfill).
The dense 66-pair matrix is over-parameterized for either dataset; the lineup approach
gave ~20× more usable observations.

**Data:** `data/processed/metrics_lineups.parquet` (29,024 lineup-stints × {ORTG, total_poss,
lineup_ids}, 2022-25). Filter ≥100 possessions → 2,390 lineups; lineup ORTG std drops from
37.16 (<100 poss) to 14.67 (100-200 poss).

**Outcome:** lineup ORTG, season-mean centered.
- Defense stays untouched. The defensive system in `team_feature_aggregation.py:520-604`
  is composition-based (presence/absence of rim & POA anchors, diversity bonus, liability
  penalty) — pair interactions there would mostly add noise.

**Talent control:** sum of oRAPM over 5 lineup players (sklearn pipeline with talent as
unpenalized control, Lasso applied to residuals).

**Sample weight:** `sqrt(total_poss)` to prevent a handful of 3,000-poss lineups from
dominating.

**Pipeline:**
1. LassoCV (5-fold, alpha grid 1e-4 → 10) on standardized pair features → candidate set
   of pairs with |β| > 1e-6.
2. Forward stepwise verify on candidate set: start talent-only, add by largest ΔCV R²,
   stop when next addition gains < 0.0005.
3. Bootstrap (500) for surviving pairs only — flag any pair whose 95% CI spans 0.
4. Rescale `V_AB = β / 75` (derived: pipeline applies `V × S_A × S_B × (1/0.2) × 0.75 × TEAM_SCALE`,
   where baseline pair weight = 0.2, LAMBDA = 0.75, TEAM_SCALE = 20).

**Output:** `INTERACTION_MATRIX` becomes a sparse dict (~5-15 entries). Missing pairs
implicitly default to 0.0 via existing fallback in `_get_interaction_value()`.

**Conditional flags (`both_low`, `j_low`):** Dropped in v2. The fit operates on actual
lineups so empirical β already reflects average talent. If a surviving pair has a sensible
conditional gate, we add it back manually during basketball-review.

**Output files:**
- `src/modeling/fit_archetype_interactions_v1_team_season.py` — v1 archived
- `src/modeling/fit_archetype_interactions_v2.py` — v2 fit script (lineup-level)
- `reports/archetype_interaction_fit_v2.json` — final result (0 surviving pairs)
- `src/profile_aggregate/team_feature_aggregation.py` — INTERACTION_MATRIX = {}, use_interaction default flipped to False
- `data/processed/metrics_lineups.parquet` — regenerated across 8 seasons (62,812 stints, was 29,024)

**Key empirical findings:**
- Talent R² ceiling (lineup oRAPM sum, z-scored within season): 0.20 at ≥100 poss, 0.29 at ≥200 poss
- Pair contribution above talent: 0 surviving pairs (Lasso α=10, max in grid, zeroed everything)
- Robustness checks confirmed result:
  - 8 seasons + ≥100 poss (4,914 lineups): 0 pairs
  - 8 seasons + ≥200 poss (1,895 lineups): 0 pairs
  - 3 seasons (2022-25) + ≥100 poss + oRAPM control: 2 pairs (PnR Popping × OB Stationary, Connector × Connector) — neither survived dataset expansion
- Pipeline verification: full model holdout R on 2024-25 = 0.96 with interaction=False, identical to interaction=True before (interaction was contributing ~0 anyway)

**Bug fixed during this work:**
- Pre-2022 lineup data existed in `data/historical/possessions_clean_*.parquet` but had never
  been processed into `metrics_lineups.parquet` (still showed only 3 seasons). Re-ran
  `src/data_compute/compute_advanced_metrics.py` to backfill all 8 seasons.
- oRAPM has inconsistent cross-season calibration: 2019-22 use a compressed scale
  (std~0.9) vs. 2017-18/2022-25 (std~2.1). Fixed in fit script via within-season z-scoring.
  This calibration drift in oRAPM is a follow-up flag for the RAPM pipeline (Phase 1 GAP).

---

## Files Touched

| File | Change |
|---|---|
| `src/data_compute/compute_player_archetypes.py` | Threshold lowered to 200/10/8; secondary tag vocabulary updated |
| `src/data_compute/compute_defensive_archetypes_v2.py` | Defensive secondary tag vocabulary updated |
| `src/player_eval/build_player_impact_profiles.py` | Add `player_tier` column |
| `src/modeling/validate_archetypes.py` | **New** — runs Tracks 1-4, outputs JSON reports |
| `docs/reference/basketball-intuitions.md` | Section 4: position-agnostic text + secondary tag tables |

---

## Dependencies & Sequencing

**Can start:** Phase 1 complete (clean RAPM + BPM).
**Must complete before:** Phase 2 (minute model) — uses archetype probability embeddings as features.
**Archetype pair validation (4.6) must complete before:** Phase 3B — interaction matrix OLS
needs stable archetype-pair counts as inputs.
