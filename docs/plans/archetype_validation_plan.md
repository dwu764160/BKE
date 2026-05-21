# Archetype Validation Plan

> **Status:** Revised 2026-05-21. Phase 4 scope updated based on session discussion.
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
4.2  Secondary archetype documentation             [IN PROGRESS — see §Secondary Tags below]
4.3  Validation tracks (4 parallel)               [PENDING]
4.4  Player tier system                            [PENDING]
4.5  Soft probability adoption                     [PENDING — design only, no new model]
4.6  Archetype pair validation + matrix OLS        [LAST — after everything else]
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

After 4.0-4.5 are complete and archetypes are verified stable across 8 seasons:

Use 8 seasons of historical lineup data (from `possessions_clean_*.parquet`) to measure
actual on-court offensive efficiency for each of the 121 archetype pair combinations.
Fit OLS regression: `team_ORtg ~ Σ archetype_pair_counts`.

Replaces the current untested 121-pair interaction matrix with empirically fitted values.

**Gate before running:** Track 1 diagonal ≥ 75% (archetypes must be stable inputs before
building a matrix on top of them).

**Output:** `reports/archetype_pair_interactions.json` — new interaction matrix

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
