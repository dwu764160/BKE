# BKE Basketball Intuitions Reference

> **Status:** Foundational — captures core philosophy as-built. Subject to revision
> after `docs/plans/basketball_intuition_plan.md` Phase A+B are executed.
> **Purpose:** Single reference for the basketball reasoning baked into every layer
> of the pipeline. New sessions should read this before touching modeling code.

---

## 1. The BKE Goal

**Basketball KPI Engine (BKE)** measures *portable talent* — a player's impact that travels with them
regardless of team, system, or role. It is deliberately distinct from a player's *actual production*
in a given season, which is heavily context-dependent.

The working thesis: if you can measure what a player contributes **independent of their surroundings**,
you can project those contributions to any future team context and build genuinely predictive team
net ratings without re-training for every roster change.

BKE is not a player rating intended for individual MVP debates. It is an input to team net
rating forecasts, which are the inputs to game-outcome win probabilities. Every architectural
choice flows from this goal.

---

## 2. Three-Layer Score Architecture

| Layer | Weight | What it captures | Source |
|---|---|---|---|
| **Layer 1A — RAPM Backbone** | 25% | Actual on/off impact, possession-matched | Regularized Adjusted Plus-Minus (multi-year pooled) |
| **Layer 1B — Playtype Efficiency** | 20% | PPP on specific play types (isolation, PnR, etc.) | Synergy play types |
| **Layer 1C — 9-Dimension Model** | 55% | Skill breadth and portability | Tracking + box + RAPM composites |

**Why RAPM is only 25%:** RAPM is noisy on small samples (<1000 possessions). For role players
and younger players, the 9-dimension model is more reliable. RAPM becomes more informative as
sample size grows, but even then it conflates team context. The dimension model isolates
individual contributions more directly.

**Why playtype efficiency (20%):** Raw efficiency (PPP on isolation, PnR, cut, etc.) measures
whether a player converts their opportunities. It complements RAPM (outcome) and dimensions
(skill levels) with "what do they do when called upon?"

---

## 3. The 9 Dimensions of Portable Talent

Organized by how "portable" (context-independent) they are. Portable dimensions use
**league-wide z-scores**. Position-conditional dimensions use **within-position-group z-scores**
to prevent big-man inflation.

### Universally Portable (42% of Layer 1C weight)

| # | Dimension | Weight | Core signal |
|---|---|---|---|
| 1 | **Shooting Gravity** | 14% | 3PT volume + efficiency + TS%. Creates spacing for teammates regardless of system. |
| 3 | **Playmaking Creation** | 14% | Assists, potential assists, passes made, drive-assist rate. Collapsing defenses benefits any lineup. |
| 9 | **Self-Creation** | 14% | Pull-up FGA, off-dribble creation, ball dominance. The ability to make something out of nothing. |

### Valuable But Positional (32% of Layer 1C weight)

| # | Dimension | Weight | Core signal |
|---|---|---|---|
| 7 | **Turnover Control** | 10% | TOV%, turnovers per touch, drive turnover rate. Bad TO control follows you to any team. |
| 6 | **Defensive Impact** | 10% | DRAPM, DRTG, defensive results percentile. On/off defensive value. |
| 8 | **Defensive Versatility** | 12% | Switch score, matchup diversity, assignment difficulty. Ability to defend multiple positions is universally valued. |

### Position-Dependent (26% of Layer 1C weight)

| # | Dimension | Weight | Core signal |
|---|---|---|---|
| 2 | **Driving Gravity** | 10% | Drives per 36, at-rim frequency, fouls drawn. Rim pressure creates collateral value but is inherently positional. |
| 4 | **Extra Possession Creation** | 8% | OREB%, DREB%, rebounding rate. 40% counted toward offense, 60% defense. Strongly position-correlated, so lower weight. |
| 5 | **Defensive Playmaking** | 8% | STL%, BLK%, deflections, hustle stats, charges drawn. Chaos creation is position-skewed (centers dominate blocks). |

**Why position-conditional z-scores for Dims 2/4/5/6/8:**
A center naturally gets more blocks and rebounds than a guard. A league-wide z-score would make
every center elite at Dim 5 and every guard bad at it — that's measuring position, not talent
above position. Within-group z-scores correct for this.

---

## 4. Player Archetypes

Archetypes serve two functions: (1) cohort-conditional z-scoring (compare players against
peers doing similar things) and (2) team structure modeling (what mix of roles produces
functional lineups).

### Position-Agnostic Design

**Archetypes describe basketball role, not listed position.** A player classified as
"PnR Rolling Big" is someone who rolls to the rim in pick-and-roll actions — regardless
of whether the NBA lists them as a Guard, Forward, or Center. Gary Payton II (listed SG)
as a PnR Rolling Big is correct classification, not an error.

This is intentional and a key advantage over position-conditional systems. Position-priors
would prevent the model from capturing positional versatility, role evolution, and
non-traditional deployments. The position bands (Section 5) exist for z-score cohort
normalization only — they do not constrain archetype assignment.

> **Secondary archetype tags** (documented in the subsection below) further clarify *what
> kind* of a given primary archetype a player is. They have no computational effect —
> they are human-readable explainers only.

### Offensive Archetypes (11 types)

| Archetype | Prototypical role |
|---|---|
| **Ball Dominant Creator** | Primary ball-handler, creates heavily off-dribble |
| **Ballhandler** | Secondary creator, initiates offense without as much scoring responsibility |
| **All-Around Scorer** | Scores from multiple areas, both off the ball and on |
| **Interior Scorer** | Post-up, paint, rim-finishing focus |
| **Perimeter Scorer** | Mid-range + 3PT dominant offense |
| **PnR Rolling Big** | Rolls to rim in PnR, finishes at the basket |
| **PnR Popping Big** | Pops to perimeter in PnR, spacing-first big |
| **Off-Ball Finisher** | Cuts, catches-and-finishes, no-dribble player |
| **Off-Ball Movement Shooter** | Shooters who run off screens, move off-ball to find shots |
| **Off-Ball Stationary Shooter** | Corner/catch-and-shoot specialists, minimal movement |
| **Connector** | High-volume passers without primary creation responsibility |

### Defensive Archetypes (9 types)

| Archetype | Prototypical role |
|---|---|
| **POA Defender** | Point-of-attack; guards primary ball-handlers, creates turnovers |
| **Wing Stopper** | Locks down wing scorers, physical and attentive |
| **Versatile Defender** | Can guard multiple positions credibly; the switchable archetype |
| **Rim Protector** | Shot-alterer, paint presence, blocks-first big |
| **Dropping Big** | Drops in pick coverage, not rim-protecting aggressively |
| **Mobile Big** | Versatile big who can switch or pressure ball |
| **Off-Ball Chaser** | Strong off-ball defender, stays attached to shooters |
| **Rotational Defender** | Solid systemic defender without standout individual trait |
| **Low-Activity Defender** | Below-average activity; often a defensive liability |

**Why archetypes matter for z-scoring:**
A spot-up shooter at average 3PT% is mediocre for a Wing Stopper but elite for an Off-Ball
Stationary Shooter. Comparing both against league average would mismeasure both. Archetype
cohorts make the comparison fair.

---

## 5. Position Bands

Three-tier grouping used for some z-score cohorts and team structure analysis.

| Band | NBA positions | Role |
|---|---|---|
| **Smalls** | PG, SG | Ball handlers, perimeter scorers, POA defenders |
| **Wings** | SF, SG-SF, SF-PF | Versatile two-way players, wing scorers |
| **Bigs** | PF, C | Rim presence, rebounding, spacing (PnR bigs) |

Five-tier grouping for granular position matching:
`Guard → Guard-Forward → Forward → Forward-Center → Center`

**Planned improvement (Phase 6 / Basketball Intuition Plan):** Make `position_band_3` and
`position_band_5` canonical computed columns in `compute_position_estimate.py`. Currently,
some scripts apply ad-hoc position groupings using raw NBA position labels.

---

## 6. Team Net Rating Construction

The path from player BKE to a projected team net rating:

```
Step 1: Minute-weighted team BKE average
        team_talent_base = Σ (player_bke × minute_share)

Step 2: TEAM_SCALE conversion
        talent_contribution = DEFAULT_TEAM_SCALE × team_talent_base
        (DEFAULT_TEAM_SCALE = 20.0; calibration implies ~25.0 — see findings doc)

Step 3: Structure modifiers (capped at 30% of talent variance)
        offensive_structure  = playmaking depth bonus/penalty
                             + spacing bonus/penalty
        defensive_structure  = rim protection penalty/bonus
                             + POA defender penalty/bonus
                             + defensive liability penalty

Step 4: Interaction matrix (121 archetype-pair values)
        team_interaction = Σ pair_bonus(arch_i, arch_j) for all roster pairs
        scaled by INTERACTION_LAMBDA = 0.75

Step 5: Volatility estimate
        team_volatility = f(3PA concentration, creation concentration, transition style)

Step 6: Net rating = talent_contribution + modifiers + interaction + intercept
```

**Why modifiers are kept small (< 30% of variance):**
If lineup context (spacing, playmaking depth) could overcome talent differences,
teams would optimize roster construction around system fit instead of talent acquisition.
The data consistently shows talent is the primary driver of net rating; structure is a
second-order adjustment.

**Current calibration issue:** `TEAM_SCALE = 20.0` is conservative. The pipeline comment
notes the mathematically implied value is ~25 (BKE std ≈ 0.22; desired net-rating std ≈ 5.5;
25 ≈ 5.5/0.22). Using 20 as a conservative default. Data-driven calibration over 2 transitions
implies 24.7–25.3. Will be updated after 5+ transitions available.

---

## 7. Game Model

A single-game win probability is modeled as a Gaussian margin model:

```
mu_margin = (team_net_rating_home + HCA + rest_adj_home) - (team_net_rating_away + rest_adj_away)
sigma      = SIGMA_LEAGUE (≈ 3.0 in pts/100-poss scale)
P(home win) = Φ(mu_margin / sigma)
```

**Game model constants:**

| Parameter | Current value | Notes |
|---|---|---|
| `SIGMA_LEAGUE` | 3.0 pts/100 poss | Game-to-game variance around expected margin |
| `HOME_COURT_ADVANTAGE` | 2.0 pts | Legacy flat default; team-specific HCA available from fitted coefficients |
| B2B penalty | 0.0 (not yet fitted) | Planned in Phase 6 — fit from game logs |
| Rest day bonus | 0.0 (not yet fitted) | Planned in Phase 6 |

**Why Gaussian:** NBA game margins are approximately normally distributed. The Gaussian
model is interpretable, fast to fit, and well-calibrated for our purposes. More complex
models (heteroskedastic, pace-adjusted) are on the roadmap but not the bottleneck.

**Current baseline (Phase 0 walk-forward):** Brier=0.2320 out-of-sample vs. naive
home-prior of ~0.247. Target: approach Vegas-level ~0.21.

---

## 8. Age Curve

Year-over-year BKE change expected by age bracket, applied in forecast projection:

| Bracket | Hardcoded delta | Empirical (2 transitions) | Flag |
|---|---|---|---|
| <21 | +0.08/yr | insufficient data | — |
| 21–24 | +0.04/yr | +0.098/yr | OK (close) |
| 24–27 | +0.01/yr | +0.034/yr | OK |
| **27–30** | **+0.01/yr** | **-0.030/yr** | **⚠ Wrong sign** |
| **30–33** | **-0.01/yr** | **-0.039/yr** | **⚠ Too optimistic** |
| **33–36** | **-0.02/yr** | **-0.074/yr** | **⚠ Too optimistic** |
| 36+ | -0.03/yr | -0.032/yr | OK |

**Basketball rationale for current hardcoded values:** Calibrated during early development
on the assumption that the 27-30 bracket still shows some "skill prime" improvement (players
learning to play smarter as athleticism levels off). Empirical data suggests this is no longer
true for the league-average player — earlier physical decline or more demanding physical play
has shifted the peak to 24-26.

**Apply empirical corrections when:** 5+ walk-forward transitions available (survivorship bias
inflates the magnitude for ages 30+; direction of 27-30 correction appears reliable).

---

## 9. Rookie / Low-Sample Prior

Players with <500 possessions fall to RAPM league-mean prior — unreliable for any individual
projection. Hardcoded rookie expectations by draft tier:

| Draft tier | Expected BKE | Expected MPG |
|---|---|---|
| Lottery (picks 1-14) | -0.05 | 20 min |
| Mid-first (picks 15-20) | -0.10 | 13 min |
| Late-first (picks 21-30) | -0.15 | 9 min |
| Second round | -0.18 | 6 min |
| Undrafted | -0.22 | 4 min |

**Planned improvement (Phase 3):** Build a DARKO-substitute prior model using draft position
+ age + box BPM + xRAPM to estimate ORAPM/DRAPM for low-sample players. DARKO CSV ingest
is already scaffolded (`src/data_fetch/fetch_darko_manual.py`).

---

## 10. What BKE Is Not Designed To Measure

- **In-game adjustment:** BKE is a season-long average. It cannot reflect that a player
  performs better in clutch situations, specific matchups, or after halftime adjustments.
- **Injury-discounted value:** Availability is handled separately via the availability
  factor (Step 4 in the player eval pipeline). BKE measures per-minute quality, not
  expected seasonal contribution.
- **Chemistry / scheme fit:** The interaction matrix captures archetype-pair synergies
  empirically, but true chemistry (specific player familiarity, trust, communication) is
  unmeasured.
- **Market information:** Confirmed lineups, injury reports, rest status on game day are
  not in the model. This is the primary gap between our Brier (0.2320) and Vegas (~0.21).
