Excellent. This changes the shape of Phase 2 significantly.

You’re not building from scratch — you already have:

* A modular RAPM-based possession pipeline
* Season-normalized BKE with shrinkage + compression reports
* Centralized archetype gating logic
* Stability/backtest reporting infrastructure
* Governance + reproducibility discipline

So Phase 2 is **not** “add features.”
It is:

> Lift the existing pipeline from a player metric generator → into a structural team evaluation engine.

Now I will rewrite Phase 2 v1 accordingly.

---

# 🏗 Phase 2 — Player Evaluation Core (PEC) v1

### (Built on Existing Pipeline Infrastructure)

---

# 🎯 Core Objective of Phase 2

Transform BKE + Archetypes into a **team-constructible evaluation backbone** that can:

1. Translate player impact → lineup impact
2. Translate lineup impact → team net rating
3. Translate team net rating → wins
4. Support minute allocation learned from real data
5. Preserve role vs impact separation

WITHOUT:

* Breaking existing pipeline modularity
* Polluting archetype gating with impact logic
* Turning the system into a black box

---

# 🧠 Architectural Philosophy (Aligned With Repo Principles)

We will preserve:

### Role vs Impact Separation

* Archetypes remain behavior-only.
* Impact overlays on top.

### Centralized Threshold Governance

* Any new gating or scaling constants go in a centralized file.
* No scattered heuristics.

### Modular Pipeline Stages

New logic must live in new modules under:

```
modeling/team_modeling/
```

NOT inside data_compute or features.

---

# 🪜 Phase 2 v1 — Layered Roadmap

---

# 🔹 STEP 0 — Data Audit & Stability Lock

Before adding logic:

### Required Actions

* Load latest:

  * modeling_inputs_report.json
  * rapm_validation_report.json
  * latest bke_v*_report.json
  * *_backtest.json
* Confirm:

  * Stability baseline
  * Predictive baseline
  * Shrinkage magnitudes
  * Archetype prevalence

This becomes:

> PEC Baseline Snapshot

We freeze BKE behavior for PEC v1.

No metric changes during PEC construction.

---

# 🔹 STEP 1 — Player Impact Object Refactor

Currently BKE is an output value.

We create:

```
class PlayerImpactProfile:
    orapm
    drapm
    bke
    stability_score
    minutes
    possession_rate
    archetype_probs
```

Source:

* data_compute outputs
* processed archetype outputs

This is a structural wrapper.
No math change yet.

---

# 🔹 STEP 2 — Wins Translation Layer

We introduce:

```
modeling/team_modeling/wins_translation.py
```

### Goal

Translate RAPM-like impact → team-level net rating → wins.

Method v1:

1. Convert player ORAPM/DRAPM into per-100 possession contribution.
2. Weight by minutes share.
3. Aggregate to team net rating.
4. Use historical mapping:
   Net Rating → Wins curve (fit from real seasons).

This curve must be learned from data.

No heuristics.
Fit regression from historical season-level:

* Team Net Rating
* Team Wins

Store curve coefficients in:

```
reports/team_win_curve.json
```

---

# 🔹 STEP 3 — Minute Allocation Model (Real Data Driven)

You were right — this is foundational.

Without minutes, team modeling collapses.

We build:

```
modeling/team_modeling/minute_model.py
```

### v1 Objective

Learn distribution of minutes by:

* Archetype
* Impact percentile
* Age band
* Position proxy

From historical data:

* Player season minutes
* Archetype labels
* Impact values

Model type:
Simple regression or gradient boosting.

Output:
Predicted minute share distribution within roster.

Important:
This is NOT rotation simulation yet.
It predicts season aggregate minute shares.

---

### Guardrails

* Must not leak future performance.
* Must not use BKE itself as sole predictor (avoid circularity).
* Minutes predicted before win translation.

---

# 🔹 STEP 4 — Team Aggregation Framework (TAF v1)

Now we create:

```
modeling/team_modeling/team_aggregator.py
```

### Inputs

* List of PlayerImpactProfiles
* Predicted minutes
* Archetype probabilities

### Process

1. Weighted ORAPM sum
2. Weighted DRAPM sum
3. Adjust for diminishing usage overlap

We implement:

### Usage Overlap Penalty (v1 simple version)

If multiple players:

* High usage archetype probability > threshold
  Apply mild diminishing coefficient.

Centralized in:

```
modeling/team_modeling/constants.py
```

---

### Defensive Redundancy Adjustment (v1 light)

If:

* Multiple low-impact defenders
* No anchor-type defender

Apply team defense penalty.

Not aggressive.
Just structural realism.

---

# 🔹 STEP 5 — Team Net Rating → Wins

Using:
team_win_curve.json

Produce:

```
Projected Wins
Confidence Band (derived from player stability aggregate)
```

---

# 🔹 STEP 6 — Validation Layer

We validate against historical seasons.

Procedure:

1. Use actual roster composition.
2. Predict minutes using minute model.
3. Compute projected wins.
4. Compare vs actual wins.

Metrics:

* RMSE
* MAE
* Rank correlation
* Playoff team classification accuracy

Store in:

```
reports/team_model_validation_v1.json
```

---

# 🔹 STEP 7 — Stability Monitoring

We check:

If minute model amplifies volatility.

Compute:

* YoY team projection correlation
* Sensitivity to player rank shifts

If volatility spikes:
Add mild impact smoothing layer.

---

# 📦 Deliverables at End of PEC v1

We will have:

1. PlayerImpactProfile abstraction
2. Learned minute allocation model
3. Team aggregation logic
4. Net rating → wins model
5. Historical validation report
6. Reproducible pipeline integration

---

# 🧩 How This Feeds Future Phases

---

## 🌐 Web Prediction App

Now possible:

* Input real roster
* Predict season wins
* Adjust minutes manually
* Simulate trade impact

Game prediction v1:
Win probability derived from:
Net rating difference + home-court adjustment.

---

## 🎮 Simulation Game

Engine already has:

* Player vector
* Archetype profile
* Salary translation possible next
* Team win engine

Add:

* Cap model
* Contract aging
* Development curves
* Injury randomness

---

# 🚫 What We Are Explicitly NOT Doing Yet

* Pairwise synergy matrices
* Possession-level lineup simulation
* ML stat forecasting
* Aging curves
* Injury modeling
* Playoff matchup modeling

Those are Phase 3.

---

# 🧠 Key Architectural Insight

You already have:

Possession-level micro modeling.

Phase 2 builds:

Season-level macro modeling.

Later we bridge the two.

---

# 🧭 Updated Phase 2 Structure

| Stage | Focus               | Complexity |
| ----- | ------------------- | ---------- |
| 2.0   | PlayerImpactProfile | Low        |
| 2.1   | Wins Translation    | Medium     |
| 2.2   | Minute Model        | Medium     |
| 2.3   | Team Aggregator     | Medium     |
| 2.4   | Validation          | Medium     |

No ML-heavy deep system yet.
Just controlled structural learning.

---

# 🔥 Why This Order Is Correct

If you build:

ML stat prediction first → you still can’t simulate teams.

If you build:

Team aggregation first → you immediately unlock simulation capability.

---

# 🎯 Final Strategic Alignment

This keeps:

BKE = Public Stat
PEC = Internal Engine
Simulation + Web App = Products

And everything remains modular and reproducible.

---

Now I’ll ask you the most important Phase 2 design constraint:

Do we want PEC v1 to be:

A) Fully deterministic given inputs
or
B) Include stochastic elements (variance bands, simulation draws) from day one?

That changes how we architect minute modeling and win projection.


------------------update after answer------------------

Excellent — this is the right level of skepticism.
We’re now stress-testing architecture, not code.

You chose:

> ✅ Deterministic v1 (stochastic later)

Good. That keeps the backbone clean.

Now let’s address each of your structural concerns carefully.

---

# 1️⃣ Are We Overusing RAPM?

Short answer:
**Not yet — but we could if we’re careless.**

## Current Reality

Your BKE already:

* Uses RAPM backbone (ORAPM / DRAPM)
* Applies shrinkage + compression
* Normalizes across seasons
* Produces stable impact

If PEC uses RAPM again as the *only* driver of team aggregation, we risk:

> Double-counting the same signal without leveraging your rich feature set.

---

## What We Actually Have (and Should Leverage)

From your repo:

* Full box score
* Per 36 + per 100
* WS, BPM, VORP, GMSC
* Hustle stats
* Play-type frequencies
* Tracking metrics
* Archetype probabilities
* RAPM outputs
* Stability diagnostics

That is much richer than a typical RAPM-only framework.

---

## Architectural Correction

For PEC v1:

**Team Net Rating should not be:**
Sum(weighted ORAPM + DRAPM)

It should be:

Team Net Rating =
Weighted Impact Signal

* Structural Feature Correction
* Archetype Interaction Adjustments

RAPM is the backbone — not the entire skeleton.

---

# 2️⃣ Is Step 1 (PlayerImpactProfile) Comprehensive Enough?

Original class proposal:

```python
class PlayerImpactProfile:
    orapm
    drapm
    bke
    stability_score
    minutes
    possession_rate
    archetype_probs
```

That is NOT comprehensive enough.

---

## What It’s Missing

Given your system, it should also include:

* Usage rate
* Assist rate
* TOV%
* 3PA rate
* Rim frequency
* Playtype distribution vector
* Hustle event rate
* Rebound rates
* Foul rates
* On/off splits (if stored)
* Variance/compression flags
* Age
* Position proxy

Why?

Because:

### Future:

* Minute model needs more than impact.
* Usage overlap penalty needs actual usage data.
* Four factor modeling needs player-level inputs.
* Archetype probability validation needs behavior vectors.

---

## Revised Step 1 Object

It becomes:

```python
class PlayerImpactProfile:
    impact:
        orapm
        drapm
        bke
        stability_score

    behavioral:
        usage
        playtype_vector
        shooting_profile
        assist_rate
        turnover_rate
        rebound_rates
        hustle_rates

    archetype:
        probability_vector

    meta:
        age
        minutes_last_season
        possession_volume
```

Now it is forward-compatible.

---

# 3️⃣ Should Step 3 (Minute Model) Use ML?

Yes.

This is the correct place to introduce ML — not for impact, but for rotation realism.

---

## Why ML Is Appropriate Here

Minute allocation depends on:

* Impact percentile
* Archetype rarity
* Usage level
* Age
* Position redundancy
* Past minute trends

This is nonlinear and interaction-heavy.

A simple regression will underfit.

---

## Recommended Approach

Gradient boosting (e.g., XGBoost or LightGBM).

Target:
Player minute share of team total minutes.

Features:

* Impact
* Archetype probabilities
* Behavioral features
* Age
* Previous season minutes
* Injury proxy (games played)

Output:
Deterministic expected minute share.

This does NOT introduce black-box danger because:

* It does not affect impact.
* It affects allocation realism only.

---

# 4️⃣ Is Step 5 (Net Rating → Wins) Too Simple?

Yes — if we stop at net rating alone.

But net rating is foundational truth.

Historical fact:
Team Net Rating correlates extremely strongly with wins (~0.95+).

However…

Net rating hides structural composition.

---

## What We Can Add (Still Deterministic)

Instead of:

TeamNet = Off - Def

We compute:

### Team Four Factors

Offense:

* eFG%
* TOV%
* ORB%
* FTr

Defense:

* Opp eFG%
* Opp TOV%
* Opp ORB%
* Opp FTr

You already have player-level building blocks for this.

---

## Better Approach

Instead of directly aggregating RAPM:

We:

1. Aggregate player-level components into team-level Four Factors.
2. Map Four Factors → Net Rating via regression.
3. Map Net Rating → Wins.

This gives structure.

---

## Why This Is Stronger

Now:

* 3P variance matters.
* Turnover clustering matters.
* Rebound archetypes matter.
* Pace-adjusted realism emerges.

And still deterministic.

---

# 5️⃣ Are We Skipping Lineup Aggregation Too Aggressively?

Important question.

Reality:

Full lineup modeling is exponentially complex.

But here’s the truth:

Over 82 games, season outcomes are much more driven by:

* Aggregate team quality
* Top 6 player strength
* Injury stability
* Net rating fundamentals

Lineup micro-synergy matters more in playoffs.

---

## Correct Compromise for v1

We do NOT model 5-man lineup permutations.

Instead:

We include:

* Usage overlap penalties
* Archetype redundancy penalties
* Anchor presence bonus

This approximates lineup realism without combinatorial explosion.

Full lineup modeling = Phase 3.

---

# 6️⃣ Are Layers Modular?

Yes — IF we structure them correctly.

We must:

Keep these separate modules:

```
modeling/player_profiles/
modeling/minute_model/
modeling/team_features/
modeling/team_projection/
```

No leakage into:

* data_compute
* archetype logic
* BKE pipeline

This preserves reproducibility.

---

# 7️⃣ Are We Making Too Big of a Leap From Player → Team?

If we use:

RAPM sum → Wins

Yes, that’s too big.

If we use:

Player features → Team Four Factors → Net Rating → Wins

No — that’s structurally sound.

That becomes:

Player micro
→ Team structural profile
→ Outcome macro

That’s realistic modeling.

---

# 🎯 Revised Phase 2 v1 (Stronger Version)

---

## STEP 1 — Expanded PlayerImpactProfile

Include:
Impact + behavioral + archetype + meta.

---

## STEP 2 — ML Minute Allocation Model

Gradient boosting trained on historical minute share.

---

## STEP 3 — Team Feature Aggregation

Aggregate:

* eFG%
* TOV%
* ORB%
* FTr
  Offense + Defense.

Include archetype redundancy adjustments.

---

## STEP 4 — Team Net Rating Model

Train regression:
Four Factors → Net Rating.

---

## STEP 5 — Wins Model

Historical mapping:
Net Rating → Wins.

---

## STEP 6 — Validation

Compare:

* Projected wins
* Net rating error
* Playoff classification accuracy
* Top seed identification accuracy

---

# 🔥 Final Strategic Insight

We are NOT building:

“Sum of RAPM simulator.”

We are building:

A structural translation system:
Player traits → Team profile → Wins.

That is fundamentally different — and much stronger.

---

Now I’ll ask you something critical:

For Phase 2 v1 validation —

Do we optimize:
A) Wins prediction accuracy
or
B) Net rating accuracy

Those are not identical objectives.

-------------------------steps 1 and 2 detailed: ------------------

🔹 STEP 1 — PlayerImpactProfile Construction
Objective

Construct a deterministic, season-aligned object that aggregates:

Impact metrics

Behavioral metrics

Archetype probabilities

Stability signals

Meta information

WITHOUT modifying BKE pipeline logic.

Data Sources

Load from:

processed/ (player season tables)

features/ (behavioral metrics)

latest bke_v*_report.json

archetype outputs

modeling_inputs_report.json

1.1 — Mathematical Definition

For each player-season i:

Define:

Impact Vector:

OR_i

DR_i

BKE_i

Stability_i

Behavioral Vector:

USG_i

AST%_i

TOV%_i

3PA_rate_i

Rim_rate_i

eFG_i

ORB%_i

DRB%_i

FTr_i

Hustle_rate_i

Playtype frequency vector P_i = (p1, p2, ..., pk)

Archetype Vector:

A_i = (a1, a2, ..., am)
Where sum(A_i) = 1

Meta:

Age_i

Minutes_i

Possessions_i

Games_i

1.2 — Normalization Rules

All rate-based stats must be:

Per 100 possessions
OR

True rate (%)

No raw counts.

Minutes normalized to:
Minute Share Potential = Minutes_i / (TeamMinutes)

Store as float.

1.3 — Stability Integration

Stability_i is derived from:

YoY correlation window

Shrinkage magnitude

Compression ratio

Define:

Stability_i =
Weighted function of:

Defensive shrinkage factor

Variance compression delta

Multi-season correlation

This remains scalar.

1.4 — Object Structure
class PlayerImpactProfile:
    # Impact
    orapm: float
    drapm: float
    bke: float
    stability: float

    # Behavioral
    usage: float
    assist_rate: float
    turnover_rate: float
    three_point_rate: float
    rim_rate: float
    efg: float
    orb_rate: float
    drb_rate: float
    free_throw_rate: float
    hustle_rate: float
    playtype_vector: np.array

    # Archetype
    archetype_probs: np.array

    # Meta
    age: float
    minutes: float
    possessions: float
    games: float

This object is saved to:

modeling/player_profiles/player_profiles_season.pkl

1.5 — Validation Checks

For all players:

No missing archetype probability mass

All rates within logical bounds

ORAPM + DRAPM consistent with BKE policy

No leakage across seasons

🔹 STEP 2 — ML Minute Allocation Model

This is deterministic ML (fixed model weights once trained).

2.1 — Problem Definition

For each team-season:

Let total minutes = 48 * 5 * 82 = 19680.

For each player i on team T:

Predict:

MinuteShare_i = Minutes_i / 19680

We train on historical seasons.

2.2 — Feature Matrix Construction

For each player-season (training data):

Features X_i:

Impact:

OR_i

DR_i

BKE_i

Stability_i

Behavior:

Usage

Assist rate

TOV%

3PA rate

Rim rate

Rebound rates

FTr

Hustle rate

Playtype vector components

Archetype:

Archetype probabilities

Meta:

Age

Previous season minutes

Games played

Team context features:

Number of high-usage players on team

Archetype redundancy counts

Target:

MinuteShare_i

2.3 — Model Type

Gradient Boosted Regression (e.g., LightGBM)

Reason:

Handles nonlinear interactions

Handles feature interactions

Robust to scale

Deterministic given seed

2.4 — Training Protocol

Train on seasons N-5 to N-1

Validate on season N

Use grouped K-fold by season

Ensure no leakage from future minutes

Evaluation metrics:

MAE of minute share

R²

Rank correlation within team

Store model in:

modeling/minute_model/minute_model_v1.pkl

2.5 — Inference Rules

When predicting future team:

Input roster PlayerImpactProfiles.

Generate predicted minute shares.

Enforce:

Constraint:
Sum(MinuteShare_i) = 1 per team.

Normalize predictions:

MinuteShare_i :=
RawPrediction_i / Sum(RawPrediction_j)

2.6 — Post-Processing Constraints

Cap individual minute share at realistic threshold (e.g., 0.22)

Floor at 0

Re-normalize

Thresholds centralized in:

modeling/team_modeling/constants.py

Why Steps 1 and 2 Must Be Perfect

Because:

Team Net Rating Projection =
Σ (MinuteShare_i × PlayerContribution_i)

If minute allocation is wrong:
Everything collapses.

Structural Soundness Check

Are we:

Avoiding double RAPM overuse? → Yes (behavioral features included)

Keeping modular separation? → Yes

Preventing black box leakage? → Yes

Optimizing for net rating? → Yes

Building foundation for game prediction? → Yes

Step 1 and 2 Note 3/3/2026: Player agg. might be redundant, minute model benefits from removing salary.

--------------Steps 3 and 4 detailed:------------------
Perfect.
We now lock Step 3 as a **fully specified, modular, implementation-ready layer**.

This will be written so an AI engineer can build it directly without guessing intent.

---

# STEP 3 — OFFENSIVE & DEFENSIVE TEAM STRUCTURE LAYER

(Mean Model + Volatility Model)

Everything here assumes:

* Projected minutes sum to **240**
* All minute shares are normalized to 1.0
* Archetypes are already assigned
* Player OffBKE and DefBKE already computed

---

# SECTION 1 — PREPROCESSING

## 1.1 Normalize Minutes

For each player:

[
m_i = \frac{Minutes_i}{240}
]

Check:

[
\sum_i m_i = 1
]

---

## 1.2 Percentile Flags

Precompute for each archetype:

* P25
* P35
* P50 (median)

Using **league-wide distribution within archetype**.

Store flags per player:

```
is_below_p25_BDC
is_below_p35_IS
is_below_p50_PS
...
```

These will trigger conditional matrix logic.

---

# SECTION 2 — OFFENSIVE MEAN MODEL

[
TeamOff_{mean} = TalentBase + InteractionTerm + StructureTerm
]

---

# 2.1 TALENT BASE

[
TalentBase = \sum_i (m_i \cdot OffBKE_i)
]

This is your stable anchor.

---

# 2.2 INTERACTION TERM (FULL 240 ROTATION)

Loop over **all player pairs i < j**.

### 2.2.1 Pair Weight

[
pair_weight_{i,j} = m_i \cdot m_j
]

This automatically ensures:

* Bench players contribute less
* 30 MPG players matter more
* No manual starter weighting required

---

### 2.2.2 Matrix Value Lookup

Convert your symbolic matrix into numeric:

| Symbol | Value |
| ------ | ----- |
| ++     | +0.10 |
| +      | +0.06 |
| Mild + | +0.03 |
| -      | -0.06 |
| Mild - | -0.03 |
| --     | -0.10 |
| n/a    | 0     |

---

### 2.2.3 Conditional Logic Rules

Apply conditional guards before assigning value.

Examples:

#### BDC × BDC

If BOTH players:

```
is_below_p25_BDC == True
```

Then value = -0.06
Else value = 0

---

#### BDC × IS

If Interior Scorer player:

```
is_below_p25_IS == True
```

Then value = -0.06
Else value = 0

---

#### BF × BF

Always:

```
value = -0.10
```

---

#### PS × OBM

If PS player:

```
is_below_p50_PS == True
```

Then value = -0.06
Else value = 0

---

Implement as:

```
value = lookup_matrix(archetype_i, archetype_j)

if value has conditional:
    evaluate percentile flags
    override if necessary
```

---

### 2.2.4 Accumulate Raw Interaction

[
RawInteraction = \sum_{i<j} pair_weight_{i,j} \cdot value_{i,j}
]

---

### 2.2.5 Scaling

Multiply by global interaction scalar:

[
InteractionTerm = \lambda_{interaction} \cdot RawInteraction
]

Initial:

[
\lambda_{interaction} = 0.75
]

We tune later.

---

### 2.2.6 Cap Interaction

To preserve stability:

[
InteractionTerm = clip(InteractionTerm, -2.0, +2.0)
]

This prevents runaway synergy effects.

---

# 2.3 STRUCTURE TERM (OFFENSIVE FEATURES)

These are team-level derived metrics.

---

## 2.3.1 Turnover Control

Compute:

[
TeamTOV = \sum_i (m_i \cdot TOV%_i)
]

Penalty:

[
TurnoverPenalty = -\beta_{tov} \cdot (TeamTOV - LeagueAvgTOV)
]

Initial:

[
\beta_{tov} = 0.12
]

---

## 2.3.2 Free Throw Rate

[
TeamFTr = \sum_i (m_i \cdot FTr_i)
]

[
FTrBonus = \beta_{ftr} \cdot (TeamFTr - LeagueAvgFTr)
]

Initial:

[
\beta_{ftr} = 0.15
]

---

## 2.3.3 Playmaking Diversity

Count players:

```
AssistRate > threshold AND mpg > 15
```

If count == 1:
-0.7

If count >= 3:
+0.3

Else:
0

---

## 2.3.4 Spacing Credibility

Count shooters:

Criteria:

* 3PA rate > threshold
* 3P% > 35%
* mpg > 15

If < 2:
-1.0

If >= 4:
+0.5

Else:
0

---

## 2.3.5 Structure Term Sum

[
StructureTerm = TurnoverPenalty + FTrBonus + PlaymakingAdj + SpacingAdj
]

Cap:

[
clip(StructureTerm, -2.0, +2.0)
]

---

# SECTION 3 — DEFENSIVE MEAN MODEL

[
TeamDef_{mean} = DefTalent + Essentials + Diversity - Liability
]

---

## 3.1 Defensive Talent Base

[
DefTalent = \sum_i (m_i \cdot DefBKE_i)
]

---

## 3.2 Essentials

### Rim Protector Check

If no player:

* Archetype = Rim Protector
* mpg > 15

Then:
-1.2

---

### POA Defender Check

If none > 15 mpg:

-0.8

If BOTH missing:

Additional -0.5

---

## 3.3 Anchor Quality Scaling

Top Rim Protector:

[

* 0.6 \cdot DefBKE_{RP}
  ]

Top POA:

[

* 0.4 \cdot DefBKE_{POA}
  ]

---

## 3.4 Diversity Bonus

Count unique defensive archetypes > 15 mpg.

If count > 3:

[
+0.15 \cdot (count - 3)
]

Cap at +0.6.

---

## 3.5 Liability Penalty

For each player:

If:

```
DefBKE < -1.0 AND mpg > 20
```

Penalty:

-0.4 each

If 2+ such players:

Additional -0.4 stacking penalty

---

Cap total defensive adjustments:

±3.0

---

# SECTION 4 — FINAL TEAM NET RATING (MEAN)

[
TeamNet_{mean} =
TeamOff_{mean}

* TeamDef_{mean}
  ]

---

# SECTION 5 — VOLATILITY MODEL

Separate from mean.

Used only in simulation.

---

## 5.1 Base Variance

[
\sigma_{base} = LeagueStdDev
]

---

## 5.2 3PT Frequency Volatility

[
\sigma_{3PA} =
\alpha_1 \cdot (Team3PARate - LeagueAvg)
]

High 3PA → higher σ

---

## 5.3 Creation Concentration

Compute:

[
C = max(UsageShare_i)
]

If C > threshold:

[
\sigma_{creation} = \alpha_2 \cdot (C - threshold)
]

---

## 5.4 Transition Frequency

High transition teams:

[
\sigma_{transition} =
\alpha_3 \cdot (TeamTransitionFreq - Avg)
]

---

## 5.5 Final Volatility

[
\sigma_{team} =
\sigma_{base}

* \sigma_{3PA}
* \sigma_{creation}
* \sigma_{transition}
  ]

Floor:

[
\sigma_{team} \ge 8
]

Ceiling:

[
\sigma_{team} \le 16
]

---

# SECTION 6 — MODULARITY DESIGN

Each component must be toggleable:

```
use_interaction = True
use_structure = True
use_defense_architecture = True
use_volatility = True
```

So you can ablate and test:

* Talent only
* Talent + Interaction
* Talent + Structure
* Full model

This is critical for stability testing.

---

# WHAT THIS ACHIEVES

You now have:

* Full 240 minute weighted interaction
* Conditional synergy logic
* Defensive structural realism
* Controlled caps to reduce volatility
* Separate mean + variance engine
* Fully modular architecture

This is Phase 2 backbone quality.

### Step 4: Injury and Availability for each player in a given season.

Injury / Availability Baseline

You do not currently have:

Expected games played per player

Availability-adjusted team strength

For margin-based predictive modeling:
This is acceptable for now.

But if building season simulation, eventually you’ll need:

Expected availability modifier
