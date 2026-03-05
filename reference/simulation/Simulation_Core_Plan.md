Understood. We are focusing strictly on **margin-based simulation core – Step 1**, no player stat simulation, no lineup logic, no playoffs adjustments yet.

Below is the **fully formal mathematical specification**, followed by a **precise engineering roadmap** for the five layers, and then a **6th validation layer against real team records**.

---

# SIMULATION CORE — STEP 1

(Margin-Based, Team-Level)

---

# I. Mathematical Specification

---

## 1️⃣ Inputs

From the Player Evaluation Engine (Step 3 output):

For each team ( i ):

* ( \mu_i ) = projected team net rating (points per 100 possessions)
* ( \sigma_i ) = projected team volatility (std dev per 100 possessions)

Global constants:

* ( \sigma_{league} ) = irreducible game-level noise floor
* ( H ) = home court advantage (in net rating points)

All values are expressed in net rating scale (per 100 possessions).

---

## 2️⃣ Game Margin Model

For a single game:
Home team ( A ), Away team ( B )

### Expected Margin

[
\Delta \mu = (\mu_A + H) - \mu_B
]

This is the expected net rating difference.

---

### Game Variance

Assume independent team volatility contributions plus irreducible noise:

[
\sigma_{game}^2 = \sigma_A^2 + \sigma_B^2 + \sigma_{league}^2
]

Then:

[
\sigma_{game} = \sqrt{\sigma_A^2 + \sigma_B^2 + \sigma_{league}^2}
]

---

### Margin Distribution

[
M_{A,B} \sim \mathcal{N}(\Delta \mu, \sigma_{game})
]

Interpretation:

Game margin is normally distributed with:

* Mean = expected strength gap
* Variance = team volatility + structural randomness

---

## 3️⃣ Win Probability (Primary Output)

We do NOT simulate to compute win probability.

We compute analytically:

[
P(A \text{ wins}) = \Phi \left( \frac{\Delta \mu}{\sigma_{game}} \right)
]

Where:

* ( \Phi ) = standard normal CDF.

This is the core prediction metric.

Monte Carlo is used only for distributional outputs.

---

## 4️⃣ League Noise Floor

[
\sigma_{league} \approx 3.0
]

Purpose:

* Prevent overconfidence
* Prevent volatility collapse
* Capture refereeing variance, random shooting variance, injuries mid-game, etc.

This parameter must be tuned later via:

* Brier score minimization
* Log loss minimization

---

## 5️⃣ Season Simulation

For a season schedule ( S ):

For each game ( g ):

1. Compute ( \Delta \mu_g )
2. Compute ( \sigma_{game,g} )
3. Sample:

[
M_g^{(k)} \sim \mathcal{N}(\Delta \mu_g, \sigma_{game,g})
]

4. If ( M_g > 0 ) → home team wins.

Repeat for all 82 games.

Repeat entire season ( N ) times:

[
N = 10,000 \text{ recommended}
]

Outputs per team:

* ( E[\text{Wins}] )
* Win variance
* Win distribution
* Playoff probability (if threshold defined)
* Seed distribution (if conference modeling included)

---

# II. Engineering Roadmap

Now we zoom into the five structural layers.

---

# Layer 1 — Parameter Layer

### Goal:

Define and manage model constants and hyperparameters.

### Parameters:

* `sigma_league`
* `home_court_advantage`
* `season_simulations`
* random seed

### Design Decisions:

1. Store in immutable config object.
2. Allow override via config file (YAML/JSON).
3. Maintain separate calibration config and production config.

### Mathematical Tasks:

* Initialize ( \sigma_{league} = 3.0 )
* Initialize ( H = 1.5 ) to 2.0 (tunable)

---

# Layer 2 — Deterministic Game Model

### Goal:

Produce expected margin and win probability WITHOUT simulation.

### Function Specification

```
compute_game_distribution(team_A, team_B, is_home_A=True)
```

Returns:

* delta_mu
* sigma_game
* win_prob_A

### Exact Computation Steps

1. Retrieve ( \mu_A, \sigma_A )
2. Retrieve ( \mu_B, \sigma_B )
3. Apply:

[
\Delta \mu = (\mu_A + H) - \mu_B
]

4. Compute:

[
\sigma_{game} = \sqrt{\sigma_A^2 + \sigma_B^2 + \sigma_{league}^2}
]

5. Compute:

[
z = \frac{\Delta \mu}{\sigma_{game}}
]

6. Compute:

[
P = \Phi(z)
]

Return all values.

This layer must be pure and stateless.

---

# Layer 3 — Schedule Engine

### Goal:

Represent season schedule independently from engine math.

### Data Structure:

```
Game:
    home_team_id
    away_team_id
    date
    game_id
```

```
SeasonSchedule:
    list[Game]
```

### Requirements:

* No model math inside this layer.
* Just scheduling structure.
* Should support:

  * Real NBA schedule
  * Synthetic schedule
  * Custom league schedule

---

# Layer 4 — Monte Carlo Simulation Engine

### Goal:

Simulate full season distribution.

### Core Loop

For simulation k in 1..N:

For each game g:

1. Call deterministic model to get:

   * delta_mu
   * sigma_game

2. Sample:

[
M_g \sim \mathcal{N}(\Delta \mu_g, \sigma_{game,g})
]

3. Update standings.

---

### Efficiency Requirements

* Vectorize sampling if possible.
* Precompute delta_mu and sigma_game for entire schedule.
* Avoid recomputing inside loop.

---

### Output Storage

For each team:

Store:

* wins array size N
* losses array size N

After simulation:

Compute:

* mean wins
* std wins
* percentile wins
* probability wins ≥ X

---

# Layer 5 — Aggregation Layer

### Goal:

Convert raw simulation output into prediction outputs.

For each team:

Compute:

1. Expected wins:

[
E[W_i] = \frac{1}{N} \sum_{k=1}^{N} W_i^{(k)}
]

2. Win variance:

[
Var(W_i)
]

3. Playoff probability:

[
P(W_i \geq \text{playoff cutoff})
]

4. Seed probabilities (if conferences implemented)

This layer has no simulation logic.

It only aggregates.

---

# Layer 6 — Validation Layer (Against Real Records)

This is critical.

---

## A. Game-Level Validation

Using historical seasons:

For each real game:

1. Compute predicted win probability.
2. Compare with actual outcome.

Metrics:

* Brier score:

[
\frac{1}{n} \sum (p_i - y_i)^2
]

* Log loss:

[
-\frac{1}{n} \sum [y_i \log(p_i) + (1-y_i)\log(1-p_i)]
]

* Calibration curve
* Reliability diagram

---

## B. Margin Validation

Compare predicted margin distribution to real margin distribution.

Metrics:

* RMSE of margin
* KS test on distribution shape
* Compare variance calibration

---

## C. Season-Level Validation

After simulating season:

Compare:

* Expected wins vs actual wins
* MAE
* RMSE
* Correlation r

[
r(\text{predicted wins}, \text{actual wins})
]

---

## D. Overconfidence Test

Check:

* Are 70% win probability games winning ~70%?
* Are 90% games overconfident?

If overconfident → increase ( \sigma_{league} )

---

## E. Parameter Tuning Loop

Tune:

* ( \sigma_{league} )
* ( H )

Objective:

Minimize:

* Weighted combination of:

  * Log loss
  * Brier score
  * Margin RMSE

This becomes optimization problem:

[
\min_{\sigma_{league}, H} L(\theta)
]

Use grid search initially.

---

# Final Architecture Summary

Step 1 Simulation Core consists of:

1. Parameter Layer
2. Deterministic Game Model
3. Schedule Engine
4. Monte Carlo Engine
5. Aggregation Layer
6. Validation Layer

No lineup logic.
No playoffs adjustments.
No player stat simulation.
Pure margin math.

Your adjustment makes sense and actually improves identifiability of the lineup model. Using **clutch data first** gives us a real behavioral signal from coaches instead of purely inferred roles.

Below is the **clean Step-2 design**, incorporating your decisions and tightening the math.

---

# Simulation Core — Step 2: Lineup Projection Model

Purpose of Step 2:

Convert **player-level impact ratings from PEC Step 3** into **three team strength phases** that more closely match how NBA games are played.

Those phases are:

```
1. Starters
2. Rotation strength
3. Clutch lineup
```

These will later feed into **Step 3 game simulation weighting**.

Step 2 adds **coaching behavior and lineup structure** without introducing archetype interactions yet.

---

# Design Philosophy

Real NBA games operate in **three structural phases**:

### Phase 1 — Opening Lineups

First ~6 minutes of each half.

Characteristics:

* highest continuity
* strongest positional structure
* often includes all stars

---

### Phase 2 — Rotations

Middle ~60% of game.

Characteristics:

* staggered stars
* bench players introduced
* performance depends heavily on **bench depth**

---

### Phase 3 — Clutch

Final ~5 minutes when game margin is small.

Characteristics:

* best players reinserted
* coaching preferences dominate
* small-ball or specialist lineups appear

---

The simulator must capture **these phases separately**.

---

# Input Data

From PEC Step 3:

For each player ( i )

```
impact_i
volatility_i
minutes_i
position_profile_i
defensive_archetype_band_i
```

Additional dataset required:

```
NBA clutch statistics
```

Fields needed:

```
clutch_minutes
team_id
player_id
```

Clutch definition:

```
score margin ≤ 7
last 5 minutes of game
```

---

# Layer 1 — Clutch Lineup Model

Clutch lineups are the **most reliable observable lineup signal**.

We combine:

```
80% clutch minutes signal
20% best-player signal
```

This prevents small-sample clutch data from dominating.

---

## Step 1A — Compute Clutch Share

For each player:

[
C_i = \frac{\text{clutch minutes}_i}{\text{team clutch minutes}}
]

Normalize:

[
C_i \in [0,1]
]

---

## Step 1B — Normalize Player Strength

Impact rating normalized within team:

[
I_i = \frac{impact_i - \min(impact)}{\max(impact)-\min(impact)}
]

Minutes normalized:

[
M_i = \frac{minutes_i}{\max(minutes)}
]

---

## Step 1C — Clutch Score

[
S_{clutch,i}
============

0.80 \cdot C_i
+
0.20 \cdot
(0.65 I_i + 0.35 M_i)
]

Interpretation:

* clutch minutes dominate
* elite players without clutch samples still rank high

---

## Step 1D — Clutch Lineup Optimization

Select lineup ( L ) maximizing:

[
\max_{L}
\sum_{i \in L} S_{clutch,i}
]

Subject to positional constraints:

```
≥1 guard
≥1 wing
≥1 big
```

No upper bounds.

This preserves positional logic without over-restricting coaches.

---

## Step 1E — Clutch Unit Strength

Let clutch lineup be ( L_c )

Mean:

[
\mu_{clutch}
============

\frac{\sum_{i \in L_c} impact_i \cdot minutes_i}{\sum_{i \in L_c} minutes_i}
]

Volatility:

[
\sigma_{clutch}^2
=================

\frac{1}{5}\sum_{i \in L_c} volatility_i^2
]

---

# Layer 2 — Starter Projection

Now we incorporate your **clutch bias into starters**.

Reason:

Teams often close with their starters.

But not always.

So clutch lineup should influence starters slightly.

---

## Starter Score

Define normalized variables again:

```
M_i = normalized minutes
I_i = normalized impact
C_i = clutch share
```

Starter score:

[
S_{start,i}
===========

0.50M_i
+
0.25I_i
+
0.10Pos_i
+
0.15C_i
]

Where:

```
Pos_i = positional scarcity score
```

This implements your idea:

```
-5% minutes
-10% impact
+15% clutch bias
```

---

## Starter Optimization

Select lineup ( L_s ) maximizing:

[
\sum_{i \in L_s} S_{start,i}
]

Subject to:

```
≥1 guard
≥1 wing
≥1 big
```

---

## Starter Strength

[
\mu_{start}
===========

\frac{\sum_{i \in L_s} impact_i \cdot minutes_i}{\sum_{i \in L_s} minutes_i}
]

Volatility:

[
\sigma_{start}^2
================

\frac{1}{5}\sum_{i \in L_s} volatility_i^2
]

---

# Layer 3 — Rotation Strength Model

Rotation minutes represent **bench depth and staggered stars**.

Instead of predicting a lineup we estimate the **expected rotation strength**.

This avoids unrealistic substitution assumptions.

---

## Step 3A — Identify Bench Players

Bench set:

[
B = \text{players not in starter lineup}
]

---

## Step 3B — Bench Impact

[
BenchImpact =
\frac{\sum_{i \in B} impact_i \cdot minutes_i}
{\sum_{i \in B} minutes_i}
]

---

## Step 3C — Staggered Starter Contribution

Some starters appear heavily in rotation units.

Define:

[
R_i = \max(0, minutes_i - 24)
]

Players above ~24 minutes usually appear in staggered rotations.

Compute:

[
StaggerImpact =
\frac{\sum_{i \in L_s} impact_i \cdot R_i}
{\sum_{i \in L_s} R_i}
]

---

## Step 3D — Rotation Strength

Combine both signals:

[
\mu_{rotation}
==============

\alpha StaggerImpact
+
(1-\alpha) BenchImpact
]

Where:

[
\alpha = 0.55
]

Reason:

Rotation units usually include **2-3 starters**.

---

## Rotation Volatility

Bench units are less stable.

[
\sigma_{rotation}^2
===================

\sigma_{bench}^2
+
0.5\sigma_{stagger}^2
]

---

# Final Step 2 Output

For each team:

```
team_lineup_profile = {

  starters:
    players
    mu_start
    sigma_start

  rotation:
    mu_rotation
    sigma_rotation

  clutch:
    players
    mu_clutch
    sigma_clutch

}
```

Rotation intentionally **does not store a player list**.

It is an aggregate phase.

---

# Validation Layer

Step 2 must be validated before integrating with the simulator.

---

## Starter Prediction Accuracy

Compare predicted starters with real starters.

Metric:

```
player overlap
```

Target:

```
≥80%
```

---

## Clutch Lineup Accuracy

Compare predicted clutch lineup with real clutch lineup.

Metric:

```
overlap
```

Target:

```
≥3.5 players out of 5
```

---

## Rotation Model Validation

Use historical lineup net ratings.

Compare:

```
predicted rotation strength
vs
actual non-starter lineup net rating
```

Metric:

```
correlation r
```

Target:

```
r ≥ 0.70
```

---

# Why This Step Matters

Without lineup modeling the simulator assumes:

```
team strength is constant across the game
```

But real NBA games vary by **phase**.

Example:

```
Celtics starters elite
bench weak
clutch strong
```

Step 2 captures those differences.

---