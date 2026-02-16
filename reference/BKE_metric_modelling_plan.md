# 🧠 v1.5 — Portable Talent vs Role-Dependent Impact Engine

# THIS FILE IS THE "PROPOSED NEXT STEP" for the BKE metric.


And yes — percentile standardization will be foundational, not cosmetic.

---

# 🏗 OVERVIEW: What v1.5 Actually Does

For every player, we estimate:

```
Total Impact
=
Portable Talent
+ Role-Dependent Impact
```

Then we decompose Role-Dependent Impact into:

```
Role-Dependent Impact
=
Role Utilization Efficiency
+ Scheme Amplification
+ Archetype Elevation
```

All layers are percentile-standardized at:

* League-wide level
* Positional cohort
* Archetype cohort

Percentiles are not presentation fluff — they are structural inputs.

---

# 🔹 LAYER 0 — Data Inputs (Foundation)

### Core Data Sources

* Adjusted RAPM (multi-year, Bayesian)
* Play-by-play lineup data
* Playtype efficiency + frequency
* Tracking data (if available)
* On/Off splits
* Shot quality models
* Lineup possession clustering
* Defensive matchup data

---

# 🔹 LAYER 1 — True Portable Talent (Context-Neutral Layer)

This estimates:

> “How good is this player independent of role volume and scheme?”

### 1A. Multi-Year Bayesian RAPM

* 3-year weighted
* Ridge regression
* Prior shrunk toward:

  * Positional mean
  * Archetype mean

Output:

* Raw impact estimate

---

### 1B. Luck Adjustment

Adjust:

* Shooting variance
* Opponent 3PT variance
* FT variance

---

### 1C. Portable Skill Components

We isolate impact that scales across contexts:

* Shooting gravity
* Rim protection
* Passing efficiency
* Defensive versatility
* Turnover control

Each converted to:

* League percentile
* Position percentile
* Archetype percentile

---

### 1D. Portable Talent Score (PTS)

Combine:

```
PTS = Weighted combination of:
  - Adjusted RAPM percentile
  - Portable skill percentiles
  - Stability-adjusted impact
```

Output:

* League Percentile
* Positional Percentile
* Archetype Percentile

This is talent that travels.

---

# 🔹 LAYER 2 — Role Utilization Layer

Now we measure:

> “How much of their talent is actually being expressed in their current role?”

---

### 2A. Usage-Conditioned Efficiency

For each playtype:

```
Adjusted PPP
– League Avg PPP at same usage bucket
= Playtype Surplus
```

Convert each to percentile (3 levels).

Weight by possession share.

This produces:

### Playtype Contribution Vector (PCV)

Example:

Isolation: +88th percentile
PnR Ball Handler: 75th
Spot-Up: 62nd
Transition: 54th

This vector defines current role expression.

---

### 2B. Role Utilization Efficiency (RUE)

We compare:

```
Observed Playtype Vector
vs
Archetype Optimal Playtype Distribution
```

RUE measures:

* How aligned usage is with skill strengths
* Whether player is underutilized or miscast

Output:

* League percentile
* Archetype percentile

---

# 🔹 LAYER 3 — Archetype Elevation

Now we isolate:

> “How much better is this player than the average player in this archetype?”

---

### 3A. Archetype Baseline Impact

For each archetype:

* Compute mean RAPM
* Compute mean playtype surplus
* Compute mean on/off impact

---

### 3B. Elevation Score

```
Player Impact – Archetype Baseline
```

Percentile-normalized within archetype.

This answers:

* Are they archetype-replacement?
* Or archetype-elite?

---

# 🔹 LAYER 4 — Scheme Amplification

This measures context sensitivity.

We estimate:

```
Impact variance across:
  - Lineups
  - Defensive schemes
  - Offensive spacing levels
```

High variance → role-dependent
Low variance → portable

Compute:

* Scheme Stability Index
* Lineup Interaction Coefficient

Standardized via percentiles.

---

# 🔥 FINAL DECOMPOSITION

For each player:

```
Total Impact
=
Portable Talent
+ (RUE + Archetype Elevation + Scheme Amplification)
```

We present:

### 1️⃣ Portable Talent Score (PTS)

How good they are anywhere.

### 2️⃣ Role-Dependent Impact Score (RDIS)

How much impact depends on environment.

### 3️⃣ Portability Ratio

```
Portable Talent / Total Impact
```

High = scalable star
Low = system-amplified player

---

# 📊 Percentile System (CRITICAL ARCHITECTURE)

We standardize EVERYTHING in 3 dimensions:

| Level                 | Purpose               |
| --------------------- | --------------------- |
| League Percentile     | Macro comparison      |
| Positional Percentile | Role fairness         |
| Archetype Percentile  | Micro peer comparison |

Percentiles are used in:

* Regression inputs (as normalized predictors)
* Output presentation
* Cross-era scaling
* Cohort shrinkage

This prevents:

* Position bias
* Archetype inflation
* Volume distortion

---

# 🎯 Final Output Card (Example)

Player X:

**Portable Talent**

* League: 91st
* Position: 94th
* Archetype: 88th

**Role-Dependent Impact**

* League: 72nd
* Archetype: 84th

**Portability Ratio: 0.78**

Interpretation:

* Scales across systems
* Slight scheme amplification
* Elite within archetype

---

# 🧮 Statistical Backbone

Core model:

```
Impact_it =
β1(Portable Talent_it)
+ β2(Role Usage Interaction_it)
+ β3(Scheme Terms_it)
+ ε
```

Bayesian hierarchical structure:

* Level 1: Player
* Level 2: Archetype
* Level 3: Position

Shrinkage applied at each level.

---

# 🛠 v1.5 Implementation Order

1. Lock percentile framework
2. Build portable RAPM layer
3. Build playtype surplus engine
4. Create archetype baseline table
5. Add scheme variance modeling
6. Run decomposition
7. Backtest on role-change players

---

# 🧪 Validation Plan

Test on:

* Players who changed teams
* Role shifts (bench → starter)
* Usage spikes
* Scheme changes (switch-heavy vs drop)

Check:

Does Portable Talent remain stable?
Does Role-Dependent portion fluctuate?

If yes → model works.

---

# 🚀 What This Unlocks

You can now answer:

* “Is this player portable?”
* “Is he being misused?”
* “If we change role, how much impact moves?”
* “Which archetypes generate surplus value?”

This is front-office level.

---

# 🔁 Summary of Changes From v1.4

Compared to previous architecture:

### 1️⃣ Major Conceptual Shift

Old focus: Impact estimation + archetype classification
New focus: **Impact decomposition into portable vs role-dependent**

---

### 2️⃣ Percentiles Moved From Output → Core Infrastructure

Previously:

* Used mostly for display and comparisons.

Now:

* Used in regression inputs
* Used for shrinkage priors
* Used for cohort normalization
* Used in archetype baseline modeling

This is structural.

---

### 3️⃣ Added Playtype Contribution Vector (PCV)

New multi-dimensional vector defining:

* How impact is generated
* How it maps to role

---

### 4️⃣ Added Role Utilization Efficiency (RUE)

Measures:

* Fit between skill and usage
* Under/over-optimization

This did not exist before.

---

### 5️⃣ Added Archetype Elevation Layer

Separates:

* Baseline archetype value
* Individual elevation above archetype

---

### 6️⃣ Added Scheme Amplification Modeling

Quantifies:

* Context sensitivity
* Lineup dependency

---

### 7️⃣ Introduced Portability Ratio

Clean scalar for:

* Trade valuation
* Scalability analysis
* Role projection

---

# 🏆 What v1.5 Is Now

It is no longer:

“A better RAPM with archetypes.”

It is now:

> A multi-layer hierarchical decomposition engine
> that separates scalable talent from system amplification,
> standardized via percentile architecture at three cohort levels.

This is rare.

---

If you'd like, next we can:

* Design the exact regression specification
* Or simulate how this behaves for 3 example player types
* Or define the math for percentile-weighted shrinkage priors

Your move.
