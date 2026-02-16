# BKE Metric Modelling Summary (v1.0)

## Overview - What we built and how it works

The BKE metric is a modern, EPM-inspired, prior-driven impact model that fuses offensive and defensive value signals. It is designed to be robust, role-aware, and future-facing, combining:
- A strong prior from archetype, box score, tracking stats, and linear advanced stats
- RAPM (lineup impact, with limited influence)
- DARKO (predictive projection, with limited influence)
- Linear advanced metrics: BPM, WS, VORP, GMSC, TS%, eFG%, TOV%, REB (OREB%, DREB%), Four Factors, etc.
- Context controls (minutes, role stability, team context)


**Guiding principles:**
- Strict separation: role assignment (archetype) is independent from value estimation (BKE)
- BKE_O (offense) and BKE_D (defense) are modeled separately, then summed for BKE (net impact)
- Uncertainty is quantified and reported

---

## Step-by-Step Process

### 1. Data Foundation

**Statistical:**
- Collect possession-level data, box score, tracking, matchup context, and external model feeds (DARKO)
- Normalize all features to comparable rates (per 100 possessions, z-scores)

**Basketball:**
- Capture what happened on the court, who played, and the context for each possession
- Ensure fair comparison across roles, minutes, and teams

---


### 2. Prior Construction (Offense/Defense)

**Statistical:**
- Build a prior estimate for each player using:
  - Archetype (role/behavior)
  - Box score stats (traditional and rate-based)
  - Linear advanced stats:
    - BPM (Box Plus-Minus), OBPM, DBPM
    - WS (Win Shares), OWS, DWS
    - VORP (Value Over Replacement Player)
    - GMSC (Game Score)
    - TS% (True Shooting), eFG% (Effective FG%), TOV% (Turnover Rate)
    - OREB%, DREB%, REB%
    - Four Factors (shooting, turnovers, rebounding, free throws)
    - Any other context-rich, interpretable stat
  - Tracking stats (e.g., contest %, matchup data)
  - Context (minutes, team, role stability, usage)
- Separate offensive and defensive priors:
  - Offense: usage, efficiency, creation, archetype, context, **OBPM, OWS, TS%, eFG%, TOV%, OREB%**
  - Defense: matchup, contest, archetype, context, **DBPM, DWS, DREB%, STL%, BLK%, defensive Four Factors**
- Use regression or machine learning (ridge, elastic net, tree-based, etc.) to predict impact from these features, **excluding RAPM and DARKO at this stage**.

**Mathematical:**
- Prior_O = $f_O$(archetype, box, tracking, linear stats, context)
- Prior_D = $f_D$(archetype, box, tracking, linear stats, context)

**Basketball:**
- Estimate a player’s value based on their role, stats, and context before seeing impact/projection data
- Offensive prior: “How much value does this player create given their role, box/advanced stats, and context?”
- Defensive prior: “How much value does this player provide as a defender given their role, advanced stats, and context?”

---


### 3. Impact Signal Integration (RAPM, DARKO)

**Statistical:**
- Incorporate RAPM (on-court impact) and DARKO (projection) as **independent, reliability-weighted signals**
- Standardize all signals and limit their influence in the fusion model (e.g., via regularization, explicit weight caps, or stacking meta-learner constraints)
- **Linear advanced stats remain in the feature set at this stage** to ensure their continued influence

**Mathematical:**
- RAPM_O, RAPM_D: Offensive/defensive RAPM coefficients (with capped/regularized influence)
- DARKO_ODPM, DARKO_DDPM: Offensive/defensive DARKO projections (with capped/regularized influence)

**Basketball:**
- RAPM: “What has your impact looked like in real lineups?”
- DARKO: “What should we expect going forward?”
- Both signals are used, but BKE can disagree if evidence supports it, and **neither can dominate the final estimate**

---


### 4. Ensemble Fusion (Bayesian/Stacking)

**Statistical:**
- Fuse prior, RAPM, DARKO, and **linear advanced stats** using Bayesian regression, stacking, or other ensemble methods
- Explicitly limit the influence of RAPM and DARKO (e.g., via regularization, meta-learner constraints, or explicit feature weighting)
- Ensure that **linear advanced stats and prior features always contribute** to the final estimate, preventing overfitting to RAPM/DARKO
- Reliability weighting by sample size, role stability, and model agreement

**Mathematical:**
- $BKE_O = f(Prior_O, RAPM_O, DARKO_ODPM, OBPM, OWS, TS\%, eFG\%, TOV\%, OREB\%, FourFactors, archetype, context, reliability)$
- $BKE_D = f(Prior_D, RAPM_D, DARKO_DDPM, DBPM, DWS, DREB\%, STL\%, BLK\%, FourFactors, archetype, context, reliability)$
- $BKE = BKE_O + BKE_D$

**Basketball:**
- Combine all evidence—prior, advanced stats, RAPM, DARKO—to estimate a player’s offensive and defensive impact
- Model can “moderate” or “amplify” signals based on context and reliability, but **no single signal can override the others**
- Final BKE is robust, explainable, and role-aware, with classic advanced stats always visible in the model’s logic

---


### 5. Calibration and Uncertainty

**Statistical:**
- Calibrate BKE by archetype cohort and context to prevent role inflation
- Estimate uncertainty/confidence based on sample size, model disagreement, and role stability

**Mathematical:**
- $BKE_{uncertainty} = g(sample\_size, model\_disagreement, role\_stability)$

**Basketball:**
- Ensure value estimates are realistic and not inflated for rare roles
- Use uncertainty to flag speculative or low-confidence estimates

---


### 6. Final Metric Computation

**Statistical:**
- Output:
  - $BKE_O$: Offensive impact per 100 possessions
  - $BKE_D$: Defensive impact per 100 possessions
  - $BKE$: Net impact ($BKE_O + BKE_D$)
  - $BKE_{uncertainty}$: Confidence interval width
  - $BKE_{tier}$: Percentile-based tiering

**Basketball:**
- Use BKE for overall impact ranking, lineup fit, and role-specific comparison
- Use uncertainty to separate “strong conviction” from “speculative upside”
- Use archetypes to frame *how* value is created, not *whether* value exists

---

## Mathematical/Statistical Definitions

- **Linear regression:** $y = X\beta + \epsilon$
- **Ridge regression:** $y = X\beta + \epsilon$, with $\lambda \sum \beta^2$ penalty
- **Bayesian regression:** Posterior combines prior and likelihood
- **Ensemble stacking:** Combines multiple models for improved prediction
- **Reliability weighting:** Larger samples get more trust
- **Calibration:** Align predicted values with observed outcomes across cohorts
- **Uncertainty estimation:** Quantifies confidence in predictions

---

## Basketball Translation (Step-by-Step)

1. **Data foundation:** “We know who played, what happened, and the context for every possession.”
2. **Prior construction:** “Estimate value based on role and stats, before seeing impact/projection.”
3. **Impact signals:** “Add evidence from lineup impact and talent projection.”
4. **Fusion:** “Combine all evidence, allowing the model to disagree when justified.”
5. **Calibration:** “Check for role inflation and flag low-confidence estimates.”
6. **Final metric:** “Produce robust, role-aware, and future-facing value estimates for ranking and decision-making.”

---


## Example Formulae

- $BKE_O = f(Prior_O, RAPM_O, DARKO_ODPM, OBPM, OWS, GMSC, TS\%, eFG\%, TOV\%, OREB\%, FourFactors, archetype, usage, efficiency, context, reliability)$
- $BKE_D = f(Prior_D, RAPM_D, DARKO_DDPM, DBPM, DWS, GMSC, DREB\%, STL\%, BLK\%, FourFactors, archetype, matchup, context, reliability)$
- $BKE = BKE_O + BKE_D$
- $BKE_{uncertainty} = g(sample\_size, model\_disagreement, role\_stability)$

---

## Deliverables (v1.0)
- Offensive and defensive BKE computed separately, then summed
- Uncertainty/confidence reported for each player-season
- **Linear advanced stats (BPM, WS, VORP, GMSC, TS%, eFG%, TOV%, REB, Four Factors, etc.) are always included as core features**
- RAPM and DARKO influence is explicitly limited to prevent overfitting and ensure all signals contribute
- Role and value strictly separated in modeling pipeline
- Modular, explainable, and basketball-first architecture
