Excellent.

We now have clear constraints:

* ✅ Lower overall rank volatility (small, layered changes)
* ✅ Optimize for predictive power + stability
* ✅ Do not over-optimize or distort distributions
* ✅ Accept defense is inherently noisier
* ✅ Improve archetype confidence carefully
* ✅ Preserve geometry integrity

This is a disciplined modeling objective.

Below is the full v3.1 experimental architecture plan.

---

# 🔵 BKE v3.1 – Stability & Expressiveness Optimization Plan

v3.1 is not a redesign.
It is a layered variance-control and geometry refinement update.

We will move along the **stability ↔ expressiveness frontier** carefully.

Each layer will be implemented independently, benchmarked, and reported.

---

# 📊 Baseline (v3.0 Reference Snapshot)

Key anchors:

* OBKE/DBKE weight: 0.60 / 0.40
* dbke_yoy_corr = 0.709
* def_specialist_dbke_yoy_corr = 0.421
* def_driver_share = 0.363
* Defensive archetype stability ≈ 0.536
* Center next-season rho ≈ 0.409 (weakest position)
* Portable rho > Total Impact rho

These are our fixed comparison anchors.

---

# 🧪 Experimental Framework Rules

For every experiment we record:

1. Global YoY rank correlation
2. Specialist YoY (Pearson + Spearman)
3. Top-10 & Top-20 retention
4. Predictive rho (holdout)
5. Position-split rho
6. Archetype confidence
7. Std(DBKE_final)
8. Penalty asymmetry
9. Rank volatility distribution (mean absolute rank shift)

We generate delta vs v3.0.

No silent geometry shifts allowed.

---

# 🟢 LAYER 1 — Offense / Defense Weight Grid

### Purpose:

Test macro stability sensitivity.

### Grid:

* 50/50
* 53/47
* 55/45
* 57/43
* 60/40

### What we expect:

* Increasing offense weight → higher global YoY
* Minor effect on defensive specialist YoY
* Slight improvement in predictive rho
* Reduced overall rank variance

### Selection Criteria:

Choose weight where:

* Predictive rho peaks or plateaus
* Global YoY improves
* Defensive identity not suppressed
* def_driver_share stays within 0.32–0.38

We expect likely winner:
53/47 or 55/45.

This becomes the new base weight.

---

# 🟢 LAYER 2 — Specialty-Aware Variance Dampening (Continuous)

### Purpose:

Reduce cross-axis volatility amplification.

### Core idea:

If a player has a demonstrated specialty,
non-specialty axis fluctuations should move rank slightly less.

### Implementation:

Define specialization index:

[
S = \frac{|O - D|}{|O| + |D|}
]

Weight adjustment:

[
w_O' = w_O \cdot (1 + \alpha S \cdot sign(O-D))
]
[
w_D' = w_D \cdot (1 - \alpha S \cdot sign(O-D))
]

Where:

* α = small constant (0.03–0.07 test range)
* Continuous, no thresholds

Normalize weights to sum to 1.

### Expected Effect:

* Lower mean absolute rank shift
* Reduced star volatility
* Minimal predictive impact

We test α = 0.03, 0.05, 0.07.

---

# 🟢 LAYER 3 — Defensive Tail Micro-Convex Scaling

### Purpose:

Improve elite separation without increasing noise.

### Implementation:

[
D_{new} = \mu + sign(D-\mu)\cdot |D-\mu|^{1.05}
]

Test exponents:

* 1.03
* 1.05
* 1.08

Constraints:

* std(DBKE) increase < 5%
* No collapse in YoY

Expected outcomes:

* Improved Spearman specialist
* Improved top-10 retention
* Reduced clustering-induced rank churn

---

# 🟢 LAYER 4 — Confidence-Weighted Defensive Blend

### Purpose:

Reduce RAPM over-reliance when archetype confidence is low.

Current:
D = 0.55 D_port + 0.45 D_stabilized

New:

[
w_{rapm} = 0.45 \cdot (0.9 + 0.2 \cdot conf)
]

Where:

* conf ∈ [0,1] normalized archetype confidence
* Low confidence → slight reduction in RAPM weight
* High confidence → slight increase

Range:
RAPM weight varies 0.40–0.50 max.

This is small, smooth, data-driven.

Expected:

* Slight improvement in specialist YoY
* Improved center stability
* Higher predictive rho

---

# 🟢 LAYER 5 — Defensive Archetype Dimensional Cleanup

### Purpose:

Increase archetype confidence without affecting ratings.

### Steps:

1. Run PCA on defensive dimensions.
2. Remove collinear axes (|r| > 0.85).
3. Reconstruct archetypes in orthogonal space.
4. Recalculate fit scores.

Constraints:

* Do not change player ratings.
* Only improve cluster geometry.

Expected:

* Archetype confidence ↑
* Defensive archetype stability ↑
* No change in rank volatility

This is low risk and likely high reward.

---

# 🟢 LAYER 6 — Historical Axis Volatility Scaling

### Purpose:

Reduce noise from historically volatile dimensions.

For each axis:

[
AxisScale = \frac{1}{1 + k \cdot historical_variance}
]

Small k (0.1–0.2 test range).

Apply scaling only to deviation component (not mean).

Expected:

* Reduced rank churn
* Increased YoY
* Slightly smoother transitions

Must confirm predictive rho does not drop.

---

# 🟢 LAYER 7 — Combined Model Evaluation

After testing individually:

Combine winning versions of:

* Weight selection
* Specialty dampening
* Tail scaling
* Confidence blend
* Dimensional cleanup

Then re-run:

* Full backtest
* Stability diagnostics
* Position splits
* Distribution checks

No more than 3 structural changes combined at once.

---

# 📈 Evaluation Priorities (Ranked)

1. Predictive rho
2. Global YoY
3. Mean absolute rank shift
4. Defensive specialist stability
5. Archetype confidence
6. Distribution integrity

We will not sacrifice 1 or 2 for 4.

---

# 🎯 Success Criteria for v3.1

* Global YoY ↑ modestly (target ~0.72–0.74)
* Specialist YoY ≥ 0.42 maintained or slightly improved
* Center rho improves from 0.409 → ≥0.45
* Mean rank shift ↓
* Archetype confidence +5–10%
* No collapse in defensive separation

---

# 🔬 What We Are NOT Doing

* No drastic shrinkage increases
* No hard-coded positional weights
* No heavy prior expansion
* No defensive compression

v3.1 is refinement, not reconstruction.

---

# 🧠 Strategic Philosophy of v3.1

We are not trying to eliminate defensive noise.

We are trying to:

* Prevent noise from distorting rank
* Improve defensive clarity
* Preserve expressive ceiling

This is about intelligent variance management.

---

If you’d like next, I can:

1. Convert this into an implementation checklist script plan, or
2. Help you decide the exact order we execute the layers in code, or
3. Simulate expected metric movement before we run experiments

Your call.

---

# Execution Addendum (2026-02-28)

This plan is now accompanied by an implementation policy used in active runs:

1. **Dual split policy is persistent**
	- Co-produce both `60/40` and `55/45` BKE outputs in v3.1 artifacts.
	- Treat `60/40` as default stability anchor and `55/45` as predictive companion split.

2. **Second-pass Layer 3 + Layer 6 profile is approved when additive checks pass**
	- Run with no Layer 1 weight shift (keep `60/40`).
	- Apply Layer 6 axis-volatility scaling and Layer 3 defensive-tail scaling.
	- Promote to output JSON only if additive checks confirm non-canceling stability gains.

3. **Viewer requirement**
	- Player viewer should expose split toggle (`60/40`, `55/45`) alongside version toggle so comparisons remain seamless.

4. **Experiment 2 rerun requirement (2026-03-01)**
	- Re-run production-importance tilt with expanded production proxy, not just TS% + ORAPM.
	- Include raw offensive box-score stats in the proxy: `PTS`, `AST`, `FGM`, `FGA`, `FG3M`, `FG3A`, `FTM`, `FTA`.
	- Sweep exactly 15 lambda values (`0.01` to `0.15` in 0.01 increments).
	- Report both low-production top-100 reduction and high-production top-100 increase to distinguish penalty-vs-reward effects.

5. **Phase transition prep**
	- Add `src/player_eval/` as dedicated next-phase script root for player evaluation engine work.
	- Keep `readme.md` and loop docs synchronized with new reports/outputs and folder conventions.
