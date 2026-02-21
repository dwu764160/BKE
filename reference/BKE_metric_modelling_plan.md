🧠 BKE v2.8 Blueprint

Theme: Distribution Integrity, Signal Preservation, and Final BKE Consolidation
Goal: Fix compression, variance collapse, and percentile distortion while avoiding overfitting.

I. Core Objectives of v2.8

v2.8 is built around four priorities, all fully addressed:

Priority A — Fix Distribution Compression
Priority B — Preserve Meaningful Variance Across Players
Priority C — Prevent Double-Smoothing / Signal Dilution
Priority D — Produce Final Comprehensive BKE (OBKE + DBKE)

Additionally:

Avoid re-percentiling intermediate layers (per v2.5 rule)

Use Option B for percentile math (monotonic transforms preserved)

Keep raw composites internally continuous

Produce final standardized BKE outputs

II. The Distribution Problems (Explicitly Addressed)

You identified three major distribution risks (a, b, c). v2.8 directly solves them:

🔴 Problem A: Percentile Compression at Upper Tiers
Issue

When many elite players cluster in high percentiles (e.g., 92–99), additive structures flatten real separation.

This:

Diminishes elite separation

Artificially tightens rankings

Creates fragile ordering

v2.8 Fix — Nonlinear Stretch Before Final Percentiling (Option B)

We introduce:

Step 1: Build raw continuous composite scores (NO percentiles)

Dimension composites remain raw weighted sums.

Step 2: Apply monotonic variance-preserving transform:

Choose one:

Logit stretch

Z-score standardization

Rankit transformation (Blom adjustment)

Mild exponential stretch on upper tail

Preferred for v2.8:
Z-score → logistic rescale

This:

Preserves ordering

Expands upper and lower tails

Prevents 92–99 compression

Only AFTER this transformation do we compute final percentiles.

This avoids percentile-on-percentile collapse.

🔴 Problem B: Variance Collapse Through Layer Aggregation
Issue

Layer stacking causes variance to shrink:

Dimension → Layer → OBKE → BKE

Each weighted average narrows spread

Over time:

Players converge artificially

Extreme profiles are muted

v2.8 Fix — Variance Monitoring + Spread Anchoring

We implement:

1. Variance Audit at Every Layer

For each layer:

Mean

Std deviation

IQR

Kurtosis

If std dev shrinks > X% from previous layer:
→ flag compression

2. Spread Anchoring Rule

Each layer must retain at least:

65–75% of variance of underlying dimension aggregate

If below threshold:

Apply variance rescaling (multiply by scaling factor)

Not nonlinear — linear stretch only

This keeps:

Outliers meaningful

Specialists visible

Archetypes intact

🔴 Problem C: Double-Smoothing via On-Off + Percentiles
Issue

On-off metrics are already context-adjusted.
Percentiling them again smooths signal further.

Result:

Role-dependent impact diluted

High-leverage impact players undervalued

v2.8 Fix — Raw On-Off Integration

For on-off metrics:

Convert to possession-normalized rates

Z-score within position group

Use directly in composite

Do NOT percentile until final output

This preserves:

True magnitude

Lineup leverage

Signal strength

III. Layer Integrity Review (Diagnostic Focus)

v2.8 performs a full system audit of:

Layer 1c (Dimensions) — Distribution Diagnostics

Each of the 8 dimensions:

Shooting Gravity

Driving Gravity

Playmaking

Extra Possession Creation

Defensive Playmaking

Defensive Impact

Turnover Control

Defensive Versatility

For each:

Add:

Raw score distribution plot

Z-score histogram

Percentile spacing check

Tail separation measurement

We compute:

90–95 separation

95–99 separation

Top 5 players raw spread

If spacing too narrow:

Adjust weighting

Apply mild nonlinear stretch

Or increase raw component weighting

IV. Offensive / Defensive Split Stabilization
OBKE = Weighted sum of 4 offensive dimensions
DBKE = Weighted sum of 4 defensive dimensions

In v2.8:

We implement:

1. Independent normalization pipelines

OBKE and DBKE remain raw until final step.

2. Offensive-Defensive Balance Audit

Compute:

Correlation between OBKE and DBKE

Variance comparison

Relative scale ratio

If one side consistently dominates magnitude:
→ Apply scale normalization to equalize variance contribution.

We want:

OBKE and DBKE to have similar distribution widths

Not identical means — just comparable spread

V. Removal of Portability Ratio (Finalized)

Per v2.5 decision:

Portability ratio removed from reports

Portability concept remains embedded in layer weights

No explicit scalar displayed

This avoids:

Misleading interpretation

Artificial simplification of multidimensional impact

VI. Overfitting Guardrails

To avoid overfitting while expanding diagnostics:

1. No manual player-by-player tuning

All changes must:

Apply league-wide

Be formulaic

Be reproducible

2. Archetype separation test

If top 20 players become homogeneous:
→ investigate over-smoothing.

3. Holdout validation

Split players into:

High minutes

Medium minutes

Ensure distributions hold in both.

VII. Final Comprehensive Output — BKE

v2.8 produces FOUR output files:

1️⃣ dimension_scores.json

Raw + z-score for each of 8 dimensions

2️⃣ layer_scores.json

Layer composites before final transformation

3️⃣ obke_dbke_scores.json

OBKE + DBKE raw and standardized

4️⃣ bke_scores.json (NEW)

For each eligible player:

{
  player_name:
    raw_OBKE,
    raw_DBKE,
    transformed_OBKE,
    transformed_DBKE,
    final_OBKE_percentile,
    final_DBKE_percentile,
    raw_BKE,
    transformed_BKE,
    final_BKE_percentile,
    rank
}
VIII. Final BKE Construction

Step-by-step:

OBKE_raw = weighted offensive composite

DBKE_raw = weighted defensive composite

BKE_raw = OBKE_raw + DBKE_raw

Apply monotonic transform to BKE_raw

Compute final percentile (league-wide eligible players)

This is the ONLY percentile that matters for ranking.

No re-percentiling earlier layers.

IX. Deliverables for v2.8
Diagnostic Outputs:

Variance report per layer

Compression report

Tail separation metrics

OBKE/DBKE balance stats

Correlation matrix of dimensions

Production Outputs:

dimension_scores.json

layer_scores.json

obke_dbke_scores.json

bke_scores.json

X. What v2.8 Is NOT Doing

No new dimensions

No weight overhaul

No portability reintroduction

No archetype modeling changes

No predictive validation yet

This is structural integrity and signal preservation only.

XI. Implementation Checklist

Before marking 2.8 complete:

 All dimensions show healthy spread

 Upper tail separation preserved

 OBKE and DBKE variances comparable

 No double percentiling

 On-off integrated raw

 BKE distribution visually normal-like

 No excessive clustering 90–99

 Reports auto-generated

XII. Summary of What v2.8 Changes

Compared to v2.5:

Major Additions

Full distribution diagnostics layer

Variance preservation rules

Tail stretch transformation

OBKE/DBKE balancing audit

Final BKE output file

Major Fixes

Percentile compression

Variance collapse

On-off double smoothing

Upper-tier flattening

Removals

Portability ratio from reports

Architectural Status

Layers remain architecturally intact

Now mathematically stabilized