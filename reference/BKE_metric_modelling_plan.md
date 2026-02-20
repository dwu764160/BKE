🧠 PORTABLE TALENT VS ROLE-DEPENDENT IMPACT
📘 Blueprint v2.0 (Mathematical Corrections + Structural Fixes)

We will keep the v1.5 format, but explicitly repair the identified flaws.

I. Core Structural Philosophy (Clarified)

Still:

Total Impact
= Portable Talent (Layer 1)
+ Role Optimization (Layer 2)
+ Archetype Elevation (Layer 3)
+ Scheme Amplification (Layer 4)


BUT:

v2.0 corrects:

Percentile math distortion

False portability ratio

Dimensional misplacement (turnovers)

Cross-layer interpretability conflicts

II. Layer 1 – Portable Talent (Repaired Structure)
Layer 1A – RAPM Backbone
Layer 1B – Playtype Efficiency
Layer 1C – Dimension Model (Primary)


Weighting remains:

25% RAPM
20% Playtype
55% Dimension


But interpretation math changes (explained below).

III. Layer 1C – Portable Dimension Model (Major Revision)

This is now the core portability engine.

We shift from artificial 8 symmetry to 8 logically independent dimensions:

🔵 OFFENSIVE PORTABLE DIMENSIONS
1️⃣ Shooting Gravity (Unchanged)

Measures:

Off-movement 3P%

C&S 3P%

Pull-up 3P%

3PA rate

Shooting percentile under contest

On/off spacing effect (team rim freq when player on court)

Outputs:

Shooting Gravity Score

League / Positional / Archetypal Percentiles

No structural change.

2️⃣ Driving Gravity (Revised)

Key corrections:

❌ Remove perimeter initiation filtering

❌ Remove FT%

✅ Focus on rim pressure creation

Metrics:

Unassisted rim attempts

Rim FGA rate

Fouls drawn per 100

And-1 frequency

Team foul rate delta (on/off)

We are measuring:

Ability to collapse defense and generate foul pressure — not scoring skill.

This is correct refinement.

3️⃣ Playmaking (Unchanged)

Metrics:

Adjusted AST%

Potential assists

Box creation

Pass-to-shot efficiency

Advantage creation events

Percentile normalized.

No change.

4️⃣ Extra Possession Creation (Refined)

Now simplified.

Metrics:

OREB%

DREB%

On/off team rebound rate delta

This dimension contributes to:

Offensive portable score

Defensive portable score

This is the first true cross-domain dimension.

Correct decision.

7️⃣ Turnover Control (Restored + Elevated)

This is important.

Portable skill:

Ability to avoid giving away possessions under any role.

Metrics:

TOV%

Bad pass frequency

Live-ball turnover rate

On/off turnover delta

Uses percentile normalization heavily.

This was a good reintroduction.

🔴 DEFENSIVE PORTABLE DIMENSIONS
5️⃣ Defensive Playmaking (Expanded)

Metrics:

STL%

BLK%

Deflections

Loose balls recovered

Charges drawn

Disruption rate (if available)

This captures chaos creation.

No conceptual change — but hustle stats now emphasized.

6️⃣ Defensive Impact (Unchanged Core)

Still RAPM-informed.

Includes:

On/off defensive rating

Matchup difficulty adjustments

Shot quality allowed

We do NOT double-count rim protection.

Correct removal of redundancy.



8️⃣ Defensive Versatility (Unchanged)

Metrics:

Matchup spectrum

Positional defensive coverage

Switch frequency

Cross-match success

Still portable across schemes.

No change.

- Weighting Adjustments in Layer 1

Because Layer 1C is now primary:

Suggested structure:

Layer 1 = 
  25% RAPM Backbone
  20% Playtype Efficiency
  55% Dimension Model (1C)


Within 1C:

Equal weighting initially across 8 dimensions.

We can later experiment with:

Variance-based weighting

Stability weighting

Predictive weighting

But start equal.

🔁 Extra Possession Creation Handling

Rebounding contributes:

60% to defensive composite

40% to offensive composite

But stored as its own raw dimension before split.

No duplication.

IV. 🚨 FIX #1 – Percentile Additive Distortion
Problem in v1.5:

We were adding percentiles directly:

Final Score = avg(percentiles)


This distorts meaning because:

Percentiles are rank-based, not interval-scaled.

The difference between 90 and 95 ≠ difference between 50 and 55.

Averaging compresses tails and exaggerates middle clusters.

This causes:

Artificial clustering

Misleading comparisons

Poor predictive validity

✅ v2.0 Fix: Convert Percentiles → Z-Scores Before Aggregation

New process:

Compute raw metric

Convert to z-score

Standardize by:

League distribution

Positional distribution

Archetype distribution

Blend standardized z-values

Only at final output convert composite back to percentile

So:

Raw → Z → Weighted Sum → Final Z → Final Percentile


Percentiles become:

Presentation tool

Not aggregation math

This preserves interval meaning.

V. 🚨 FIX #2 – Portability Ratio Was Fake
Problem in v1.5:

We implied:

Portability Ratio = Portable Talent / Total Impact


But this does NOT measure portability.

Why?

Because:

Total Impact already contains portable influence.

Denominator is endogenous.

Ratio shrinks for high-impact players even if portable.

This measures composition — not transfer stability.

✅ v2.0 Fix: True Portability Measurement

We now define portability as:

Stability of impact across context shifts.

New portability measures:

1️⃣ Lineup Stability Index

Variance of impact across:

Different teammate clusters

Different spacing contexts

Different defensive environments

Low variance = high portability.

2️⃣ Role Elasticity Test

Simulate usage shifts:

+5% usage

−5% usage

Recalculate projected impact.

Players whose impact changes minimally = portable.

3️⃣ Archetype Transfer Simulation

Project player into:

3 alternative archetype usage templates

Measure projected efficiency change

Less dropoff = more portable.

4️⃣ On/Off Context Sensitivity

Measure:

On/off impact across:

Bench-heavy lineups

Starter-heavy lineups

Different pace environments

Variance-based portability.

New Portability Index (True Definition)
Portability Index
= 1 – Normalized Impact Variance Across Contexts


This is structural.

Not compositional.

Now it measures what we claim.

VI. Cross-Layer Interpretation Fix

In v1.5 we risked:

Double attributing improvement to Layer 1 and Layer 2.

Mislabeling role efficiency as portable skill.

v2.0 clarification:

Layer 1 measures skill capacity.

Layer 2 measures usage alignment.

Layer 3 measures relative dominance.

Layer 4 measures environmental amplification.

No overlap.

Each layer must be measurable with others held constant.

VII. Weighting Philosophy Correction

We must stop assuming equal dimension weight is optimal.

v2.0 introduces:

Stability-Weighted Dimension Scaling

Dimensions weighted by:

Year-to-year stability

Predictive correlation with future RAPM

Cross-team transfer reliability

Unstable metrics receive shrinkage.

VIII. Bayesian Shrinkage Introduction

v2.0 introduces:

Empirical Bayes shrinkage for:

Defensive playmaking

On/off metrics

Small sample role splits

This prevents noise from inflating portability scores.

IX. v2.0 Mathematical Pipeline
STEP 1: Fetch raw metrics
STEP 2: Clean + adjust for role
STEP 3: Convert to z-scores
STEP 4: Apply shrinkage
STEP 5: Aggregate within dimensions
STEP 6: Aggregate within layers
STEP 7: Simulate context variance
STEP 8: Compute portability index
STEP 9: Convert final composites to percentiles
STEP 10: Output standardized player card

X. Updated Known Fixes Summary
Issue	v1.5 Problem	v2.0 Fix
Turnover placement	Misclassified	Now offensive
Percentile averaging	Rank distortion	Z-score aggregation
Portability ratio	Fake compositional stat	Variance-based stability index
Rim protection redundancy	Double counted	Removed
Driving gravity misdefinition	Included FT%	Removed FT%
Symmetry forcing	4/4 artificial	Dimension-based logic
XI. Remaining v2.0 To-Do List (High Priority)

We still must:

Multi-year stabilization

Playoff portability testing

Aging curve integration

Injury-adjusted variance modeling

Archetype clustering validation

Cross-team transfer case studies

Impact volatility score

Outlier tail handling correction

XII. Is Layer 1C More Important Than Entire Layers?

Yes.

Portable traits can outweigh role optimization entirely.

That is philosophically correct.

Role does not create skill.

Skill survives role.

So the model is now aligned with that principle.

XIII. Summary of Changes from v1.5 → v2.0

Major:

Fixed percentile math distortion.

Rebuilt portability measurement properly.

Corrected turnover dimension classification.

Removed fake ratio logic.

Introduced variance-based portability.

Introduced z-score aggregation.

Introduced shrinkage.

Clarified layer independence.

Removed redundant rim protection.