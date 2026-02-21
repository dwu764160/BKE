BKE v2.7 Blueprint

Goal: Statistical Maturity Pass — Correct Structural Weaknesses Without Adding Features

1️⃣ Replace Hard Archetype Assignment with Probabilistic Membership (CRITICAL)

(Fixes Issue #7 — largest distortion source)

Problem in v2.6

Archetype is hard-labeled:

primary_archetype

Everything depends on it:

Neutralization expectations

RUE optimal vector

Elevation baselines

Portability transfer component

Boundary players get artificial discontinuities.

✅ v2.7 Solution: Soft Archetype Membership

Replace:

player → archetype A

With:

player → {A: 0.52, B: 0.48}
Implementation Plan

A. Use archetype_embeddings.parquet
You already load:

ARCHETYPE_EMBEDDINGS_PATH

Use distance-to-centroid → convert to softmax probabilities.

p_i = exp(-distance_i / τ) / Σ exp(-distance_j / τ)

Add to model_config.py:

@dataclass
class ArchetypeMembershipConfig:
    temperature: float = 0.5
    min_probability_floor: float = 0.05
Apply Soft Membership To:
Component	v2.6	v2.7
Neutralization	subtract mean of 1 archetype	subtract weighted mean across archetypes
RUE optimal vector	1 archetype mean	weighted archetype mean
Elevation	compare to 1 baseline	compare to weighted baseline
Portability transfer	binary	probabilistic

This removes boundary instability entirely.

2️⃣ Upgrade Neutralization from Mean-Centering → Full Conditional Standardization

(Fixes Issue #1 — scale bias)

Problem

Current:

Neutralized_z = Observed_z - E[z | archetype]

This removes location bias but not scale differences.

✅ v2.7 Solution: Conditional Z Residuals

New formula:

z_conditional =
(Observed - μ_archetype) / σ_archetype

Then optionally re-scale to league std:

z_final = z_conditional * league_std

Add to NeutralizationConfig:

use_full_conditional_standardization: bool = True
rescale_to_league_variance: bool = True

This makes Layer 1C:

“Performance relative to role distribution”
instead of
“Performance above role mean”

Huge improvement.

3️⃣ Replace Heuristic Shrinkage with Empirical Bayes Weighting

(Fixes Issue #3 — shrinkage math not variance-aware)

Problem

Current shrinkage:

a = shrinkage_strength * max(0, 1 - GP/min_gp)

This is deterministic and linear.

✅ v2.7 Solution: Variance-Based Shrinkage

Compute:

a = σ_within² / (σ_within² + σ_between² / n)

Posterior:

x_post = (1 - a) * x + a * μ_prior

Add to config:

@dataclass
class BayesianShrinkageConfig:
    use_empirical_bayes: bool = True
    estimate_variance_components: bool = True

This makes shrinkage dimension-specific and data-driven.

4️⃣ Fix Defensive Impact Asymmetry

(Fixes Issue #2 — defense not treated symmetrically)

Currently:

Defensive impact uses position-z

No archetype conditioning

No variance conditioning

This gives defensive archetypes structural advantage.

✅ v2.7 Solution

For defensive dimensions:

Apply soft archetype-weighted conditional standardization

Use full conditional residuals (same as offense)

Remove exemption logic

Remove from skip_neutralization.

Defense must obey same statistical logic as offense.

5️⃣ Fix Dimension Model Variance Compression

(Fixes Issue #4 — composite std too low)

dimension_model_z std = 0.247 → too compressed.

Likely causes:

Over-neutralization

Strong inter-dimension correlation

Shrinkage stacking

✅ v2.7 Solution

After Layer 1C composite:

Compute observed variance

Re-normalize to target variance (e.g., 0.40–0.45)

Add to config:

target_dimension_model_std: float = 0.42
enforce_target_variance: bool = True

This keeps interpretability and separation strength intact.

6️⃣ Reduce RUE Endogeneity

(Fixes Issue #5 — RAPM feedback loop)

Problem:
Optimal archetype vector derived from top-half RAPM players.

That makes RAPM influence RUE baseline.

✅ v2.7 Solution

Instead of:

Top-half RAPM

Use:

Top-half PTS_z

or better:

Top-half neutralized dimension composite

This removes circular reinforcement.

Add flag:

rue_optimal_source: str = "portable_talent"  # not RAPM
7️⃣ Portability Index Clarification + Structural Tightening

(Fixes Issue #6 — proxy vs real portability clarity)

You already improved this in v2.6.

For v2.7:

Ensure portability uses conditional-residual dimensions

Remove any dependency on raw archetype label

Use entropy of soft membership for transfer stability

Add:

use_soft_membership_in_transfer: bool = True
📊 v2.7 Change Summary Table
Priority	Fix	Structural Impact
1	Soft archetype membership	Removes boundary distortion
2	Full conditional neutralization	Removes scale bias
3	Empirical Bayes shrinkage	Correct reliability weighting
4	Defensive symmetry	Removes defensive bias
5	Variance restoration	Restores elite separation
6	RUE decoupling	Removes RAPM feedback
7	Portability tightening	Clarifies structural interpretation
🔧 Required Addition to model_config.py

You currently have dimension weights, but you do NOT have:

A centralized master weight tuning vector

Or a global scaling hook

Or automated normalization enforcement

I strongly recommend adding:

@dataclass
class DimensionWeightTuningConfig:
    """
    Centralized dimension importance scaling.
    Allows global tuning without editing individual weight keys.
    """
    global_multiplier: float = 1.0
    auto_normalize: bool = True
    allow_runtime_override: bool = True

This ensures future stability-weighted re-scaling can occur in one place.