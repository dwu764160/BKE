# BKE v2.6 System Audit — Statistical + Basketball Assessment
*Generated after v2.6 implementation (Bayesian shrinkage, L4 fix, backtesting)*

---

## 1. System Overview

| Metric              | Value        |
|---------------------|-------------|
| Qualified players   | 950         |
| Seasons             | 3 (2022-25) |
| Columns             | 344         |
| Dimensions          | 9           |
| Archetypes          | 11          |
| Runtime             | ~12s        |

**Pipeline**: L1 (Portable Talent, 45%) → L2 (Role Utilization, 20%) → L3 (Archetype Elevation, 20%) → L4 (Scheme Stability, 15% bonus-only) → Total Impact

**L1 Composition**: 25% RAPM backbone + 20% playtype efficiency + 55% 9-dimension model

---

## 2. Statistical Assessment

### 2.1 Backtest Results (Combined 2022-24 → 2024-25, n=263 returning)

| Metric                          | Value   | Assessment     |
|---------------------------------|---------|----------------|
| PTS Spearman ρ                  | +0.553  | Moderate-Good  |
| TI Spearman ρ                   | +0.507  | Moderate       |
| TI Exact Tier Accuracy          | 37.3%   | OK (5 tiers)   |
| TI Within-1-Tier Accuracy       | 76.0%   | Good           |
| PTS Within-1-Tier Accuracy      | 80.6%   | Strong         |
| TI z-score RMSE                 | 0.436   | Moderate       |
| PTS z-score RMSE                | 0.379   | Good           |
| Offensive Archetype Stability   | 53.6%   | Fair           |
| Defensive Archetype Stability   | 53.6%   | Fair           |
| RUE Spearman ρ                  | +0.433  | Moderate       |
| Portability Spearman ρ          | +0.384  | Weak-Moderate  |

**Interpretation**: PTS (Portable Talent Score) is the most stable and predictive
metric, which validates the design — talent is more transferable year-to-year than
total impact (which includes role-dependent components that shift when players change
teams, roles, or teammates). Within-1-tier accuracy of 76-81% means the system
correctly predicts a player's approximate tier 3 out of 4 times.

### 2.2 Dimension Stability (Year-over-Year Spearman ρ)

| Dimension                   | ρ      | Assessment                 |
|-----------------------------|--------|----------------------------|
| dim_driving_gravity_z       | +0.840 | Very Strong (physical)     |
| dim_defensive_playmaking_z  | +0.750 | Strong (instinct-based)    |
| dim_extra_possession_z      | +0.575 | Moderate (rebounding+)     |
| dim_turnover_control_z      | +0.572 | Moderate (handles)         |
| dim_defensive_versatility_z | +0.561 | Moderate                   |
| dim_shooting_gravity_z      | +0.515 | Moderate (shot profile)    |
| dim_defensive_impact_z      | +0.497 | Moderate                   |
| dim_self_creation_z         | +0.474 | Moderate                   |
| dim_playmaking_creation_z   | +0.297 | Weak ⚠️                   |

**Key Finding**: Playmaking creation is the least stable dimension (ρ=0.297).
This suggests the dimension either measures something volatile (assists are
context-dependent) or the component metrics have noise. Since playmaking is
55% of the dimension model's playmaking weight, this instability propagates
into PTS. Consider: (a) weighting playmaking creation lower, (b) adding
stabilizing metrics like passing efficiency, or (c) using multi-season smoothing.

### 2.3 Bayesian Shrinkage Calibration

Post-fix shrinkage is now mild and well-calibrated:
- Shooting std: 0.522 → 0.493 (5.5% reduction)
- Playmaking std: 0.370 → 0.351 (5.1% reduction)  
- Driving std: 0.676 → 0.650 (3.8% reduction)

Shrinkage preserves nearly all variance while pulling extreme values toward
group means. Pre-post correlations remain >0.93.

### 2.4 Distribution Health

| Score             | Mean  | Std   | Min    | Max   |
|-------------------|-------|-------|--------|-------|
| total_impact_z    | 0.06  | 0.48  | -1.48  | 1.25  |
| portable_talent_z | 0.00  | 0.44  | -1.48  | 1.26  |
| portability_index | 0.60  | 0.08  | raw    | raw   |

Slightly positive mean TI (0.06) reflects the qualified cutoff filtering out
the worst players. Good separation between top and bottom (~2.7σ range).

---

## 3. Basketball Assessment (Emphasis)

### 3.1 Star Rankings — 2024-25

| Rank | Player                    | TI%   | PTS%  | RAPM  | Archetype              |
|------|---------------------------|-------|-------|-------|------------------------|
| 1    | Shai Gilgeous-Alexander   | 100.0 | 99.7  | 9.83  | Ball Dominant Creator  |
| 2    | Giannis Antetokounmpo     | 99.7  | 100.0 | ~13+  | Ball Dominant Creator  |
| 3    | **Daniel Gafford**        | 99.4  | 98.8  | ~6+   | PnR Rolling Big ⚠️    |
| 4    | Nikola Jokić              | 99.1  | 99.4  | ~11+  | Ball Dominant Creator  |
| 5    | **Payton Pritchard**      | 98.8  | 81.6  | ~0.5  | Perimeter Scorer ⚠️   |
| 6    | Kevin Durant              | 98.5  | 95.1  |       | All-Around Scorer      |
| 7    | Jimmy Butler III          | 98.2  | —     |       | All-Around Scorer      |
| 20   | Luka Dončić               | 94.2  |       |       | Ball Dominant Creator  |
| 24   | Anthony Edwards           | 92.9  |       |       | All-Around Scorer      |
| 29   | Stephen Curry             | 91.4  |       |       | Ball Dominant Creator  |

**What's Right** ✅:
- Top 4 features SGA, Giannis, Jokic — consensus top-3 NBA players
- Kevin Durant, Jimmy Butler consistently top-10
- The system successfully identifies that stars have high portable talent
  AND high archetype elevation (they're exceptional relative to anyone)
- Bottom 5 are legitimately bad/limited players (Elfrid Payton, KJ Simpson)
- Archetype assignments mostly make basketball sense (SGA=BDC, Giannis=BDC,
  Jokic=BDC, Curry=BDC, Edwards=All-Around)

**What's Wrong** ❌:
1. **Daniel Gafford at #3**: A rim-running center who catches lobs and
   blocks shots should NOT rank above Jokic. His RAPM is legitimately
   high (~6+ based on lineup data) and his PnR Rolling Big archetype has
   a low baseline, giving massive elevation. The system over-rewards
   "being good at a simple role" relative to "carrying a franchise."

2. **Payton Pritchard at #5**: An excellent bench guard, but #5 in the
   NBA is absurd. He benefits from very high RUE (1.736 — highest in the
   league) because his simple perimeter shooting role has near-perfect
   role alignment, plus high scheme stability (bench players are inherently
   more "portable"). His PTS is only 81.6, showing L1 correctly identifies
   him as mid-tier talent. L2 + L3 inflate him.

3. **Luka Dončić at #20**: A consensus top-5 player should not be 20th.
   His L1=0.625 is moderate (9-dim model gives him mixed scores across
   dimensions), and his elevation L3=0.702 is modest (BDCs have high
   baselines, so outperforming by a small margin gives small elevation).

4. **Curry at #29**: Stephen Curry being 29th in 2024-25 is too low.
   His L1=0.617 reflects moderate RAPM (1.344) and excellent shooting (1.105)
   but low scores on other dimensions.

5. **LeBron at 28.5%**: LeBron's RAPM in 2024-25 was -5.16, which
   dramatically tanks his score. This could be a data issue or reflect
   genuinely poor team performance (Lakers' struggles). Either way, it
   shows how RAPM-dependent the system is for separating top from bottom.

### 3.2 Root Cause Analysis

**Problem 1: Layer 3 Elevation Inflation**
The elevation formula rewards players proportional to how much they exceed
their archetype's average. Archetypes with low averages (PnR Rolling Big,
Off-Ball Shooters) give massive elevation for modest performance, while
high-average archetypes (Ball Dominant Creator) require exceptional
performance for similar elevation. This is WAR's "replacement level"
problem — the replacement level for a PnR rolling big is very low,
so any competent one looks amazing.

Fix options:
- Cap elevation contribution per archetype (e.g., max 1.5σ)
- Weight elevation by archetype difficulty/rarity
- Use within-season percentile of elevation rather than raw z-score
- Reduce L3 weight from 20% to 10%

**Problem 2: RUE Favors Simple Roles**
Role Utilization Efficiency measures how efficiently a player executes
their role. Players with simple, well-defined roles (spot-up shooter,
rim runner) can achieve near-perfect role alignment easily. Complex
roles (primary creator, all-around scorer) have higher variance in
execution. The 40% cosine similarity weight still penalizes players
whose actual distribution doesn't match their archetype template.

Fix options:
- Weight RUE by role complexity (harder roles get scaling bonus)
- Use absolute efficiency rather than role-relative efficiency
- Reduce cosine similarity weight further (from 40% to 20%)

**Problem 3: RAPM Noise in Small Samples**
RAPM for players on bad teams or with limited lineup variety can be
noisy. LeBron's -5.16 RAPM in 2024-25 is an example — it reflects
the Lakers' overall performance more than LeBron's individual talent.
The Bayesian shrinkage helps but targets dimensions, not RAPM itself.

Fix options:
- Apply Bayesian shrinkage to RAPM directly (not just dimensions)
- Cap RAPM's negative contribution to L1
- Multi-season RAPM smoothing (already exists but may need tuning)

### 3.3 Per-Season Top-5 Assessment

**2024-25**: SGA ✅, Giannis ✅, Gafford ❌, Jokic ✅, Pritchard ⚠️
**2023-24**: Paul George ⚠️, Kawhi ⚠️, Butler ⚠️, Derrick White ⚠️, Danté Exum ❌
**2022-23**: KD ✅, Kennard ❌, Markkanen ⚠️, Reaves ⚠️, Curry ✅

Pattern: 2-3 legit stars + 2-3 inflated role players per season in top 5.
The system consistently identifies superstars but struggles to properly
separate tier-2 stars from elite role players.

### 3.4 Archetype Assessment

| Archetype                     | Count | Basketball Validity           |
|-------------------------------|-------|-------------------------------|
| Off-Ball Stationary Shooter   | 128   | ✅ Largest group (3&D wings)  |
| Ball Dominant Creator         | 117   | ✅ Guards + Jokic/Giannis     |
| All-Around Scorer             | 106   | ✅ Versatile wings             |
| PnR Rolling Big               | 98    | ✅ Traditional bigs            |
| Ballhandler                   | 92    | ✅ Secondary creators          |
| Connector                     | 89    | ✅ Glue guys                   |
| Off-Ball Movement Shooter     | 76    | ✅ Klay Thompson types         |
| Off-Ball Finisher             | 75    | ✅ Cutters/putback specialists |
| Perimeter Scorer              | 58    | ✅ High-volume guards          |
| Interior Scorer               | 54    | ✅ Post players                |
| PnR Popping Big               | 54    | ✅ Stretch bigs                |

Distribution is reasonable. No single archetype dominates excessively.
The 11 archetypes cover the modern NBA role taxonomy well.

### 3.5 Portability Assessment

Top 10 Most Portable: Aaron Gordon, Jamal Murray, Kyrie Irving, Evan Mobley,
Kawhi Leonard, Jayson Tatum, Amen Thompson, Donovan Mitchell, Jimmy Butler,
Pascal Siakam.

**Assessment**: This is a strong list. These are all players who would be
valuable on any team — two-way players with diverse skills and no dependency
on a specific offensive system. Giannis at 0.10 portability (lowest) makes
basketball sense — his game is uniquely built around his physical tools and
a specific offensive structure.

---

## 4. Summary Scorecard

| Category                    | Grade | Notes                                     |
|-----------------------------|-------|-------------------------------------------|
| Star identification         | B+    | Top 3-4 correct, but role player noise    |
| Skill dimension model       | B     | 9 dims reasonable, playmaking unstable    |
| RAPM integration            | B+    | Anchors stars correctly, noisy for some   |
| Archetype system            | A-    | Good taxonomy, stable assignments         |
| Portability index           | A     | Excellent face validity                   |
| Role utilization (RUE)      | C+    | Favors simple roles, penalizes unique     |
| Archetype elevation (L3)    | C     | Inflates role players excessively         |
| Scheme stability (L4)       | B-    | Fixed (bonus-only), still proxy-based     |
| Bayesian shrinkage          | A-    | Well-calibrated after fix                 |
| Year-over-year prediction   | B     | ρ=0.55 PTS, 76% within-1-tier TI         |
| Overall basketball validity | B-    | Good framework, execution needs tuning    |

---

## 5. Priority Recommendations (Ordered)

1. **Cap Layer 3 elevation per archetype** — Prevent role player inflation.
   Max elevation_z of 1.5σ for any archetype, or weight by archetype rarity.

2. **Add role complexity weighting to RUE** — Ball Dominant Creators and
   All-Around Scorers should get RUE scaling bonus for executing harder roles.

3. **Stabilize playmaking creation dimension** — Year-over-year ρ=0.297 is
   too low. Consider multi-season smoothing or adding stabilizing metrics.

4. **Apply RAPM shrinkage** — Bayesian shrinkage for RAPM itself (before
   feeding into L1), not just dimensions. Would help LeBron-type cases.

5. **Reduce L3 weight** — From 20% to 12-15%. Elevation is informative
   but overweighted relative to its signal quality.

6. **Incorporate unused tracking data** — Defense 2Pointers, LessThan10Ft,
   GreaterThan15Ft files + hustle stats (contested shots, box outs) could
   feed defensive impact and defensive versatility dimensions.

---

*Audit completed. System is functional and produces reasonable outcomes for
the majority of players. The primary weakness is Layer 3 elevation inflating
role players into star territory. This is an architectural issue that requires
structural changes to the elevation formula, not just weight tuning.*
