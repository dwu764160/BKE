# Step 1 pre-flight — Cross-team archetype interaction: SIGNAL CONFIRMED

**Date:** 2026-05-30
**Code:** `scripts/test_archetype_interactions.py`
**Report:** `reports/archetype_interaction_signal_test.json`
**Question:** Before building anything, does attacker-archetype × defender-archetype
matchup carry REAL signal *beyond player talent*? (The plan's gate #1.)

---

## 0. TL;DR

**Yes — unambiguous, talent-purged, basketball-coherent signal.** Confirmed on
two runs: the original 3-season pre-flight (2022-23…2024-25; 361k rows) and the
**locked 8-season run (2017-18…2024-25; 810k rows, 9.6M partial possessions)**
using **two-way player fixed effects** to strip both players' average levels.
**8-season result: 71 of 99 archetype-pair cells significant at 95% CI (≈5 by
chance); 68 survive Benjamini-Hochberg FDR.** The motivating hypotheses hold with
tighter CIs on larger N:

- **POA Defender vs Ball Dominant Creator: −0.031 ppp** [−0.042, −0.019], 166k
  poss — real matchup suppression on top of talent. Wing Stopper vs Creator −0.028.
  (3-season: −0.038; slight attenuation adding pre-2022 data, still highly
  significant.)
- **Rim protection vs rollers remains the strongest cell:**
  PnR Rolling Big vs Rim Protector **−0.078** [−0.083, −0.073], 172k poss —
  talent was *masking* this one; the cell grew stronger with larger N and tighter
  CI. Same direction for Dropping Big (−0.069) and Mobile Big (−0.069).

**Verdict: GO.** This is a different result from the repo's within-team
null (`INTERACTION_MATRIX = {}`); cross-team attacker-vs-defender is real.

## 1. Method (talent washout)

`ppp = PLAYER_PTS / PARTIAL_POSS` per matchup row, filtered ≥2 partial poss.
We remove offensive-player and defensive-player fixed effects by **weighted
alternating-projection demeaning** (12 passes; residual weighted mean ≈ 1e-18).
The residual, aggregated to (off_arch × def_arch) cells, is the matchup delta
purged of both players' opponent-averaged talent — exactly the **double-counting
guard** the design requires (PTS/RAPM already carry talent; we add only the
deviation). Significance = **cluster bootstrap over offensive players** (2000
resamples) → 95% CI per cell + BH-FDR.

Talent removal shrinks the cross-cell spread modestly: FE-adjusted std
**0.035 ppp** vs naive 0.045 (8-season; washout ratio 0.787). Most structure is
genuine matchup interaction, not talent confound. (3-season was 0.044 vs 0.055;
the slight compression adding pre-2022 seasons reflects era differences in scheme
and archetype composition, not signal degradation.)

## 2. The dominant axis is SIZE/POSITION — important for Step 1

The largest cells are **cross-position**: guards/wings (Off-Ball Movement
Shooter, All-Around Scorer, Ball Dominant Creator) **boost** vs big-man defenders
(Rim Protector, Dropping/Mobile Big) by **+0.14 to +0.19 ppp**. That is real
switch-hunting basketball (attack the big in space), but those pairings only
occur on **switches/mismatches**, not standard assignments.

⇒ **The interaction model must respect lineup matchup *assignment*.** Most
possessions are same-position-band (guard guards guard). The *assignable* signal
that a position-band matchup step will actually invoke is:

8-season locked values (2017-18…2024-25; replace 3-season figures above):

| Assignable matchup (same band) | FE-adj interaction | poss |
|---|---|---|
| PnR Rolling Big vs Rim Protector | **−0.078** | 172k |
| PnR Rolling Big vs Dropping Big | **−0.069** | 219k |
| PnR Rolling Big vs Mobile Big | **−0.069** | 20k |
| PnR Popping Big vs Rim Protector | −0.021 (n.s. 8-season) | 45k |
| Interior Scorer vs Dropping Big | **−0.050** | 34k |
| Ball Dominant Creator vs POA Defender | **−0.031** | 166k |
| All-Around Scorer vs POA Defender | **−0.021** | 89k |

The cross-position boost cells become the **switch term** the possession engine
can use when a mismatch is forced — kept, but gated behind a switch/assignment
flag so they don't fire on every possession.

## 3. Cautions / caveats

- **Versatile Defender shows a POSITIVE interaction with creators** (+0.057):
  likely a labeling/assignment artifact (the "versatile" tag lands on switch-bigs
  who end up on creators in space). Treat positive-vs-creator cells skeptically;
  shrink hard and inspect the underlying player lists before trusting.
- **PPP only captures the guarded scorer's own points**, not playmaking spillover
  (a POA that forces a pass shows up as suppression of the creator but not the
  teammate's resulting shot). The team-level adjustment will understate
  playmaking disruption — fine for a first cut, note for later.
- **Matchup data is now 8 seasons** (2017-18…2024-25; 2025-26 profiles not yet
  generated). The matrix is locked on the 8-season run. Walk-forward validation
  across seasons is feasible.
- **Magnitudes are modest per possession.** A creator sees a POA ~30 partial
  poss/game → ≈ −1.1 pts; a roller vs rim protector ~20 poss → ≈ −1.8 pts. Summed
  across a lineup the team `interaction_adj` is a few points — meaningful for
  **props and possession realism**, likely **marginal for game-level Brier**
  (consistent with prior findings). Judge Step 1 on BOTH, not game Brier alone.

## 4. Implication for the Step 1 build

Proceed to build the interaction matrix (`docs/plans/cross_team_interaction_model.md`),
with two design locks this test forces:
1. **Shrink per cell by sample (empirical-Bayes)** and **mean-zero each
   offensive archetype over defender archetypes** — already specified; this test
   confirms it's necessary (positive artifacts exist).
2. **Apply via position-band matchup assignment**, with a separate **switch/
   mismatch term** for the cross-position cells — do NOT apply the full matrix to
   every off×def pair indiscriminately.
