---
name: forecast-leakage-audit
description: |
  Contamination and temporal-leakage audit for any game-level / forecast
  probability, Brier, or CLV comparison. Fires before trusting an out-of-sample
  result, when one model surprisingly beats or loses to another, or before
  promoting a forecast model or measuring closing-line value. Distinct from
  data-audit (schema), audit-compute (formulas), and audit-model (RAPM/BKE
  internals): this skill checks WHEN information enters a prediction.
tags:
  - audit
  - leakage
  - forecast
  - brier
  - clv
  - out-of-sample
version: "1.0"
last_updated: "2026-05-30"
persona:
  - "Out-of-sample integrity auditor"
  - "Temporal-leakage and train/test contamination reviewer"
preferred_tools:
  - read_file
  - grep_search
  - run_in_terminal
avoid_tools:
  - create_new_workspace
when_to_use:
  - "Before trusting any walk-forward / OOS Brier, log-loss, or accuracy number."
  - "When a model surprisingly outperforms or underperforms another (rule out leakage vs dilution)."
  - "Before promoting a forecast/game model or running a CLV test vs market lines."
  - "After changing forecast features, YTD blending, schedule, or rating joins."
example_prompts:
  - "Make sure our Brier result isn't from leaked or contaminated data."
  - "Is Elo beating BKE because of a leak, or is BKE's signal diluted?"
  - "Audit the game-level forecast for train/test contamination before the CLV run."
log_usage: true
usage_log: loop/skill_usage.log
---
# Forecast / Game-Model Leakage & Contamination Audit

The project's alpha thesis depends entirely on **leakage-free, out-of-sample**
validation. A single contaminated or diluted signal invalidates every Brier and
CLV number downstream. This skill exists to make that failure mode impossible to
miss.

**Primary tool:** `scripts/audit_forecast_leakage.py` (run it; expect 12/12 pass).

## Audit both directions — never just one

A surprising result has two innocent-looking explanations. Always check both:

- **Is the strong model leaking?** (its strength is fake)
- **Is the weak model contaminated/diluted?** (its weakness is fake — a join
  bug, a label swap, a broken lag, or signal compression)

Only after both are ruled out is a comparison trustworthy.

## The six check families

1. **Structural** — no duplicate game_ids, no null features (rules out
   double-counting dilution).
2. **Outcome correctness** — predicted label cross-checks 100% against the
   official source (W/L); per-season base rates are sane.
3. **Lag integrity** — any "to-date" feature (YTD margin, rolling form) equals a
   **strictly-prior** expanding/rolling aggregate (`shift(1)` before cumulate).
   Reproduce derived ratings from their formula to byte tolerance.
4. **Future-blindness** — recompute the sequential model with later games
   removed; earlier predictions must be **identical** (max Δp = 0). Add a
   **placebo**: shuffling within-period order should *degrade* a legitimately
   temporal model.
5. **Signal reality** — correlate the suspected-weak rating against realized
   outcomes (e.g., preseason rating ↔ actual season margin). A positive,
   season-stable r means the metric is real but *mis-deployed* (dilution), not
   noise.
6. **Calibration** — confirm a weak model's loss is explained by
   under/over-confidence (probability spread) and unit/σ mismatch, not by data.

## Leakage-safety reasoning notes

- **Walk-forward**: train strictly on periods < target.
- **Per-period feature scaling** (e.g., z-scoring by season) is transductive and
  acceptable **only if it uses no outcomes**; when in doubt, test a causal
  expanding-window variant and confirm the metric barely moves.
- A "to-date" feature that includes the current event is the most common leak —
  always verify the `shift(1)`.

## When NOT to use

- Schema drift / dtype issues → `data-audit`.
- Formula/normalization correctness in compute stage → `audit-compute`.
- RAPM/BKE modeling internals and backtests → `audit-model`.
- Downstream artifact regeneration after a fetch → `pipeline-integrity`.

## Minimum output

Report: checks run, pass/fail per family, which direction(s) were ruled out, and
an explicit "trustworthy / not trustworthy" verdict with the residual caveats
(e.g., correlation limited to seasons where `margin` exists).
