# BKE Gen 1 Summary (through 2026-03-01)

## Objective
BKE Gen 1 established a stable, reproducible possession-level player-impact pipeline that separates:
- **Role assignment (archetype behavior)**
- **Impact/value estimation (RAPM/ORAPM/DRAPM and BKE composites)**

## Core Gen 1 Strides
- End-to-end deterministic pipeline from fetch → normalize → features → compute → modeling → viewers.
- Offensive/defensive archetype stack matured with stricter defensive rarity and confidence calibration.
- BKE decomposition evolved from baseline outputs into a multi-layer framework with diagnostics and backtesting.
- v3.1 split policy operationalized as persistent dual outputs:
  - `60/40` (stability anchor)
  - `55/45` (predictive companion)

## Key v3.x Outcomes
- **Layer experimentation framework** (`bke_v31_experimental_layers`): each layer tested independently, then combined under guardrails.
- **Layer 3 + Layer 6 second-pass** passed additive gates and was promoted to output artifacts.
- **Viewer parity**: BKE version toggle + split toggle now coexist for analysis UX.

## Defensive Archetype Confidence Progress
- Confidence became more structurally balanced rather than over-dominated by top-score magnitude.
- Margin-sensitive confidence behavior now better differentiates strong single-role fits vs noisy mixed-signal profiles.
- Resulting downstream effect in v3.1 context:
  - improved cross-season defensive rank stability signals
  - lower DBKE volatility/churn
  - small expected tradeoff in one-step predictive correlation

## Experiment 2 (Production-Tilt) Evolution
- Initial sweep used compact production proxies.
- Rerun expanded proxy to include:
  - ORAPM, TS_PCT
  - raw offensive box-score stats (PTS, AST, FGM, FGA, FG3M, FG3A, FTM, FTA)
- Purpose: reduce low-production over-ranking while preserving minimal rank disruption.

## Artifact Footprint (Gen 1)
- Primary modeling/decomposition artifacts: `data/processed/bke/`
- Diagnostic and experiment reports: `reports/`
- Viewer exports: `app/`

## Next Phase Hand-off
Gen 1 closes with stable architecture and diagnostics. The next major phase is the **player evaluation engine**, scaffolded under:
- `src/player_eval/`
