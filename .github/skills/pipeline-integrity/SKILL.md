---
name: pipeline-integrity
description: |
  Post-fetch pipeline integrity gate. After any data fetch or update (box scores,
  tracking, PBP, DARKO, salaries), verify that all downstream artifacts are
  consistent, documentation matches actual data state, and the correct re-run
  chain is queued or confirmed complete. Prevents silent stale data in the
  aggregate.
tags:
  - fetch
  - pipeline
  - integrity
  - data-quality
  - downstream
version: "1.0"
last_updated: "2026-05-29"
persona:
  - "Pipeline integrity guardian"
  - "Data freshness auditor"
preferred_tools:
  - run_in_terminal
  - read_file
  - grep_search
  - file_search
  - get_errors
avoid_tools:
  - create_new_workspace
when_to_use:
  - "After any fetch script runs or completes (box scores, tracking, PBP, DARKO, salaries, Kalshi)."
  - "After backfill scripts run (backfill_tracking_missing.py, fetch_2025_26_game_logs.py, etc.)."
  - "After model_config.py SEASONS list is changed to add or remove a season."
  - "Any time the user says 'update data', 're-fetch', 'backfill', or 'refresh'."
  - "When the aggregate or CLV report is suspected stale."
example_prompts:
  - "I just ran fetch_box_scores_complete.py — make sure everything downstream is still good."
  - "Backfill is done. Check that the tracking files look right and re-run what's needed."
  - "We fetched 2025-26 PBP. What do we need to re-run?"
  - "Data pipeline health check."
log_usage: true
usage_log: loop/skill_usage.log
---
# Pipeline Integrity Skill

This skill is a post-fetch gate. Run it whenever data is added or updated. Its job is to verify artifact completeness, identify the minimal re-run chain, and ensure docs are honest about what data exists.

---

## Step 1 — Artifact Completeness Check

Run this probe immediately after any fetch to know the current state:

```python
import pandas as pd, glob, os

# 1. Tracking files: expect 36 per season (28 base + 8 backfilled measures)
TRACKING_DIR = "data/tracking"
seasons = sorted(os.listdir(TRACKING_DIR)) if os.path.isdir(TRACKING_DIR) else []
print("=== TRACKING FILES ===")
for s in seasons:
    files = glob.glob(f"{TRACKING_DIR}/{s}/*.parquet")
    status = "✓" if len(files) >= 36 else f"⚠ only {len(files)}"
    print(f"  {s}: {status}")

# 2. Box scores: expect one per season
print("\n=== BOX SCORES ===")
for f in sorted(glob.glob("data/historical/player_box_scores_*.parquet")):
    df = pd.read_parquet(f, columns=["game_date"])
    print(f"  {os.path.basename(f)}: {len(df):,} rows")

# 3. PBP normalized files
print("\n=== PBP NORMALIZED ===")
for f in sorted(glob.glob("data/historical/pbp_normalized_*.parquet")):
    df = pd.read_parquet(f, columns=["game_id"])
    print(f"  {os.path.basename(f)}: {df['game_id'].nunique()} games")

# 4. Aggregate seasons + null rate
print("\n=== AGGREGATE ===")
if os.path.exists("aggregate/player_profile_aggregate.parquet"):
    agg = pd.read_parquet("aggregate/player_profile_aggregate.parquet")
    seasons_agg = agg["season"].value_counts().sort_index()
    null_pct = agg.groupby("season").apply(lambda g: (g.isnull().sum() / len(g) * 100).mean())
    print(f"  Seasons: {seasons_agg.to_dict()}")
    print(f"  Mean null % by season:\n{null_pct.round(1)}")
    print(f"  Total cols: {len(agg.columns)}")
else:
    print("  MISSING — build_profile_aggregate.py has not run")

# 5. RAPM seasons
if os.path.exists("data/processed/player_rapm.parquet"):
    rapm = pd.read_parquet("data/processed/player_rapm.parquet", columns=["season"])
    print(f"\n=== RAPM SEASONS ===\n  {sorted(rapm['season'].unique())}")

# 6. YTD ratings — expect 2018-26 (2017-18 excluded: no prior-season BKE projection)
ytd = "data/processed/forecast/team_ratings_ytd.parquet"
if os.path.exists(ytd):
    df_ytd = pd.read_parquet(ytd, columns=["season"])
    ytd_seasons = sorted(df_ytd["season"].unique())
    expected_ytd = {"2018-19","2019-20","2020-21","2021-22","2022-23","2023-24","2024-25","2025-26"}
    missing_ytd = expected_ytd - set(ytd_seasons)
    status = "✓" if not missing_ytd else f"⚠ missing: {missing_ytd}"
    print(f"\n=== YTD RATINGS === {status}\n  {ytd_seasons}")
else:
    print("\n=== YTD RATINGS === MISSING — run build_ytd_team_ratings.py")

# 7. Rest features — expect 2017-26
rest = "data/processed/forecast/game_rest_features.parquet"
if os.path.exists(rest):
    df_rest = pd.read_parquet(rest, columns=["season"])
    rest_seasons = sorted(df_rest["season"].unique())
    print(f"\n=== REST FEATURES ===\n  {rest_seasons}")
else:
    print("\n=== REST FEATURES === MISSING — run build_rest_features.py")
```

---

## Step 2 — Identify the Minimal Re-Run Chain

**The full pipeline always ends with `build_rest_features` → `build_ytd_team_ratings` → `forecast_2025_26_games`. Never skip these terminal steps — they produce the game-level training data Step 5 (GBDT) depends on.**

Use this decision table to determine the *start* of the chain. Run everything from that point to the end.

| What changed | Start here | Then run to end |
|---|---|---|
| Box scores only | `summarize_team_logs` | → rest → YTD → CLV |
| Tracking files (new season or backfill) | BKE layers 1-4 | → decomp → shrinkage → pts_v40 → aggregate → rest → YTD → CLV |
| PBP added/completed for a season | `run_normalization` | → lineups → possessions → RAPM → *(BKE chain above)* |
| DARKO updated | `build_profile_aggregate` | → rest → YTD → CLV |
| Salaries updated | `build_profile_aggregate` | → rest → YTD → CLV |
| model_config.py SEASONS changed | Full pipeline | `scripts/run_full_downstream.sh` |
| Kalshi closing lines updated | `forecast_2025_26_games` | (terminal — just re-run CLV) |

**Terminal steps (always run at the end of any pipeline):**
```
build_rest_features      — game-level rest/B2B/3-in-4 for all seasons (Step 5 training input)
build_ytd_team_ratings   — per-game blended mu for 2018-26 (Step 4 + Step 5 training input)
forecast_2025_26_games   — CLV vs Kalshi closing lines
```

**YTD coverage note:** `build_ytd_team_ratings` covers seasons 2018-26 (all seasons with projected team features). 2017-18 is excluded — projecting 2017-18 would require 2016-17 BKE scores which are not in the pipeline. This is the correct design; do not attempt to add 2017-18 YTD.

For a full re-run after a new season is added, use:
```bash
bash scripts/run_full_downstream.sh &> logs/downstream.log &
```

For the targeted 2025-26 RAPM + BKE chain only:
```bash
bash scripts/rerun_2025_26_rapm.sh &> logs/rerun_2025_26.log &
```

---

## Step 3 — Documentation Sync

After confirming artifacts are up to date, update these docs if the data state changed:

| Doc | What to update |
|---|---|
| `docs/reference/tracking_availability.md` | Season × measure matrix; mark any newly backfilled measures |
| `loop/in_progress_context.txt` | Replace with current task + pipeline status |
| `docs/integration/for-alpha-thesis.md` | Update if Brier or CLV results changed materially (>0.002) |

**Rule:** If the aggregate's season list or null rate changes, `docs/reference/tracking_availability.md` and `loop/in_progress_context.txt` must both be updated in the same session.

---

## Step 4 — Column Coverage Sanity

A healthy aggregate has mean null % ≤ 20% for 2022-26 seasons. Pre-2022 seasons may be higher (15–40%) due to missing synergy/tracking measures before backfill.

Flag any season where null % is > 30 ppt worse than the prior season — that signals a new measure added to the fetch layer that hasn't backfilled yet.

```python
# Quick null-rate delta check
import pandas as pd
agg = pd.read_parquet("aggregate/player_profile_aggregate.parquet")
null_pct = agg.groupby("season").apply(lambda g: (g.isnull().sum() / len(g) * 100).mean()).sort_index()
deltas = null_pct.diff()
flagged = deltas[deltas.abs() > 30]
if not flagged.empty:
    print("⚠ Null-rate jumps > 30ppt between seasons:")
    print(flagged)
else:
    print("✓ No null-rate anomalies")
```

---

## Step 5 — CLV Freshness Gate

If box scores or forecasts were updated, re-run the CLV to confirm the delta:

```bash
python3 scripts/forecast_2025_26_games.py
```

Check `reports/kalshi_clv_2025_26.json`:
- `brier_bke` — target ≤ 0.235 walk-forward, ≤ 0.210 alpha-ready
- `mean_clv` — target > 0.02 for alpha-ready; currently −0.069 (data staleness, not model quality)
- `n_games` — should be 1,219 for full 2025-26 season

---

## Non-Scope

- Do not re-fetch data in this skill — use `audit-fetch` for that.
- Do not modify model weights or RAPM hyperparameters — use `audit-model`.
- Do not change the aggregate schema — use `audit-aggregate`.
- Do not run the full pipeline speculatively; only queue what the decision table requires.
