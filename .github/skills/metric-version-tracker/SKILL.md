---
name: metric-version-tracker
description: |
  Ensures every produced metric, artifact, and pipeline script references the
  canonical latest version defined in docs/multiple-versions.md. Fires whenever
  any versioned metric or artifact is touched, read, or produced (BKE, PTS,
  archetypes, RAPM, team projections). Updates docs/multiple-versions.md when
  a version is promoted, deprecated, or newly created.
tags:
  - versioning
  - metrics
  - bke
  - pts
  - archetypes
  - pipeline-quality
version: "1.0"
last_updated: "2026-05-29"
persona:
  - "Metric version guardian"
  - "Artifact lineage auditor"
preferred_tools:
  - read_file
  - grep_search
  - file_search
  - run_in_terminal
  - get_errors
avoid_tools:
  - create_new_workspace
when_to_use:
  - "When any script touching a versioned artifact is read or modified (BKE, PTS, archetypes, RAPM, team projections)."
  - "When decomposition_engine.py, construct_bke_scores_v27.py, build_pts_v40.py, compute_player_archetypes.py, compute_defensive_archetypes_v2.py, or model_config.py are touched."
  - "After any pipeline run that produces versioned outputs."
  - "When the user mentions bke, pts, archetype, scoring, decomp, v27, v28, v32, v40, or any artifact with a version number."
  - "When a new version of any metric is being developed, swept, or promoted to production."
  - "When checking if all seasons use the same version of a metric."
example_prompts:
  - "Make sure all seasons use the latest PTS version."
  - "Check that the archetypes in the aggregate are current."
  - "Is the BKE decomp using v2.8 or v2.7?"
  - "Promote PTS v4.1 to production."
  - "What version of BKE is the walk-forward using?"
log_usage: true
usage_log: loop/skill_usage.log
---
# Metric Version Tracker Skill

The ground truth for all versioned artifacts is **`docs/multiple-versions.md`**.
Always read that document first. This skill provides the process for checking,
enforcing, and updating versions.

---

## Step 1 — Read the Ground Truth Registry

```bash
cat docs/multiple-versions.md
```

The registry defines **CURRENT PRODUCTION** status for every versioned family:

| Family | Current production version | Canonical artifact |
|---|---|---|
| PTS scoring | **v4.0** | `data/processed/bke/pts_v40.parquet` (cols: `pts_o_v40`, `pts_d_v40`) |
| BKE decomposition | **v2.8** | `data/processed/bke/bke_v28_decomposition.parquet` |
| BKE viewer JSON | **v2.7** | `data/processed/bke/BKE_Scores_v27.json` |
| Defensive archetypes | **v3.4** | `data/processed/bke/defensive_archetypes_v2.parquet` |
| Offensive archetypes | **v4.3** | `data/processed/bke/player_archetypes.parquet` |
| Team projections | **current** | `data/processed/forecast/projected_team_features.parquet` |
| YTD blended ratings | **current** | `data/processed/forecast/team_ratings_ytd.parquet` |

**Stale / deprecated versions — never use in production pipeline:**

| Stale artifact | Superseded by | Allowed in |
|---|---|---|
| `pts_v32.parquet` / `pts_o_v32` | pts_v40 | Dev sweep scripts only |
| `bke_v27_decomposition.parquet` | bke_v28 | Legacy comparison only |
| `bke_v30_decomposition.parquet` | — (failed) | Never |
| `bke_v31_components.json` | — | Experimental viewer only |
| `compute_defensive_archetypes.py` (v1) | v2 script | Never |

---

## Step 2 — Artifact Existence Check

Run this probe to confirm all current-version files exist on disk:

```python
import os

CURRENT_ARTIFACTS = {
    "pts_v40":              "data/processed/bke/pts_v40.parquet",
    "pts_v40_a":            "data/processed/bke/pts_v40_a.parquet",
    "pts_v40_c":            "data/processed/bke/pts_v40_c.parquet",
    "bke_v28_decomp":       "data/processed/bke/bke_v28_decomposition.parquet",
    "bke_v27_scores":       "data/processed/bke/BKE_Scores_v27.json",
    "def_archetypes_v2":    "data/processed/defensive_archetypes_v2.parquet",
    "off_archetypes":       "data/processed/player_archetypes.parquet",
    "team_projections":     "data/processed/forecast/projected_team_features.parquet",
    "ytd_ratings":          "data/processed/forecast/team_ratings_ytd.parquet",
    "game_rest_features":   "data/processed/forecast/game_rest_features.parquet",
    "player_rapm":          "data/processed/player_rapm.parquet",
    "player_profile_agg":   "aggregate/player_profile_aggregate.parquet",
}

print("=== VERSION ARTIFACT CHECK ===")
all_ok = True
for name, path in CURRENT_ARTIFACTS.items():
    exists = os.path.exists(path)
    status = "✓" if exists else "✗ MISSING"
    if not exists:
        all_ok = False
    print(f"  {status}  {name}: {path}")

if all_ok:
    print("\n✓ All current-version artifacts present")
else:
    print("\n⚠ Missing artifacts — run the appropriate pipeline step to regenerate")
```

---

## Step 3 — Season Coverage Check

All current-version artifacts must cover the same set of seasons:

```python
import pandas as pd

SEASON_CHECKS = {
    "pts_v40":           ("data/processed/bke/pts_v40.parquet", "season"),
    "player_rapm":       ("data/processed/player_rapm.parquet", "season"),
    "team_projections":  ("data/processed/forecast/projected_team_features.parquet", "season"),
    "ytd_ratings":       ("data/processed/forecast/team_ratings_ytd.parquet", "season"),
    "game_rest_features":("data/processed/forecast/game_rest_features.parquet", "season"),
    "profile_aggregate": ("aggregate/player_profile_aggregate.parquet", "season"),
}

EXPECTED_PLAYER_SEASONS = {"2017-18","2018-19","2019-20","2020-21","2021-22","2022-23","2023-24","2024-25","2025-26"}
EXPECTED_TEAM_SEASONS   = {"2018-19","2019-20","2020-21","2021-22","2022-23","2023-24","2024-25","2025-26"}
# 2017-18 excluded from team projections/YTD (no prior-season BKE projection)

print("=== SEASON COVERAGE CHECK ===")
for name, (path, col) in SEASON_CHECKS.items():
    try:
        df = pd.read_parquet(path, columns=[col])
        seasons = sorted(df[col].unique())
        print(f"  {name}: {seasons}")
    except Exception as e:
        print(f"  {name}: ERROR — {e}")
```

**Expected:**
- `pts_v40`, `player_rapm`, `profile_aggregate`: 2017-26 (9 seasons) after 2025-26 RAPM completes
- `team_projections`, `ytd_ratings`: 2018-26 (8 seasons; 2017-18 has no prior-year projection)
- `game_rest_features`: 2017-26 (9 seasons; rest features available from first season)

If any artifact is missing seasons the pipeline has produced, re-run the appropriate step.

---

## Step 4 — Stale Reference Scan

When modifying a production script, scan it for references to deprecated versions:

```bash
# Stale version patterns that must NOT appear in production pipeline scripts
grep -n "pts_v32\|bke_v27_decomp\|bke_v30\|bke_v31\|compute_defensive_archetypes\.py" \
  scripts/run_full_downstream.sh \
  scripts/rerun_2025_26_rapm.sh \
  src/simulation/validate_forecast.py \
  src/simulation/forecast_2025_26_games.py 2>/dev/null
```

**Allowed stale references** (dev scripts, not in production pipeline):
- `scripts/pts_v40_multiseason.py` — reads `pts_v32` as baseline input (OK)
- `scripts/pts_v40_defense.py` — reads `pts_v32` as sweep input (OK)
- `scripts/validate_lineup_pts_v2.py` — defaults to `pts_v32` path (OK; overrideable)
- `scripts/diagnose_brier_baseline.py` — imports `build_pts_v32` (OK)

**Not allowed:** any of the above patterns appearing in `run_full_downstream.sh`,
`rerun_2025_26_rapm.sh`, or any script called by those runners.

---

## Step 5 — When to Update docs/multiple-versions.md

Update the registry immediately when any of the following happen:

| Event | What to update |
|---|---|
| New PTS version developed (e.g., v4.1) | Add new row, mark old as SUPERSEDED, update canonical column table |
| New PTS version locked/promoted | Change status to **CURRENT PRODUCTION**; mark prior as BASELINE |
| BKE decomp produces a new version (e.g., v2.9) | Add row to BKE table, update canonical artifact name |
| Archetype script updated (internal version bump) | Update version tag in Defensive / Offensive Archetype table |
| Version fails / is rolled back | Mark status **FAILED** or **ROLLED BACK**, document reason |
| New artifact family added to pipeline | Add new section to the doc |

**Update rule:** The `docs/multiple-versions.md` entry must be updated **in the same commit** that promotes or deprecates a version. Never defer the doc update.

---

## Step 6 — Promoting a New Version (Checklist)

When a new metric version is ready to become production:

1. Confirm the artifact file exists at the expected path
2. Confirm season coverage matches expected (Step 3 probe above)
3. Confirm the validation gate passed (Brier, YoY_r, star sanity — per `CLAUDE.md`)
4. Update `src/modeling/model_config.py` constants (e.g., `PTS_V40_PARQUET`, `PtsV40CompositeConfig`)
5. Update `docs/multiple-versions.md` — mark old as SUPERSEDED, new as CURRENT PRODUCTION
6. Grep for hardcoded old version paths in production scripts and replace
7. Run `bash scripts/run_full_downstream.sh` to regenerate all downstream artifacts
8. Commit: doc update + config change + any script fixes in the **same commit**

---

## Non-Scope

- Do not change model weights or training logic — use `audit-model` for that.
- Do not run sweeps or experiments — this skill only tracks and enforces versions, not develops them.
- Do not touch dev/sweep scripts (pts_v40_multiseason.py etc.) — they legitimately read older versions.
