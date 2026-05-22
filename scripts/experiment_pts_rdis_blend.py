"""
scripts/experiment_pts_rdis_blend.py
=============================================================================
Phase 3B Experiment: PTS / RDIS Blend Grid Search

Sweeps α across Approach A (fixed split) and Approach B (continuity-discounted)
to find the optimal weighting of Portable Talent (PTS) vs Role-Dependent Impact
(RDIS) for game-outcome prediction (Brier score).

For each α variant:
  new_OBKE_player = α * PTS_O_z + (1-α) * RDIS_O_z   (player level, unit-std normalized)
  new_DBKE_player = α * PTS_D_z + (1-α) * RDIS_D_z

  new_off_talent_team = Σ(minute_share * new_OBKE_player)
  new_def_talent_team = Σ(minute_share * new_DBKE_player)

  new_net_rating = old_net_rating
    + TEAM_SCALE * (new_off_talent - old_off_talent)
    + TEAM_SCALE * (new_def_talent - old_def_talent)

Approach B additionally discounts RDIS by a role_continuity factor per player.

Outputs:
  reports/experiment_pts_rdis_blend.json
=============================================================================
"""

import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

DECOMP_PATH      = os.path.join(REPO, "data/processed/bke/bke_v28_decomposition.parquet")
PROJ_PROFILES    = os.path.join(REPO, "data/processed/forecast/projected_player_profiles.parquet")
PROJ_TEAM_FEAT   = os.path.join(REPO, "data/processed/forecast/projected_team_features.parquet")
VALIDATE_SCRIPT  = os.path.join(REPO, "src/simulation/validate_forecast.py")
OUTPUT_JSON      = os.path.join(REPO, "reports/experiment_pts_rdis_blend.json")

TEAM_SCALE = 20.0  # DEFAULT_TEAM_SCALE

# Approach A alpha grid
ALPHAS_A = [1.00, 0.70, 0.60, 0.55, 0.50, 0.40, 0.00]
# Approach B runs these two alphas with continuity discount
ALPHAS_B = [0.60, 0.50]

ROLE_CONTINUITY = {
    "same_team": 0.90,
    "traded":    0.35,
    "rookie":    0.00,
}

# ---------------------------------------------------------------------------

def _unit_std(series: pd.Series) -> pd.Series:
    v = pd.to_numeric(series, errors="coerce").fillna(0.0)
    s = v.std()
    return v / s if s > 1e-9 else v * 0.0


def build_components(decomp: pd.DataFrame) -> pd.DataFrame:
    """
    Build unit-std-normalized PTS_O, PTS_D, RDIS_O, RDIS_D per player-season.

    PTS_O  = offensive_portable_z     (Layer 1 offensive dimensions)
    PTS_D  = defensive_portable_z     (Layer 1 defensive dimensions)
    RDIS_O = 0.55*RUE_z + 0.45*elevation_orapm_z  (mirrors v27 OBKE RDIS weights)
    RDIS_D = 0.60*elevation_drapm_z + 0.40*scheme_bonus_z
    """
    df = decomp[["player_id", "season"]].copy()

    pts_o  = pd.to_numeric(decomp.get("offensive_portable_z"),   errors="coerce").fillna(0.0)
    pts_d  = pd.to_numeric(decomp.get("defensive_portable_z"),   errors="coerce").fillna(0.0)
    rue    = pd.to_numeric(decomp.get("role_utilization_raw_z"), errors="coerce").fillna(0.0).clip(-2.5, 2.5)
    scheme = pd.to_numeric(decomp.get("scheme_stability_z"),     errors="coerce").fillna(0.0).clip(lower=0.0)

    # elevation_orapm/drapm are in pts/100 — normalize per season
    def season_znorm(raw_col):
        return decomp.groupby("season")[raw_col].transform(_unit_std).fillna(0.0)

    elev_o = season_znorm("elevation_orapm")
    elev_d = season_znorm("elevation_drapm")

    rdis_o = 0.55 * rue + 0.45 * elev_o
    rdis_d = 0.60 * elev_d + 0.40 * scheme

    # Normalize all four to unit std (league-wide, not per season)
    df["pts_o"]  = _unit_std(pts_o)
    df["pts_d"]  = _unit_std(pts_d)
    df["rdis_o"] = _unit_std(rdis_o)
    df["rdis_d"] = _unit_std(rdis_d)

    return df


def build_continuity_map(projected: pd.DataFrame) -> dict:
    """player_id × target_season → continuity_factor (how stable is their context)."""
    pp = projected[["player_id", "season", "team_abbreviation"]].copy()
    pp["player_id"] = pp["player_id"].astype(str)
    pp = pp.sort_values(["player_id", "season"])
    pp["prev_team"] = pp.groupby("player_id")["team_abbreviation"].shift(1)
    pp["cf"] = pp.apply(
        lambda r: ROLE_CONTINUITY["rookie"] if pd.isna(r["prev_team"])
        else (ROLE_CONTINUITY["same_team"] if r["prev_team"] == r["team_abbreviation"]
              else ROLE_CONTINUITY["traded"]),
        axis=1,
    )
    return {(str(r["player_id"]), str(r["season"])): r["cf"] for _, r in pp.iterrows()}


def compute_new_talent_bases(
    projected: pd.DataFrame,
    components: pd.DataFrame,
    alpha: float,
    continuity_map: dict = None,
) -> pd.DataFrame:
    """
    Per-player: compute new_obke, new_dbke from blend.
    Per-team:   minute-weight to new off/def talent base.
    Returns DataFrame: season, team_abbreviation, new_off_talent, new_def_talent.
    """
    # Map decomp season N → projected season N+1
    all_seasons = sorted(components["season"].unique())
    season_to_next = {s: all_seasons[i+1] for i, s in enumerate(all_seasons[:-1])}

    comp = components.copy()
    comp["target_season"] = comp["season"].map(season_to_next)
    comp = comp.dropna(subset=["target_season"])

    # Merge components into projected profiles on (player_id, target_season)
    pp = projected.copy()
    pp["player_id"] = pp["player_id"].astype(str)
    pp["season"]    = pp["season"].astype(str)

    comp_for_merge = comp.drop(columns=["season"]).rename(columns={"target_season": "season"})
    merged = pp.merge(
        comp_for_merge,
        on=["player_id", "season"],
        how="left",
    )

    # Fill missing components with 0 (no decomp data = treated as league avg)
    for c in ["pts_o", "pts_d", "rdis_o", "rdis_d"]:
        merged[c] = merged[c].fillna(0.0)

    # Continuity discount on RDIS (Approach B only)
    if continuity_map is not None:
        cf = merged.apply(
            lambda r: continuity_map.get((str(r["player_id"]), str(r["season"])),
                                         ROLE_CONTINUITY["same_team"]),
            axis=1,
        )
        merged["new_obke"] = alpha * merged["pts_o"] + (1.0 - alpha) * cf * merged["rdis_o"]
        merged["new_dbke"] = alpha * merged["pts_d"] + (1.0 - alpha) * cf * merged["rdis_d"]
    else:
        merged["new_obke"] = alpha * merged["pts_o"] + (1.0 - alpha) * merged["rdis_o"]
        merged["new_dbke"] = alpha * merged["pts_d"] + (1.0 - alpha) * merged["rdis_d"]

    # Scale new scores to match original impact_obke/dbke distribution
    orig_obke_std = pp["impact_obke"].std()
    orig_dbke_std = pp["impact_dbke"].std()
    orig_obke_mean = pp["impact_obke"].mean()
    orig_dbke_mean = pp["impact_dbke"].mean()

    new_obke_std = merged["new_obke"].std()
    new_dbke_std = merged["new_dbke"].std()

    if new_obke_std > 1e-9:
        merged["new_obke_scaled"] = (merged["new_obke"] / new_obke_std * orig_obke_std + orig_obke_mean)
    else:
        merged["new_obke_scaled"] = orig_obke_mean

    if new_dbke_std > 1e-9:
        merged["new_dbke_scaled"] = (merged["new_dbke"] / new_dbke_std * orig_dbke_std + orig_dbke_mean)
    else:
        merged["new_dbke_scaled"] = orig_dbke_mean

    # Compute minute share within team-season
    merged["minutes"] = pd.to_numeric(merged.get("minutes"), errors="coerce").fillna(0.0)
    team_min = merged.groupby(["season", "team_abbreviation"])["minutes"].transform("sum")
    merged["ms"] = merged["minutes"] / team_min.replace(0, np.nan).fillna(1.0)

    merged["w_off"] = merged["ms"] * merged["new_obke_scaled"]
    merged["w_def"] = merged["ms"] * merged["new_dbke_scaled"]

    team_talent = (
        merged.groupby(["season", "team_abbreviation"])
        .agg(new_off=("w_off", "sum"), new_def=("w_def", "sum"))
        .reset_index()
    )
    return team_talent


def patch_team_features(
    team_features: pd.DataFrame,
    new_talent: pd.DataFrame,
) -> pd.DataFrame:
    """
    Replace off_talent_base and def_talent_base with blended values,
    then recompute team_net_rating_projected.
    """
    tf = team_features.copy()
    tf["team_abbreviation"] = tf["team_abbreviation"].astype(str).str.upper()
    new_talent["team_abbreviation"] = new_talent["team_abbreviation"].astype(str).str.upper()

    merged = tf.merge(new_talent, on=["season", "team_abbreviation"], how="left")

    # Net rating delta from changed talent bases
    delta_off = (merged["new_off"].fillna(merged["off_talent_base"]) - merged["off_talent_base"])
    delta_def = (merged["new_def"].fillna(merged["def_talent_base"]) - merged["def_talent_base"])

    merged["team_net_rating_projected"] = (
        merged["team_net_rating_projected"]
        + TEAM_SCALE * delta_off
        + TEAM_SCALE * delta_def
    )
    merged["off_talent_base"] = merged["new_off"].fillna(merged["off_talent_base"])
    merged["def_talent_base"] = merged["new_def"].fillna(merged["def_talent_base"])
    merged = merged.drop(columns=["new_off", "new_def"])
    return merged


def run_brier(patched_features: pd.DataFrame) -> dict:
    """Save patched features to temp file, run validate_forecast, parse Brier."""
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        tmp_path = f.name
    try:
        patched_features.to_parquet(tmp_path, index=False)
        result = subprocess.run(
            [sys.executable, VALIDATE_SCRIPT, "--features-path", tmp_path],
            capture_output=True, text=True, cwd=REPO
        )
        out = result.stdout + result.stderr
        # Parse Brier from output
        m = re.search(r"Brier\s*[:=]\s*([\d.]+)", out, re.IGNORECASE)
        # Also try aggregate Brier from JSON-like output
        m2 = re.search(r'"brier_score"\s*:\s*([\d.]+)', out)
        brier = float(m2.group(1)) if m2 else (float(m.group(1)) if m else None)

        # Parse log_loss and accuracy
        ll_m = re.search(r"[Ll]og.?[Ll]oss\s*[:=]\s*([\d.]+)", out)
        acc_m = re.search(r"[Aa]ccuracy\s*[:=]\s*([\d.]+)", out)

        return {
            "brier": brier,
            "log_loss": float(ll_m.group(1)) if ll_m else None,
            "accuracy": float(acc_m.group(1)) if acc_m else None,
        }
    finally:
        os.unlink(tmp_path)


def run_variant(label, alpha, components, projected, team_features, continuity_map=None):
    print(f"  {label:<22} α={alpha:.2f}  B={'yes' if continuity_map else 'no '}  ", end="", flush=True)
    try:
        talent = compute_new_talent_bases(projected, components, alpha, continuity_map)
        patched = patch_team_features(team_features, talent)
        metrics = run_brier(patched)
        brier = metrics.get("brier")
        print(f"Brier={brier:.6f}" if brier else "Brier=N/A")
        return {"label": label, "alpha": alpha, "approach_b": continuity_map is not None,
                "brier": brier, **{k: v for k, v in metrics.items() if k != "brier"}}
    except Exception as e:
        print(f"ERROR: {e}")
        return {"label": label, "alpha": alpha, "approach_b": continuity_map is not None, "error": str(e)}


# ---------------------------------------------------------------------------
def main():
    print("=== Phase 3B: PTS/RDIS Blend Experiment ===\n")

    decomp   = pd.read_parquet(DECOMP_PATH)
    decomp["player_id"] = decomp["player_id"].astype(str)
    decomp["season"]    = decomp["season"].astype(str)

    projected = pd.read_parquet(PROJ_PROFILES)
    projected["player_id"] = projected["player_id"].astype(str)

    team_features = pd.read_parquet(PROJ_TEAM_FEAT)

    print("Building PTS/RDIS component vectors...")
    components = build_components(decomp)
    print(f"  {len(components)} player-seasons | "
          f"pts_o std={components['pts_o'].std():.3f}  "
          f"rdis_o std={components['rdis_o'].std():.3f}")

    continuity_map = build_continuity_map(projected)
    same = sum(1 for v in continuity_map.values() if v == ROLE_CONTINUITY["same_team"])
    print(f"  Continuity map: {len(continuity_map)} entries — "
          f"{same} same-team ({same/len(continuity_map):.0%})\n")

    results = []

    # --- Baseline ---
    print("BASELINE (no patching):", end="  ", flush=True)
    try:
        base = run_brier(team_features)
        b = base.get("brier")
        print(f"Brier={b:.6f}" if b else "Brier=N/A")
        results.append({"label": "BASELINE", "alpha": None, "approach_b": False, **base})
    except Exception as e:
        print(f"ERROR: {e}")
        results.append({"label": "BASELINE", "error": str(e)})

    # --- Approach A ---
    print("\nApproach A — fixed split:")
    for alpha in ALPHAS_A:
        label = f"A_α{int(alpha*100):03d}"
        results.append(run_variant(label, alpha, components, projected, team_features))

    # --- Approach B ---
    print("\nApproach B — continuity-discounted RDIS:")
    for alpha in ALPHAS_B:
        label = f"B_α{int(alpha*100):03d}"
        results.append(run_variant(label, alpha, components, projected, team_features, continuity_map))

    # --- Summary ---
    print("\n" + "="*55)
    print(f"{'Label':<22} {'α':>5} {'B?':>4} {'Brier':>10}")
    print("-"*55)
    baseline_brier = next((r.get("brier") for r in results if r["label"] == "BASELINE"), None)
    for r in results:
        if "error" in r:
            print(f"  {r['label']:<20} {'':>5} {'':>4}  ERROR")
            continue
        a_str = f"{r['alpha']:.2f}" if r.get("alpha") is not None else " cur"
        b_str = "yes" if r.get("approach_b") else "no"
        brier = r.get("brier")
        delta = f"({brier - baseline_brier:+.4f})" if (brier and baseline_brier) else ""
        print(f"  {r['label']:<20} {a_str:>5} {b_str:>4}  "
              f"{brier:.6f} {delta}" if brier else f"  {r['label']:<20}  N/A")

    valid = [r for r in results if r.get("brier") is not None and r["label"] != "BASELINE"]
    if valid:
        best = min(valid, key=lambda r: r["brier"])
        print(f"\nBest blend: {best['label']}  α={best['alpha']:.2f}  Brier={best['brier']:.6f}")
        if baseline_brier:
            d = best["brier"] - baseline_brier
            print(f"vs baseline: {d:+.6f} ({'improvement ✓' if d < 0 else 'regression ✗'})")

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump({"baseline_brier": baseline_brier, "experiments": results}, f, indent=2)
    print(f"\nSaved → {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
