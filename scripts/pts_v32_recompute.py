"""
scripts/pts_v32_recompute.py
=============================================================================
BKE v3.2 PTS Recomputer — Post-hoc rebuild of PTS from existing decomp

Operates on data/processed/bke/bke_v28_decomposition.parquet WITHOUT
re-running the full Layer 1 pipeline. This lets us sweep weights cheaply.

The decomp already contains, per player-season:
  - dim_shooting_gravity_z         (Dim 1)
  - dim_driving_gravity_z          (Dim 2)
  - dim_playmaking_creation_z      (Dim 3)
  - dim_extra_possession_z         (Dim 4)
  - dim_defensive_playmaking_z     (Dim 5)
  - dim_defensive_impact_z         (Dim 6 — contaminated by DRAPM)
  - dim_turnover_control_z         (Dim 7)
  - dim_defensive_versatility_z    (Dim 8)
  - dim_self_creation_z            (Dim 9)
  - rapm_z, playtype_efficiency_z  (Layer 1A/1B — excluded in v3.2)

Architecture changes versus v2.7 PTS (`offensive_portable_z`,
`defensive_portable_z`):

  Fix 1 — Layer 1A RAPM:        excluded (already not in dim composites)
  Fix 2 — Dim 6 DRAPM:          REPLACE with matchup-based "def_impact_v32"
                                from defensive_archetypes_v2 (D_FG_DIFF,
                                d_results_pctl, contested_shots_pctl,
                                rim_protection_index_pctl). Fallback to
                                shrunken Dim 6 for seasons with poor
                                matchup coverage.
  Fix 3 — Dim 5 prune:          We don't have raw STL/BLK/DEFL components
                                in the decomp. Approximated by lowering
                                Dim 5's weight (0.08 → 0.05) and routing
                                the reclaimed budget into Dim 6 + Dim 8.
  Fix 4 — Layer 1B playtype:    excluded (already not in dim composites)

Outputs: parquet with player_id, season, pts_o_v32, pts_d_v32 + tracing
columns (dim contribution per player).
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DECOMP_PATH = REPO / "data/processed/bke/bke_v28_decomposition.parquet"
DEF_ARCH_PATH = REPO / "data/processed/defensive_archetypes_v2.parquet"
DEFAULT_OUTPUT = REPO / "data/processed/bke/pts_v32.parquet"

# ---------------------------------------------------------------------------
# v3.2 Config
# ---------------------------------------------------------------------------

@dataclass
class PtsV32Config:
    """All knobs for v3.2 PTS recomputation. Section 5 candidates."""

    # 5B — Dimension weights (must sum to 1.0; offensive >= 0.52)
    # Defaults: start near the plan's "redistribute Fix-3 budget into Dim 6 + 8"
    w_shooting_gravity: float = 0.16     # 0.14 → 0.16 (universally portable)
    w_driving_gravity: float = 0.10      # unchanged
    w_playmaking: float = 0.15           # 0.14 → 0.15
    w_extra_possession: float = 0.07     # 0.08 → 0.07 (slight trim)
    w_turnover_control: float = 0.09     # 0.10 → 0.09
    w_defensive_playmaking: float = 0.05  # 0.08 → 0.05 (Fix 3 prune-via-weight)
    w_defensive_impact: float = 0.12     # 0.10 → 0.12 (boost matchup-based Dim 6)
    w_defensive_versatility: float = 0.13  # 0.12 → 0.13
    w_self_creation: float = 0.13        # 0.14 → 0.13

    # 5B constraint allocations: offense ∈ {1,2,3,7,9} + extra_poss_off
    extra_poss_offensive_share: float = 0.40
    extra_poss_defensive_share: float = 0.60

    # 5C — Neutralization (already applied in decomp; toggle to BYPASS)
    use_neutralized_dims: bool = True   # True = use neutralized dim_*_z; False = use _raw_preshrink

    # 5D — Z-score clipping
    per_dim_clip: float = 3.0
    final_pts_clip: float = 3.5

    # Fix 2 — Dim 6 matchup replacement
    use_matchup_dim6: bool = True
    # When matchup data present, blend new dim6 = w_match * matchup_z + (1 - w_match) * legacy_dim6_z
    matchup_dim6_blend: float = 0.85  # 0.85 matchup, 0.15 legacy as residual stability
    matchup_dim6_components: Dict[str, float] = field(default_factory=lambda: {
        # column → weight (within matchup composite, normalized)
        "d_results_pctl": 0.35,           # opponent eFG% percentile (inverted lower=better is applied)
        "d_fg_diff": 0.30,                # DFG% vs expected
        "contested_shots_pctl": 0.15,
        "rim_protection_index_pctl": 0.20,
    })

    # 5G — Bayesian shrinkage toward archetype mean (defensive dims only)
    defensive_shrinkage: float = 0.15  # 0 = off

    # Variance restoration to keep offense and defense composites comparable
    enforce_unit_variance: bool = True

    # Multi-season smoothing (Section 5E)
    multi_season_smoothing: float = 0.0  # 0 = single season; 0.3 = 70/30 current/prior

    def to_dict(self) -> dict:
        return asdict(self)

    def dim_weights(self) -> Dict[str, float]:
        return {
            "dim_shooting_gravity_z": self.w_shooting_gravity,
            "dim_driving_gravity_z": self.w_driving_gravity,
            "dim_playmaking_creation_z": self.w_playmaking,
            "dim_extra_possession_z": self.w_extra_possession,
            "dim_turnover_control_z": self.w_turnover_control,
            "dim_defensive_playmaking_z": self.w_defensive_playmaking,
            "dim_defensive_impact_z": self.w_defensive_impact,
            "dim_defensive_versatility_z": self.w_defensive_versatility,
            "dim_self_creation_z": self.w_self_creation,
        }

    def validate(self):
        w = self.dim_weights()
        total = sum(w.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Dim weights sum to {total:.4f}, must be ~1.0")

        # Constraints from plan
        off = (self.w_shooting_gravity + self.w_driving_gravity + self.w_playmaking
               + self.w_turnover_control + self.w_self_creation
               + self.w_extra_possession * self.extra_poss_offensive_share)
        if off < 0.52:
            raise ValueError(f"Offensive weight share = {off:.3f}, plan minimum 0.52")
        def_total = (self.w_defensive_playmaking + self.w_defensive_impact
                     + self.w_defensive_versatility
                     + self.w_extra_possession * self.extra_poss_defensive_share)
        if def_total > 0.40:
            raise ValueError(f"Defensive weight share = {def_total:.3f}, plan ceiling 0.40")


# ---------------------------------------------------------------------------
# Dim 6 matchup-based recompute (Fix 2)
# ---------------------------------------------------------------------------

def build_matchup_dim6_z(
    def_arch: pd.DataFrame, cfg: PtsV32Config,
) -> pd.DataFrame:
    """Build dim_defensive_impact_matchup_z from defensive_archetypes_v2.

    Output: DataFrame with [player_id, season, dim6_matchup_z]
    """
    df = def_arch[["player_id", "season"]].copy()
    df["player_id"] = df["player_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["season"] = df["season"].astype(str)

    components = cfg.matchup_dim6_components
    z_components = []
    for col, w in components.items():
        if col not in def_arch.columns:
            continue
        vals = pd.to_numeric(def_arch[col], errors="coerce")
        # z-score within season; lower opponent shooting = better defense → invert if needed
        # D_FG_DIFF: negative = good (opponent shoots worse than expected)
        # d_results_pctl: high pctl = good (defense allows worse opp results)
        # contested_shots_pctl: high = active = good
        # rim_protection_index_pctl: high = good
        invert = col in {"d_fg_diff"}  # d_fg_diff "low=good" so invert
        z = vals.groupby(def_arch["season"].astype(str)).transform(
            lambda x: (x - x.mean()) / max(x.std(), 1e-6)
        )
        if invert:
            z = -z
        z = z.clip(-cfg.per_dim_clip, cfg.per_dim_clip)
        z_components.append((col, w, z))

    if not z_components:
        df["dim6_matchup_z"] = np.nan
        return df[["player_id", "season", "dim6_matchup_z"]]

    total_w = sum(w for _, w, _ in z_components)
    composite = sum(w / total_w * z.fillna(0) for _, w, z in z_components)
    df["dim6_matchup_z"] = composite.values
    # Re-standardize within season for unit variance
    df["dim6_matchup_z"] = df.groupby("season")["dim6_matchup_z"].transform(
        lambda x: (x - x.mean()) / max(x.std(), 1e-6)
    )
    return df[["player_id", "season", "dim6_matchup_z"]]


# ---------------------------------------------------------------------------
# Core recompute
# ---------------------------------------------------------------------------

DIM_COLS = [
    "dim_shooting_gravity_z",
    "dim_driving_gravity_z",
    "dim_playmaking_creation_z",
    "dim_extra_possession_z",
    "dim_turnover_control_z",
    "dim_defensive_playmaking_z",
    "dim_defensive_impact_z",
    "dim_defensive_versatility_z",
    "dim_self_creation_z",
]

OFF_DIMS = [
    "dim_shooting_gravity_z",
    "dim_driving_gravity_z",
    "dim_playmaking_creation_z",
    "dim_turnover_control_z",
    "dim_self_creation_z",
]
DEF_DIMS = [
    "dim_defensive_playmaking_z",
    "dim_defensive_impact_z",
    "dim_defensive_versatility_z",
]


def recompute_pts_v32(
    decomp: pd.DataFrame,
    def_arch: Optional[pd.DataFrame],
    cfg: PtsV32Config,
) -> pd.DataFrame:
    """Build pts_o_v32 and pts_d_v32 from existing dim z's + optional matchup Dim 6."""
    cfg.validate()
    df = decomp[["player_id", "season"] + DIM_COLS].copy()

    # Coerce
    df["player_id"] = df["player_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["season"] = df["season"].astype(str)
    for c in DIM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # ---- Fix 2 Dim 6 matchup replacement ----
    if cfg.use_matchup_dim6 and def_arch is not None:
        matchup6 = build_matchup_dim6_z(def_arch, cfg)
        df = df.merge(matchup6, on=["player_id", "season"], how="left")
        legacy_d6 = df["dim_defensive_impact_z"].clip(-cfg.per_dim_clip, cfg.per_dim_clip)
        match_d6 = df["dim6_matchup_z"].fillna(legacy_d6).clip(-cfg.per_dim_clip, cfg.per_dim_clip)
        blend = cfg.matchup_dim6_blend
        df["dim_defensive_impact_z_v32"] = blend * match_d6 + (1.0 - blend) * legacy_d6
    else:
        df["dim_defensive_impact_z_v32"] = df["dim_defensive_impact_z"].clip(
            -cfg.per_dim_clip, cfg.per_dim_clip
        )

    # ---- Defensive Bayesian shrinkage (Section 5G) ----
    if cfg.defensive_shrinkage > 0:
        s = cfg.defensive_shrinkage
        for c in ["dim_defensive_playmaking_z",
                  "dim_defensive_impact_z_v32",
                  "dim_defensive_versatility_z"]:
            df[c] = df.groupby("season")[c].transform(
                lambda x: (1 - s) * x + s * x.mean()
            )

    # ---- Clip per dim ----
    for c in DIM_COLS:
        df[c] = df[c].clip(-cfg.per_dim_clip, cfg.per_dim_clip)
    df["dim_defensive_impact_z_v32"] = df["dim_defensive_impact_z_v32"].clip(
        -cfg.per_dim_clip, cfg.per_dim_clip
    )

    # ---- Offensive PTS composite ----
    # Note: extra_possession splits 0.40 off / 0.60 def
    w_xp_off = cfg.w_extra_possession * cfg.extra_poss_offensive_share
    w_xp_def = cfg.w_extra_possession * cfg.extra_poss_defensive_share

    off_terms = (
        cfg.w_shooting_gravity * df["dim_shooting_gravity_z"]
        + cfg.w_driving_gravity * df["dim_driving_gravity_z"]
        + cfg.w_playmaking * df["dim_playmaking_creation_z"]
        + cfg.w_turnover_control * df["dim_turnover_control_z"]
        + cfg.w_self_creation * df["dim_self_creation_z"]
        + w_xp_off * df["dim_extra_possession_z"]
    )
    off_total_w = (cfg.w_shooting_gravity + cfg.w_driving_gravity + cfg.w_playmaking
                   + cfg.w_turnover_control + cfg.w_self_creation + w_xp_off)
    df["pts_o_v32_raw"] = off_terms / max(off_total_w, 1e-6)

    def_terms = (
        cfg.w_defensive_playmaking * df["dim_defensive_playmaking_z"]
        + cfg.w_defensive_impact * df["dim_defensive_impact_z_v32"]
        + cfg.w_defensive_versatility * df["dim_defensive_versatility_z"]
        + w_xp_def * df["dim_extra_possession_z"]
    )
    def_total_w = (cfg.w_defensive_playmaking + cfg.w_defensive_impact
                   + cfg.w_defensive_versatility + w_xp_def)
    df["pts_d_v32_raw"] = def_terms / max(def_total_w, 1e-6)

    # ---- Variance restoration (per season) ----
    if cfg.enforce_unit_variance:
        for col in ["pts_o_v32_raw", "pts_d_v32_raw"]:
            df[col] = df.groupby("season")[col].transform(
                lambda x: (x - x.mean()) / max(x.std(), 1e-6)
            )

    # ---- Multi-season smoothing ----
    if cfg.multi_season_smoothing > 0:
        s = cfg.multi_season_smoothing
        df = df.sort_values(["player_id", "season"])
        for col in ["pts_o_v32_raw", "pts_d_v32_raw"]:
            df[f"{col}_prev"] = df.groupby("player_id")[col].shift(1)
            df[col] = np.where(
                df[f"{col}_prev"].notna(),
                (1 - s) * df[col] + s * df[f"{col}_prev"],
                df[col],
            )
            df = df.drop(columns=[f"{col}_prev"])

    # ---- Final clip ----
    df["pts_o_v32"] = df["pts_o_v32_raw"].clip(-cfg.final_pts_clip, cfg.final_pts_clip)
    df["pts_d_v32"] = df["pts_d_v32_raw"].clip(-cfg.final_pts_clip, cfg.final_pts_clip)

    # Compatibility columns for downstream tools that look for offensive/defensive_portable_z
    df["offensive_portable_z"] = df["pts_o_v32"]
    df["defensive_portable_z"] = df["pts_d_v32"]

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decomp", default=str(DECOMP_PATH))
    parser.add_argument("--def-arch", default=str(DEF_ARCH_PATH))
    parser.add_argument("--config", default=None,
                        help="Optional JSON config file (overrides defaults)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()

    cfg = PtsV32Config()
    if args.config:
        with open(args.config) as f:
            overrides = json.load(f)
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
            else:
                print(f"  WARNING: unknown config key {k}")
    if args.print_config:
        print(json.dumps(cfg.to_dict(), indent=2))

    decomp = pd.read_parquet(args.decomp)
    def_arch = pd.read_parquet(args.def_arch) if os.path.exists(args.def_arch) else None
    out = recompute_pts_v32(decomp, def_arch, cfg)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(f"Saved → {args.output}  ({len(out)} rows, {len(out.columns)} cols)")
    print(f"  pts_o_v32 std: {out['pts_o_v32'].std():.3f}  "
          f"pts_d_v32 std: {out['pts_d_v32'].std():.3f}")


if __name__ == "__main__":
    main()
