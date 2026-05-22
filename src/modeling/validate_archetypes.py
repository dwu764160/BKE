"""
validate_archetypes.py — Phase 4.3 archetype validation

Track 1: Year-to-year stability matrix (11×11 offensive transitions, 7 season-pairs)
Track 2: Threshold sensitivity (±2 pp shift on ball_dominant_pct gate)
Track 3: Within-archetype feature coherence (signal/noise per archetype)
Track 4: 30-player manual spot-check per archetype (top minutes, readable stats)

Run:
    python3 src/modeling/validate_archetypes.py
    python3 src/modeling/validate_archetypes.py --track 1
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

AGGREGATE_PATH = Path("aggregate/player_profile_aggregate.parquet")
ARCHE_PATH = Path("data/processed/player_archetypes.parquet")

OFFENSIVE_ARCHETYPES = [
    "Ball Dominant Creator",
    "Ballhandler",
    "All-Around Scorer",
    "Interior Scorer",
    "Perimeter Scorer",
    "PnR Rolling Big",
    "PnR Popping Big",
    "Off-Ball Finisher",
    "Off-Ball Movement Shooter",
    "Off-Ball Stationary Shooter",
    "Connector",
]

# Season pairs available: 7 pairs from 8 seasons.
# 2021-22→2022-23 flagged: tracking/synergy feature set changed (box-score-only pre-2022).
ALL_PAIRS = [
    ("2017-18", "2018-19"),
    ("2018-19", "2019-20"),
    ("2019-20", "2020-21"),
    ("2020-21", "2021-22"),
    ("2021-22", "2022-23"),  # FLAGGED — feature-set discontinuity
    ("2022-23", "2023-24"),
    ("2023-24", "2024-25"),
]
FLAGGED_PAIR = ("2021-22", "2022-23")

# Key feature columns for Track 3 coherence
COHERENCE_FEATURES = [
    "arche_ball_dominant_pct",
    "arche_ast",
    "arche_at_rim_freq",
    "arche_drives",
    "arche_cut_pnrrm_pct",
    "arche_spotup_pct",
    "arche_movement_pct",
    "arche_prrollman_pct",
    "arche_midrange_freq",
    "arche_fg3a_per36",
    "arche_fg3_pct",
    "arche_post_up_pct",
]


def load_data():
    agg = pd.read_parquet(AGGREGATE_PATH)
    # Normalize archetype column name
    arch_col = None
    for c in ["primary_archetype", "arche_primary_archetype"]:
        if c in agg.columns:
            arch_col = c
            break
    if arch_col is None:
        raise ValueError("No primary_archetype column found in aggregate")
    agg["_arch"] = agg[arch_col]
    min_col = "min" if "min" in agg.columns else "MIN"
    agg["_min"] = pd.to_numeric(agg[min_col], errors="coerce").fillna(0)
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Track 1: Stability matrix
# ─────────────────────────────────────────────────────────────────────────────

def build_transition_matrix(df, season_a, season_b, min_gate):
    """Return (n×n transition matrix, n_players) for a season pair."""
    a = df[df["season"] == season_a][["player_id", "_arch", "_min"]].copy()
    b = df[df["season"] == season_b][["player_id", "_arch", "_min"]].copy()
    merged = a.merge(b, on="player_id", suffixes=("_a", "_b"))
    if min_gate > 0:
        merged = merged[(merged["_min_a"] >= min_gate) & (merged["_min_b"] >= min_gate)]
    merged = merged[
        merged["_arch_a"].isin(OFFENSIVE_ARCHETYPES) &
        merged["_arch_b"].isin(OFFENSIVE_ARCHETYPES)
    ]
    n = len(OFFENSIVE_ARCHETYPES)
    idx = {a: i for i, a in enumerate(OFFENSIVE_ARCHETYPES)}
    mat = np.zeros((n, n), dtype=int)
    for _, row in merged.iterrows():
        i = idx.get(row["_arch_a"])
        j = idx.get(row["_arch_b"])
        if i is not None and j is not None:
            mat[i, j] += 1
    return mat, len(merged)


def diagonal_rate(mat):
    total = mat.sum()
    if total == 0:
        return 0.0
    return mat.diagonal().sum() / total


def archetype_retention(mat):
    """Per-archetype retention (row-normalized diagonal)."""
    row_sums = mat.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.where(row_sums > 0, mat.diagonal() / row_sums, np.nan)
    return ret


def run_track1(df):
    print("\n" + "=" * 72)
    print("TRACK 1: YEAR-TO-YEAR ARCHETYPE STABILITY")
    print("=" * 72)

    result = {"gate_500min": {}, "gate_200min": {}}
    gate_keys = ["gate_500min", "gate_200min"]

    for (min_gate, gate_label), rkey in zip(
        [(500, "≥500 min gate"), (200, "≥200 min full population")], gate_keys
    ):
        print(f"\n── {gate_label} ──")
        print(f"{'Pair':<25} {'N':>5} {'Diag%':>7} {'Flag'}")
        print("-" * 50)
        rates_no_flag = []
        mats = {}
        pair_rows = []
        for s_a, s_b in ALL_PAIRS:
            mat, n = build_transition_matrix(df, s_a, s_b, min_gate)
            dr = diagonal_rate(mat) * 100
            is_flagged = (s_a, s_b) == FLAGGED_PAIR
            flag = " ← EXCLUDED (feature discontinuity)" if is_flagged else ""
            mats[(s_a, s_b)] = mat
            print(f"  {s_a}→{s_b}   {n:>5}  {dr:>6.1f}%{flag}")
            pair_rows.append({"pair": f"{s_a}→{s_b}", "n": n, "diag_pct": round(dr, 1), "excluded": is_flagged})
            if not is_flagged:
                rates_no_flag.append(dr)

        gate_mean = np.mean(rates_no_flag) if rates_no_flag else 0
        print(f"\n  Mean diagonal (6 valid pairs): {gate_mean:.1f}%")

        # Per-archetype retention aggregated across valid pairs
        agg_ret = np.zeros(len(OFFENSIVE_ARCHETYPES))
        agg_count = np.zeros(len(OFFENSIVE_ARCHETYPES), dtype=int)
        for s_a, s_b in ALL_PAIRS:
            if (s_a, s_b) == FLAGGED_PAIR:
                continue
            mat = mats[(s_a, s_b)]
            row_sums = mat.sum(axis=1)
            for i, arch in enumerate(OFFENSIVE_ARCHETYPES):
                if row_sums[i] > 0:
                    agg_ret[i] += mat[i, i] / row_sums[i]
                    agg_count[i] += 1

        print(f"\n  Per-archetype mean retention ({gate_label}):")
        print(f"  {'Archetype':<30} {'Retention':>10} {'Pairs':>6}")
        print(f"  {'-'*48}")
        arch_retention = {}
        for i, arch in enumerate(OFFENSIVE_ARCHETYPES):
            if agg_count[i] > 0:
                r = agg_ret[i] / agg_count[i] * 100
            else:
                r = float("nan")
            flag = " ← low" if (not np.isnan(r) and r < 40) else ""
            print(f"  {arch:<30} {r:>9.1f}%{flag}" if not np.isnan(r) else f"  {arch:<30} {'N/A':>10}")
            arch_retention[arch] = round(r, 1) if not np.isnan(r) else None

        result[rkey] = {
            "mean_diagonal_pct": round(gate_mean, 1),
            "gate_pct": int(min_gate),
            "pairs": pair_rows,
            "per_archetype_retention": arch_retention,
        }

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Track 2: Threshold sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def run_track2(df):
    print("\n" + "=" * 72)
    print("TRACK 2: THRESHOLD SENSITIVITY (±2 pp on ball_dominant_pct)")
    print("=" * 72)
    print()

    bd_col = "arche_ball_dominant_pct"
    arch_col = "_arch"

    if bd_col not in df.columns:
        print(f"  [{bd_col} not found — Track 2 skipped]")
        return {}

    # Focus on 2024-25 full population
    season_df = df[df["season"] == "2024-25"].copy()
    has_bd = season_df[bd_col].notna()
    season_df = season_df[has_bd].copy()
    bd = pd.to_numeric(season_df[bd_col], errors="coerce")

    result = {"season": "2024-25", "gates": []}
    # Thresholds from model_config (approximate canonical values)
    for gate_name, center in [("BDC min (BD_MIN ~0.30)", 0.30), ("Perimeter BD_BASE (~0.17)", 0.17)]:
        near_gate = (bd - center).abs() <= 0.02
        near_players = season_df[near_gate]
        n_near = len(near_players)
        arch_dist = near_players[arch_col].value_counts()
        print(f"  Gate: {gate_name}")
        print(f"    Players within ±2pp of {center:.2f}: {n_near}")
        dist_dict = {}
        if n_near > 0:
            print(f"    Archetype distribution of borderline players:")
            for arch, cnt in arch_dist.head(6).items():
                print(f"      {arch:<30} {cnt:>4} ({cnt/n_near*100:.0f}%)")
                dist_dict[str(arch)] = int(cnt)
        print()
        result["gates"].append({
            "gate": gate_name,
            "center": center,
            "n_borderline": n_near,
            "arch_distribution": dist_dict,
        })

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Track 3: Within-archetype coherence
# ─────────────────────────────────────────────────────────────────────────────

def run_track3(df):
    print("\n" + "=" * 72)
    print("TRACK 3: WITHIN-ARCHETYPE FEATURE COHERENCE")
    print("=" * 72)
    print()

    available = [c for c in COHERENCE_FEATURES if c in df.columns]
    if not available:
        print("  [No coherence features found — Track 3 skipped]")
        return {}

    # Use all 8 seasons, only primary-archetype-labeled rows
    labeled = df[df["_arch"].isin(OFFENSIVE_ARCHETYPES)].copy()
    for c in available:
        labeled[c] = pd.to_numeric(labeled[c], errors="coerce")

    # Between-archetype variance (grand mean per feature)
    between_var = labeled.groupby("_arch")[available].mean().var(axis=0)
    # Within-archetype variance (mean of per-group variances)
    within_var = labeled.groupby("_arch")[available].var().mean(axis=0)
    # Cohesion ratio: between / (between + within) — higher = more separable
    total_var = between_var + within_var
    cohesion = (between_var / total_var.replace(0, np.nan)).fillna(0)

    print(f"  {'Feature':<35} {'Cohesion':>9}  (higher = archetypes more distinct)")
    print(f"  {'-'*55}")
    for feat in sorted(cohesion.index, key=lambda x: -cohesion[x]):
        short = feat.replace("arche_", "").replace("_pct", "%").replace("_per36", "/36")
        print(f"  {short:<35} {cohesion[feat]:>8.3f}")

    print()
    # Per-archetype: mean and std on top 2 features
    top2 = cohesion.nlargest(2).index.tolist()
    print(f"  Per-archetype mean ± std on top-2 features ({', '.join(f.replace('arche_','') for f in top2)}):")
    print(f"  {'Archetype':<30}  {'Feature 1':>18}  {'Feature 2':>18}")
    print(f"  {'-'*70}")
    grp = labeled.groupby("_arch")[top2]
    means = grp.mean()
    stds = grp.std()
    arch_detail = {}
    for arch in OFFENSIVE_ARCHETYPES:
        if arch in means.index:
            m1, s1 = means.loc[arch, top2[0]], stds.loc[arch, top2[0]]
            m2, s2 = means.loc[arch, top2[1]], stds.loc[arch, top2[1]]
            print(f"  {arch:<30}  {m1:>7.3f}±{s1:<8.3f}  {m2:>7.3f}±{s2:<8.3f}")
            arch_detail[arch] = {
                top2[0].replace("arche_", ""): {"mean": round(float(m1), 4), "std": round(float(s1), 4)},
                top2[1].replace("arche_", ""): {"mean": round(float(m2), 4), "std": round(float(s2), 4)},
            }

    return {
        "cohesion_scores": {feat.replace("arche_", ""): round(float(v), 4) for feat, v in cohesion.items()},
        "top2_features": [f.replace("arche_", "") for f in top2],
        "per_archetype_top2": arch_detail,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Track 4: 30-player spot-check per archetype
# ─────────────────────────────────────────────────────────────────────────────

def run_track4(df):
    print("\n" + "=" * 72)
    print("TRACK 4: 30-PLAYER SPOT-CHECK PER ARCHETYPE (2024-25, top minutes)")
    print("=" * 72)

    season_df = df[df["season"] == "2024-25"].copy()
    season_df = season_df[season_df["_arch"].isin(OFFENSIVE_ARCHETYPES)]

    display_cols = []
    for c in ["player_name", "player_tier", "secondary_archetype", "_min",
              "arche_ball_dominant_pct", "arche_ast", "arche_fg3_pct",
              "arche_at_rim_freq", "arche_post_up_pct", "arche_drives",
              "arche_cut_pnrrm_pct", "arche_spotup_pct", "arche_movement_pct",
              "arche_prrollman_pct", "role_confidence", "role_effectiveness"]:
        if c in season_df.columns:
            display_cols.append(c)

    all_rows = []
    for arch in OFFENSIVE_ARCHETYPES:
        arch_df = season_df[season_df["_arch"] == arch].sort_values("_min", ascending=False).head(30)
        n = len(arch_df)
        print(f"\n{'─'*72}")
        print(f"  {arch.upper()} (n={n} in 2024-25)")
        print(f"{'─'*72}")
        if n == 0:
            print("  [No players]")
            continue
        sub = arch_df[display_cols].copy()
        sub.insert(0, "archetype", arch)
        sub = sub.rename(columns={
            "_min": "min",
            "arche_ball_dominant_pct": "bd_pct",
            "arche_ast": "ast/g",
            "arche_fg3_pct": "3p%",
            "arche_at_rim_freq": "rim%",
            "arche_post_up_pct": "post%",
            "arche_drives": "drives",
            "arche_cut_pnrrm_pct": "cut%",
            "arche_spotup_pct": "spotup%",
            "arche_movement_pct": "move%",
            "arche_prrollman_pct": "roll%",
            "role_confidence": "conf",
            "role_effectiveness": "eff",
        })
        # Format numerics
        for c in sub.columns:
            if c in ("archetype", "player_name", "player_tier", "secondary_archetype"):
                continue
            sub[c] = pd.to_numeric(sub[c], errors="coerce").round(3)
        print(sub.drop(columns=["archetype"]).to_string(index=False, max_colwidth=28))
        all_rows.append(sub)

        # Sanity check: flag any player with conf < 0.65
        if "conf" in sub.columns:
            low_conf = arch_df[pd.to_numeric(arch_df.get("role_confidence"), errors="coerce").fillna(1) < 0.65]
            if len(low_conf) > 0:
                names = low_conf.get("player_name", pd.Series(["?"] * len(low_conf))).tolist()
                print(f"  ⚠  Low confidence (<0.65): {', '.join(str(n) for n in names[:5])}")

    return pd.concat(all_rows, ignore_index=True) if all_rows else None


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", type=int, choices=[1, 2, 3, 4], default=0,
                        help="Run only a specific track (default: all)")
    args = parser.parse_args()

    print("Loading aggregate...")
    df = load_data()
    print(f"  Loaded {len(df)} rows, {df['season'].nunique()} seasons")
    print(f"  Archetype coverage: {df['_arch'].isin(OFFENSIVE_ARCHETYPES).sum()} labeled rows")

    reports = Path("reports")
    reports.mkdir(exist_ok=True)

    run_all = args.track == 0
    if run_all or args.track == 1:
        stability = run_track1(df)
        if stability and run_all:
            (reports / "archetype_stability.json").write_text(
                json.dumps(stability, indent=2, default=str)
            )
            print(f"\n  [Saved reports/archetype_stability.json]")
    if run_all or args.track == 2:
        sensitivity = run_track2(df)
        if sensitivity and run_all:
            (reports / "archetype_sensitivity.json").write_text(
                json.dumps(sensitivity, indent=2, default=str)
            )
            print(f"  [Saved reports/archetype_sensitivity.json]")
    if run_all or args.track == 3:
        coherence = run_track3(df)
        if coherence and run_all:
            (reports / "archetype_coherence.json").write_text(
                json.dumps(coherence, indent=2, default=str)
            )
            print(f"  [Saved reports/archetype_coherence.json]")
    if run_all or args.track == 4:
        sample_df = run_track4(df)
        if sample_df is not None and run_all:
            sample_df.to_csv(reports / "archetype_manual_sample.csv", index=False)
            print(f"  [Saved reports/archetype_manual_sample.csv]")


if __name__ == "__main__":
    main()
