"""
src/player_eval/project_next_season.py
=============================================================================
Forward Projection Pipeline — Project Player Impact Profiles to Next Season

Takes prior season player impact profiles and produces projected profiles
for the upcoming season. Handles:
  1. Age-based regression/progression (piecewise linear curve)
  2. Team mapping (carry forward or from roster file)
  3. Rookie/newcomer projection (from rookies CSV or backtest actual data)
  4. Minute projection (age-adjusted carry-forward + team normalization)

Modes:
  - Backtest: Project season N from season N-1 using actual N data for
    team mappings and rookie identification. Enables validation.
  - Forecast: Project next season from latest available season using
    a roster/rookies input file. No actuals available.

Inputs:
  data/processed/player_eval/player_impact_profiles.parquet
  data/forecasting/rookies.csv  (optional, for true forecast)

Output:
  data/processed/forecast/projected_player_profiles.parquet

Usage:
  python3 src/player_eval/project_next_season.py                    # Backtest all available seasons
  python3 src/player_eval/project_next_season.py --forecast 2025-26 # True forecast
=============================================================================
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.player_eval.constants import (
    AGE_CURVE_BREAKPOINTS,
    AGE_CURVE_DELTAS,
    HISTORICAL_DIR,
    PLAYER_PROFILES_PARQUET,
    PROJECTED_PROFILES_PATH,
    FORECAST_VALIDATION_REPORT,
    REPORTS_DIR,
    ROOKIE_DEFAULT_3PT_RATE,
    ROOKIE_DEFAULT_AST_RATE,
    ROOKIE_DEFAULT_EFG,
    ROOKIE_DEFAULT_FTR,
    ROOKIE_DEFAULT_TOV_RATE,
    ROOKIE_DEFAULT_USAGE,
    ROOKIE_IMPACT_BY_TIER,
    ROOKIE_MPG_BY_TIER,
    ROOKIES_INPUT_PATH,
    DATA_DIR,
)


# ═════════════════════════════════════════════════════════════════════
# Age Curve
# ═════════════════════════════════════════════════════════════════════

def age_delta(age: float) -> float:
    """Return per-year impact delta for a player at given age.

    Uses piecewise linear interpolation between AGE_CURVE_BREAKPOINTS.
    Returns the expected change in BKE-scale impact for one year of aging.
    """
    if np.isnan(age):
        return 0.0

    # Below first breakpoint
    if age < AGE_CURVE_BREAKPOINTS[0]:
        return AGE_CURVE_DELTAS[0]

    # Between breakpoints: use the delta for the bracket the age falls in
    for i, bp in enumerate(AGE_CURVE_BREAKPOINTS):
        if age < bp:
            return AGE_CURVE_DELTAS[i]

    # Above last breakpoint
    return AGE_CURVE_DELTAS[-1]


def apply_age_curve(row: pd.Series) -> pd.Series:
    """Apply one year of aging to a player's impact metrics."""
    row = row.copy()
    age = float(row.get("age", 25))
    delta = age_delta(age)

    # Scale delta for different impact metrics:
    # Delta is already on BKE-scale (empirically calibrated ~0.01-0.08 per year).
    # ORAPM/DRAPM are on per-100 scale (~3.0 std vs BKE ~0.25 std), so scale up ~12x.
    bke_scale = delta
    rapm_scale = delta * 12.0  # BKE→RAPM expansion factor

    # Apply to impact metrics
    row["impact_bke"] = float(row.get("impact_bke", 0)) + bke_scale
    row["impact_obke"] = float(row.get("impact_obke", 0)) + bke_scale * 0.5
    row["impact_dbke"] = float(row.get("impact_dbke", 0)) + bke_scale * 0.5
    row["impact_orapm"] = float(row.get("impact_orapm", 0)) + rapm_scale * 0.5
    row["impact_drapm"] = float(row.get("impact_drapm", 0)) + rapm_scale * 0.5
    row["impact_total_impact"] = float(row.get("impact_total_impact", 50)) + delta * 15.0

    # BPM/WS/VORP: scale proportionally
    row["impact_bpm"] = float(row.get("impact_bpm", 0)) + rapm_scale * 0.3
    row["impact_ws"] = max(0.0, float(row.get("impact_ws", 0)) * (1.0 + delta * 0.5))
    row["impact_vorp"] = max(0.0, float(row.get("impact_vorp", 0)) * (1.0 + delta * 0.5))

    # Age the player
    row["age"] = age + 1.0
    row["experience_years"] = float(row.get("experience_years", 0)) + 1.0

    return row


# ═════════════════════════════════════════════════════════════════════
# Minute Projection (forecast-safe, no leakage)
# ═════════════════════════════════════════════════════════════════════

def project_minutes(
    players: pd.DataFrame,
    team_mpg_target: float = 240.0,
    mpg_cap: float = 40.0,
) -> pd.DataFrame:
    """Project next-season MPG from prior-season MPG + age adjustment.

    Age adjustment:
      Young players (<24): +1.5 MPG per year
      Developing (24-27): +0.5 MPG per year
      Peak (27-30): stable
      Declining (30-34): -1.0 MPG per year
      Late (34+): -2.0 MPG per year

    After individual adjustments, normalize per-team to sum to ~240.
    """
    df = players.copy()

    def _mpg_age_delta(age):
        if np.isnan(age):
            return 0.0
        if age < 22:
            return +2.0
        if age < 24:
            return +1.0
        if age < 27:
            return +0.5
        if age < 30:
            return 0.0
        if age < 34:
            return -1.0
        return -2.0

    df["projected_mpg"] = df.apply(
        lambda r: max(0.0, min(mpg_cap, float(r.get("mpg", 0)) + _mpg_age_delta(float(r.get("age", 25))))),
        axis=1,
    )

    # Team normalization: scale to 240 per team
    for (season, team), idx in df.groupby(["season", "team_abbreviation"]).groups.items():
        vals = df.loc[idx, "projected_mpg"].copy()
        total = vals.sum()
        if total > 0:
            scale = team_mpg_target / total
            vals = (vals * scale).clip(upper=mpg_cap)
            # Iterative redistribution after capping
            for _ in range(3):
                capped = vals.clip(upper=mpg_cap)
                excess = vals.sum() - capped.sum()
                if excess <= 0.1:
                    vals = capped
                    break
                free_mask = capped < mpg_cap
                if free_mask.sum() == 0:
                    vals = capped
                    break
                free_vals = capped[free_mask]
                capped.loc[free_mask] = free_vals + excess * (free_vals / free_vals.sum())
                vals = capped
            df.loc[idx, "projected_mpg"] = vals

    df["mpg"] = df["projected_mpg"]
    return df


# ═════════════════════════════════════════════════════════════════════
# Team Mapping
# ═════════════════════════════════════════════════════════════════════

def map_players_to_teams_backtest(
    prior_profiles: pd.DataFrame,
    target_profiles: pd.DataFrame,
) -> pd.DataFrame:
    """Map players to their actual next-season teams (for backtesting).

    For players in both seasons: use target season team.
    For players only in prior season: drop (they left the league).
    """
    prior = prior_profiles.copy()
    target_teams = target_profiles[["player_id", "team_abbreviation", "team_id"]].drop_duplicates(
        subset=["player_id"], keep="first"
    ).rename(columns={"team_abbreviation": "target_team", "team_id": "target_team_id"})
    target_teams["player_id"] = target_teams["player_id"].astype(str)
    prior["player_id"] = prior["player_id"].astype(str)

    merged = prior.merge(target_teams, on="player_id", how="inner")
    merged["team_abbreviation"] = merged["target_team"]
    merged["team_id"] = merged["target_team_id"]
    merged = merged.drop(columns=["target_team", "target_team_id"])
    return merged


def map_players_to_teams_forecast(
    prior_profiles: pd.DataFrame,
    roster_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Map players to next-season teams for true forecasting.

    If a roster file is provided, use it. Otherwise, assume players stay.
    """
    df = prior_profiles.copy()

    if roster_path and roster_path.exists():
        roster = pd.read_csv(roster_path)
        roster["player_id"] = roster["player_id"].astype(str)
        if "new_team" in roster.columns:
            roster_map = dict(zip(roster["player_id"], roster["new_team"]))
            df["team_abbreviation"] = df["player_id"].map(
                lambda pid: roster_map.get(str(pid), df.loc[df["player_id"] == pid, "team_abbreviation"].iloc[0]
                                            if pid in df["player_id"].values else "FA")
            )
    # Otherwise: carry forward current teams
    return df


# ═════════════════════════════════════════════════════════════════════
# Rookie Projection
# ═════════════════════════════════════════════════════════════════════

def _draft_tier(pick: int) -> str:
    if pick <= 14:
        return "lottery"
    if pick <= 25:
        return "mid_first"
    if pick <= 30:
        return "late_first"
    if pick <= 60:
        return "second_round"
    return "undrafted"


def _position_from_height(height_inches: float) -> str:
    """Estimate position from height."""
    if np.isnan(height_inches):
        return "Forward"
    if height_inches <= 75:  # <= 6'3"
        return "Guard"
    if height_inches <= 79:  # <= 6'7"
        return "Guard-Forward"
    if height_inches <= 81:  # <= 6'9"
        return "Forward"
    if height_inches <= 83:  # <= 6'11"
        return "Forward-Center"
    return "Center"


def build_rookie_profiles_backtest(
    target_profiles: pd.DataFrame,
    prior_player_ids: set,
    target_season: str,
) -> pd.DataFrame:
    """Build rookie profiles from actual target-season data (backtest).

    Rookies are players in the target season who were NOT in the prior season.
    We use their actual data but at REPLACEMENT level impact to simulate
    what a pre-season projection would look like.
    """
    target = target_profiles.copy()
    target["player_id"] = target["player_id"].astype(str)
    rookies = target[~target["player_id"].isin(prior_player_ids)].copy()

    # Filter out rookies with no valid team
    rookies = rookies[rookies["team_abbreviation"].notna() &
                      (rookies["team_abbreviation"].astype(str) != "nan")].copy()

    if rookies.empty:
        return pd.DataFrame(columns=target.columns)

    # Determine tier from experience_years
    def _tier_from_exp(exp):
        if exp <= 0:
            return "lottery"  # True rookies — optimistic default
        if exp <= 2:
            return "mid_first"
        return "undrafted"

    for idx, row in rookies.iterrows():
        exp = float(row.get("experience_years", 0))
        tier = _tier_from_exp(exp)

        # Override impact with tier-based replacement estimates
        rookies.at[idx, "impact_bke"] = ROOKIE_IMPACT_BY_TIER[tier]
        rookies.at[idx, "impact_obke"] = ROOKIE_IMPACT_BY_TIER[tier] * 0.5
        rookies.at[idx, "impact_dbke"] = ROOKIE_IMPACT_BY_TIER[tier] * 0.5
        rookies.at[idx, "impact_orapm"] = ROOKIE_IMPACT_BY_TIER[tier] * 8.8 * 0.5
        rookies.at[idx, "impact_drapm"] = ROOKIE_IMPACT_BY_TIER[tier] * 8.8 * 0.5
        rookies.at[idx, "impact_total_impact"] = 50.0 + ROOKIE_IMPACT_BY_TIER[tier] * 20.0
        rookies.at[idx, "impact_bpm"] = ROOKIE_IMPACT_BY_TIER[tier] * 3.0
        rookies.at[idx, "impact_stability"] = 0.35  # Low stability for projections

        # Use actual minutes but cap at tier maximum
        actual_mpg = float(row.get("mpg", 0))
        tier_mpg = ROOKIE_MPG_BY_TIER[tier]
        rookies.at[idx, "mpg"] = min(actual_mpg, tier_mpg * 1.3)  # Allow slight overshoot

    rookies["season"] = target_season
    return rookies


def build_rookie_profiles_forecast(
    rookies_path: Path,
    target_season: str,
    profile_columns: list,
) -> pd.DataFrame:
    """Build rookie profiles from a user-supplied CSV (true forecast).

    Expected CSV columns: player_name, team, draft_position, height_inches,
    weight_lbs, position (optional)
    """
    if not rookies_path.exists():
        print(f"  No rookies file found at {rookies_path}")
        return pd.DataFrame(columns=profile_columns)

    rookies_csv = pd.read_csv(rookies_path)
    rows = []

    for _, r in rookies_csv.iterrows():
        pick = int(r.get("draft_position", 60))
        tier = _draft_tier(pick)
        height = float(r.get("height_inches", 78))
        weight = float(r.get("weight_lbs", 210))
        position = str(r.get("position", _position_from_height(height)))
        team = str(r.get("team", "FA")).upper()

        impact = ROOKIE_IMPACT_BY_TIER.get(tier, -0.05)
        mpg = ROOKIE_MPG_BY_TIER.get(tier, 5.0)

        profile = {col: 0.0 for col in profile_columns}
        profile.update({
            "player_id": str(r.get("player_id", f"rookie_{pick}_{team}")),
            "player_name": str(r.get("player_name", f"Rookie #{pick}")),
            "season": target_season,
            "team_abbreviation": team,
            "team_id": "",
            "impact_bke": impact,
            "impact_obke": impact * 0.5,
            "impact_dbke": impact * 0.5,
            "impact_orapm": impact * 8.8 * 0.5,
            "impact_drapm": impact * 8.8 * 0.5,
            "impact_total_impact": 50.0 + impact * 20.0,
            "impact_bpm": impact * 3.0,
            "impact_stability": 0.35,
            "impact_ws": max(0, mpg * 0.04 * 82),
            "impact_vorp": max(0, impact * 3.0 * 0.05),
            "impact_portable_talent": 50.0,
            "behavioral_usage": ROOKIE_DEFAULT_USAGE,
            "behavioral_assist_rate": ROOKIE_DEFAULT_AST_RATE,
            "behavioral_turnover_rate": ROOKIE_DEFAULT_TOV_RATE,
            "behavioral_three_point_rate": ROOKIE_DEFAULT_3PT_RATE,
            "behavioral_efg": ROOKIE_DEFAULT_EFG,
            "behavioral_free_throw_rate": ROOKIE_DEFAULT_FTR,
            "behavioral_rim_rate": 0.15,
            "behavioral_orb_rate": 0.05,
            "behavioral_drb_rate": 0.12,
            "behavioral_hustle_pctl": 0.5,
            "behavioral_foul_rate": 3.0,
            "position_proxy": position,
            "age": float(r.get("age", 20)),
            "height_inches": height,
            "weight_lbs": weight,
            "experience_years": 0.0,
            "minutes": mpg * 72,  # Assume 72 games for rookies
            "mpg": mpg,
            "minute_share_potential": 0.0,
            "possessions": mpg * 72 * 2.0,
            "games": 72.0,
            "on_off_diff": 0.0,
            "scheme_stability_index": 0.0,
            "compression_flag": 0.0,
            "defensive_shrinkage_lambda": 0.5,
            "salary": float(r.get("salary", 3_000_000)),
            "availability_score": 0.85,
            "off_primary_archetype": str(r.get("off_archetype", "Connector")),
            "off_secondary_archetype": "",
            "off_role_confidence": 0.3,
            "off_role_effectiveness": 0.4,
            "def_primary_archetype": str(r.get("def_archetype", "Wing Defender")),
            "def_secondary_archetype": "",
            "def_role_confidence": 0.3,
        })

        # Set default archetype probabilities
        for arch_col in [c for c in profile_columns if c.startswith("off_prob_")]:
            profile[arch_col] = 1.0 / 11.0  # uniform prior
        for arch_col in [c for c in profile_columns if c.startswith("def_prob_")]:
            profile[arch_col] = 1.0 / 7.0   # uniform prior
        for pt_col in [c for c in profile_columns if c.startswith("playtype_")]:
            profile[pt_col] = 1.0 / 11.0    # uniform prior

        rows.append(profile)

    if not rows:
        return pd.DataFrame(columns=profile_columns)

    result = pd.DataFrame(rows)
    # Ensure proper types for non-numeric columns
    for col in ["player_id", "player_name", "season", "team_abbreviation", "team_id",
                 "position_proxy", "off_primary_archetype", "off_secondary_archetype",
                 "def_primary_archetype", "def_secondary_archetype"]:
        if col in result.columns:
            result[col] = result[col].astype(str)

    # Handle dict/list columns
    if "offensive_archetype_probs" in profile_columns:
        result["offensive_archetype_probs"] = [{}] * len(result)
    if "defensive_archetype_probs" in profile_columns:
        result["defensive_archetype_probs"] = [{}] * len(result)
    if "playtype_vector" in profile_columns:
        result["playtype_vector"] = [np.zeros(11).tolist()] * len(result)

    return result


# ═════════════════════════════════════════════════════════════════════
# Season Helpers
# ═════════════════════════════════════════════════════════════════════

def next_season_str(season: str) -> str:
    """Convert '2023-24' to '2024-25'."""
    start = int(season[:4])
    return f"{start + 1}-{str(start + 2)[-2:]}"


def prior_season_str(season: str) -> str:
    """Convert '2024-25' to '2023-24'."""
    start = int(season[:4])
    return f"{start - 1}-{str(start)[-2:]}"


# ═════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═════════════════════════════════════════════════════════════════════

def project_season(
    base_season: str,
    target_season: str,
    all_profiles: pd.DataFrame,
    mode: str = "backtest",
    roster_path: Optional[Path] = None,
    rookies_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Project players from base_season to target_season.

    Args:
        base_season: Source season (e.g., '2023-24')
        target_season: Target season to project into (e.g., '2024-25')
        all_profiles: Full impact profiles DataFrame (all seasons)
        mode: 'backtest' or 'forecast'
        roster_path: Optional CSV with team mappings for forecast mode
        rookies_path: Optional CSV with rookie data for forecast mode

    Returns:
        DataFrame with projected player profiles for target_season
    """
    print(f"\n  Projecting {base_season} → {target_season} (mode={mode})")

    # 1. Get base season profiles
    base = all_profiles[all_profiles["season"] == base_season].copy()
    base["player_id"] = base["player_id"].astype(str)
    if base.empty:
        print(f"    WARNING: No profiles for base season {base_season}")
        return pd.DataFrame(columns=all_profiles.columns)

    print(f"    Base season players: {len(base)}")

    # 2. Map players to new teams
    if mode == "backtest":
        target = all_profiles[all_profiles["season"] == target_season].copy()
        target["player_id"] = target["player_id"].astype(str)
        if target.empty:
            print(f"    WARNING: No target season data for {target_season}")
            return pd.DataFrame(columns=all_profiles.columns)
        projected = map_players_to_teams_backtest(base, target)
        print(f"    Returning players mapped: {len(projected)}")
    else:
        projected = map_players_to_teams_forecast(base, roster_path)
        print(f"    Players carried forward: {len(projected)}")

    # Drop players with no valid team (suspended, waived, etc.)
    before = len(projected)
    projected = projected[projected["team_abbreviation"].notna() &
                          (projected["team_abbreviation"].astype(str) != "nan")].copy()
    dropped = before - len(projected)
    if dropped > 0:
        print(f"    Dropped {dropped} players with no valid team assignment")

    # 3. Apply age curve to impact metrics
    projected = projected.apply(apply_age_curve, axis=1)

    # 4. Update season identifier
    projected["season"] = target_season

    # 5. Project minutes (age-adjusted carry-forward + team normalization)
    projected = project_minutes(projected)

    # Recompute derived minute fields
    projected["minutes"] = projected["mpg"] * projected["games"].clip(lower=60)
    projected["possessions"] = projected["minutes"] * 2.0  # rough approximation

    # 6. Add rookies
    prior_ids = set(projected["player_id"].astype(str))
    if mode == "backtest":
        target = all_profiles[all_profiles["season"] == target_season].copy()
        rookie_df = build_rookie_profiles_backtest(target, prior_ids, target_season)
    else:
        rp = rookies_path if rookies_path else ROOKIES_INPUT_PATH
        rookie_df = build_rookie_profiles_forecast(rp, target_season, list(all_profiles.columns))

    if not rookie_df.empty:
        # Ensure rookie_df has same columns
        for col in projected.columns:
            if col not in rookie_df.columns:
                rookie_df[col] = np.nan if projected[col].dtype in [float, np.float64] else ""
        rookie_df = rookie_df[[c for c in projected.columns if c in rookie_df.columns]]
        projected = pd.concat([projected, rookie_df], ignore_index=True)
        print(f"    Rookies/newcomers added: {len(rookie_df)}")

    print(f"    Total projected roster: {len(projected)}")

    # 7. Re-normalize team minutes after adding rookies
    projected = project_minutes(projected)

    return projected


def main() -> None:
    parser = argparse.ArgumentParser(description="Project player profiles to next season")
    parser.add_argument("--forecast", type=str, default=None,
                        help="Target season for true forecast (e.g., 2025-26)")
    parser.add_argument("--roster", type=str, default=None,
                        help="Path to roster CSV for forecast mode")
    parser.add_argument("--rookies", type=str, default=None,
                        help="Path to rookies CSV")
    args = parser.parse_args()

    print("=" * 60)
    print("Forward Projection Pipeline")
    print("=" * 60)

    if not PLAYER_PROFILES_PARQUET.exists():
        raise FileNotFoundError(f"Missing: {PLAYER_PROFILES_PARQUET}")

    all_profiles = pd.read_parquet(PLAYER_PROFILES_PARQUET)
    all_profiles["player_id"] = all_profiles["player_id"].astype(str)
    all_profiles["season"] = all_profiles["season"].astype(str)
    seasons = sorted(all_profiles["season"].unique())
    print(f"Available seasons: {seasons}")

    all_projected = []
    validation_results = {}

    if args.forecast:
        # True forecast mode: project from latest available season
        base_season = seasons[-1]
        target_season = args.forecast
        roster_path = Path(args.roster) if args.roster else None
        rookies_path = Path(args.rookies) if args.rookies else ROOKIES_INPUT_PATH

        projected = project_season(
            base_season=base_season,
            target_season=target_season,
            all_profiles=all_profiles,
            mode="forecast",
            roster_path=roster_path,
            rookies_path=rookies_path,
        )
        all_projected.append(projected)
    else:
        # Backtest mode: project each season pair
        for i in range(len(seasons) - 1):
            base = seasons[i]
            target = seasons[i + 1]

            projected = project_season(
                base_season=base,
                target_season=target,
                all_profiles=all_profiles,
                mode="backtest",
            )
            all_projected.append(projected)

            # Validate against actuals
            actual = all_profiles[all_profiles["season"] == target].copy()
            actual["player_id"] = actual["player_id"].astype(str)
            if not actual.empty:
                merged = projected.merge(
                    actual[["player_id", "impact_bke", "mpg", "team_abbreviation"]].rename(
                        columns={"impact_bke": "actual_bke", "mpg": "actual_mpg",
                                 "team_abbreviation": "actual_team"}
                    ),
                    on="player_id", how="inner",
                )
                if len(merged) > 5:
                    bke_corr = float(merged["impact_bke"].corr(merged["actual_bke"]))
                    bke_mae = float((merged["impact_bke"] - merged["actual_bke"]).abs().mean())
                    mpg_corr = float(merged["mpg"].corr(merged["actual_mpg"]))
                    mpg_mae = float((merged["mpg"] - merged["actual_mpg"]).abs().mean())
                    team_match = float((merged["team_abbreviation"] == merged["actual_team"]).mean())
                    n_rookies = len(projected) - len(merged)

                    validation_results[f"{base}→{target}"] = {
                        "n_returning": len(merged),
                        "n_rookies": n_rookies,
                        "bke_correlation": round(bke_corr, 4),
                        "bke_mae": round(bke_mae, 4),
                        "mpg_correlation": round(mpg_corr, 4),
                        "mpg_mae": round(mpg_mae, 2),
                        "team_match_rate": round(team_match, 4),
                    }
                    print(f"\n    Validation {base}→{target}:")
                    print(f"      BKE: r={bke_corr:.4f}, MAE={bke_mae:.4f}")
                    print(f"      MPG: r={mpg_corr:.4f}, MAE={mpg_mae:.2f}")
                    print(f"      Team match: {team_match:.1%}")

    if not all_projected:
        print("No projections generated.")
        return

    result = pd.concat(all_projected, ignore_index=True)

    # Save
    PROJECTED_PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(PROJECTED_PROFILES_PATH, index=False)
    print(f"\nSaved projected profiles: {PROJECTED_PROFILES_PATH}")
    print(f"  Shape: {result.shape}")
    print(f"  Seasons: {sorted(result['season'].unique())}")

    # Save validation report
    if validation_results:
        report = {
            "pipeline": "Forward Projection",
            "mode": "backtest",
            "projections": validation_results,
        }
        FORECAST_VALIDATION_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved validation: {FORECAST_VALIDATION_REPORT}")


if __name__ == "__main__":
    main()
