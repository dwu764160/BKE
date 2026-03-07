"""
src/modeling/model_rapm.py
=============================================================================
RAPM / ORAPM / DRAPM modeling with pooled-prior refinement.

Key design choices:
- RAPM (net) uses lineup matrix with +1 offense / -1 defense.
- ORAPM/DRAPM use opponent-controlled split matrix in one joint fit.
- Multi-season mode uses pooled prior, then refines to target-season coefficients.
=============================================================================
"""

import glob
import os
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, identity, vstack
from sklearn.linear_model import RidgeCV


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

DATA_DIR = "data/historical"
OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEASON_DECAY_WEIGHTS = {
    0: 1.00,
    1: 0.70,
    2: 0.50,
    3: 0.35,
}

RAPM_ALPHAS_SINGLE = [25, 50, 100, 200, 400, 800, 1600, 3200]
RAPM_ALPHAS_POOLED = [10, 25, 50, 100, 200, 400, 800, 1600]
PRIOR_REFINEMENT_STRENGTH = 30.0

POSSESSION_WEIGHT_MODE = "uniform"  # options: uniform, duration_sqrt


def clean_id(val) -> str:
    if pd.isna(val):
        return "0"
    return str(val).replace(".0", "")


def _safe_clock_to_seconds(clock_value) -> float:
    try:
        if pd.isna(clock_value):
            return 0.0
        parts = str(clock_value).split(":")
        if len(parts) != 2:
            return 0.0
        return int(parts[0]) * 60 + float(parts[1])
    except Exception:
        return 0.0


def load_clean_possessions() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(DATA_DIR, "possessions_clean_*.parquet")))
    if not files:
        print("❌ No clean possession files found in data/historical/")
        return pd.DataFrame()

    print(f"Loading {len(files)} possession files...")
    all_parts: List[pd.DataFrame] = []
    for path in files:
        part = pd.read_parquet(path)
        if "season" not in part.columns:
            season = os.path.basename(path).replace("possessions_clean_", "").replace(".parquet", "")
            part["season"] = season
        all_parts.append(part)

    full_df = pd.concat(all_parts, ignore_index=True)
    full_df = add_possession_duration(full_df)
    return full_df


def add_possession_duration(df: pd.DataFrame) -> pd.DataFrame:
    if "start_clock" not in df.columns or "end_clock" not in df.columns:
        df["duration_weight"] = 1.0
        return df

    df = df.copy()
    df["start_seconds"] = df["start_clock"].apply(_safe_clock_to_seconds)
    df["end_seconds"] = df["end_clock"].apply(_safe_clock_to_seconds)
    df["duration"] = (df["start_seconds"] - df["end_seconds"]).clip(lower=1, upper=60)
    df["duration_weight"] = np.sqrt(df["duration"])
    return df


def get_sample_weights(df: pd.DataFrame) -> np.ndarray:
    if POSSESSION_WEIGHT_MODE == "duration_sqrt" and "duration_weight" in df.columns:
        return df["duration_weight"].values.astype(float)
    return np.ones(len(df), dtype=float)


def collect_player_index(df: pd.DataFrame) -> Tuple[List[str], Dict[str, int]]:
    players = set()
    for col in ["off_lineup", "def_lineup"]:
        for lineup in df[col]:
            if isinstance(lineup, (list, np.ndarray)):
                for pid in lineup:
                    players.add(clean_id(pid))
    players.discard("0")
    sorted_players = sorted(players)
    mapping = {pid: idx for idx, pid in enumerate(sorted_players)}
    return sorted_players, mapping


def build_rapm_matrix(df: pd.DataFrame, player_to_idx: Dict[str, int]) -> Tuple[csr_matrix, np.ndarray]:
    n_players = len(player_to_idx)
    n_rows = len(df)

    data: List[int] = []
    rows: List[int] = []
    cols: List[int] = []

    off_lineups = df["off_lineup"].values
    def_lineups = df["def_lineup"].values

    for row_idx in range(n_rows):
        off = off_lineups[row_idx]
        if isinstance(off, (list, np.ndarray)):
            for pid in off:
                key = clean_id(pid)
                if key in player_to_idx:
                    rows.append(row_idx)
                    cols.append(player_to_idx[key])
                    data.append(1)

        defn = def_lineups[row_idx]
        if isinstance(defn, (list, np.ndarray)):
            for pid in defn:
                key = clean_id(pid)
                if key in player_to_idx:
                    rows.append(row_idx)
                    cols.append(player_to_idx[key])
                    data.append(-1)

    x_matrix = csr_matrix((data, (rows, cols)), shape=(n_rows, n_players))
    y = df["points"].values.astype(float)
    return x_matrix, y


def build_split_matrix(df: pd.DataFrame, player_to_idx: Dict[str, int]) -> Tuple[csr_matrix, np.ndarray]:
    """
    Joint offense/defense matrix (opponent-controlled):
    - First n_players columns: offensive presence (+1)
    - Next n_players columns: defensive presence (+1)

    Target = points scored by offense on each possession.
    DRAPM = - defensive coefficient * 100 (higher is better defense).
    """
    n_players = len(player_to_idx)
    n_rows = len(df)
    total_cols = n_players * 2

    data: List[int] = []
    rows: List[int] = []
    cols: List[int] = []

    off_lineups = df["off_lineup"].values
    def_lineups = df["def_lineup"].values

    for row_idx in range(n_rows):
        off = off_lineups[row_idx]
        if isinstance(off, (list, np.ndarray)):
            for pid in off:
                key = clean_id(pid)
                if key in player_to_idx:
                    rows.append(row_idx)
                    cols.append(player_to_idx[key])
                    data.append(1)

        defn = def_lineups[row_idx]
        if isinstance(defn, (list, np.ndarray)):
            for pid in defn:
                key = clean_id(pid)
                if key in player_to_idx:
                    rows.append(row_idx)
                    cols.append(n_players + player_to_idx[key])
                    data.append(1)

    x_matrix = csr_matrix((data, (rows, cols)), shape=(n_rows, total_cols))
    y = df["points"].values.astype(float)
    return x_matrix, y


def fit_ridge(x_matrix: csr_matrix, y: np.ndarray, sample_weights: np.ndarray, alphas: Sequence[float]) -> RidgeCV:
    model = RidgeCV(alphas=list(alphas), fit_intercept=True)
    model.fit(x_matrix, y, sample_weight=sample_weights)
    return model


def _extract_target_and_prior_seasons(all_seasons: Sequence[str], target_season: str, n_prior_seasons: int) -> List[Tuple[str, float]]:
    if target_season not in all_seasons:
        return []
    target_idx = list(all_seasons).index(target_season)
    output: List[Tuple[str, float]] = []
    for back in range(n_prior_seasons + 1):
        idx = target_idx - back
        if idx >= 0:
            output.append((all_seasons[idx], SEASON_DECAY_WEIGHTS.get(back, 0.25)))
    return output


def _fit_prior_then_refine(
    x_pooled: csr_matrix,
    y_pooled: np.ndarray,
    w_pooled: np.ndarray,
    x_target: csr_matrix,
    y_target: np.ndarray,
    w_target: np.ndarray,
    pooled_alphas: Sequence[float],
    refine_alphas: Sequence[float],
    prior_strength: float,
) -> Tuple[RidgeCV, RidgeCV]:
    pooled_model = fit_ridge(x_pooled, y_pooled, w_pooled, pooled_alphas)

    n_features = x_target.shape[1]
    prior_x = identity(n_features, format="csr") * prior_strength
    prior_y = pooled_model.coef_ * prior_strength
    prior_w = np.ones(n_features, dtype=float)

    x_aug = vstack([x_target, prior_x])
    y_aug = np.concatenate([y_target, prior_y])
    w_aug = np.concatenate([w_target, prior_w])

    refined_model = fit_ridge(x_aug, y_aug, w_aug, refine_alphas)
    return pooled_model, refined_model


def run_single_season_rapm(df: pd.DataFrame, season: str, player_to_idx: Dict[str, int], sorted_players: Sequence[str]) -> pd.DataFrame:
    print(f"   [SINGLE RAPM] {season}: building matrix ({len(df):,} possessions, {len(sorted_players):,} players)")
    x_matrix, y = build_rapm_matrix(df, player_to_idx)
    weights = get_sample_weights(df)
    print(f"   [SINGLE RAPM] {season}: fitting RidgeCV (alphas={list(RAPM_ALPHAS_SINGLE)})")
    model = fit_ridge(x_matrix, y, weights, RAPM_ALPHAS_SINGLE)
    print(f"   [SINGLE RAPM] {season}: done (alpha={model.alpha_}, intercept={model.intercept_ * 100:.2f})")

    rows = []
    for pid, coef in zip(sorted_players, model.coef_):
        rows.append(
            {
                "season": season,
                "player_id": pid,
                "RAPM": coef * 100,
                "RAPM_type": "single_season",
                "n_seasons_pooled": 1,
                "alpha": model.alpha_,
                "intercept": model.intercept_ * 100,
            }
        )
    return pd.DataFrame(rows)


def run_single_season_split(df: pd.DataFrame, season: str, player_to_idx: Dict[str, int], sorted_players: Sequence[str]) -> pd.DataFrame:
    print(f"   [SINGLE SPLIT] {season}: building matrix ({len(df):,} possessions, {len(sorted_players):,} players)")
    x_matrix, y = build_split_matrix(df, player_to_idx)
    weights = get_sample_weights(df)
    print(f"   [SINGLE SPLIT] {season}: fitting RidgeCV (alphas={list(RAPM_ALPHAS_SINGLE)})")
    model = fit_ridge(x_matrix, y, weights, RAPM_ALPHAS_SINGLE)
    print(f"   [SINGLE SPLIT] {season}: done (alpha={model.alpha_}, intercept={model.intercept_ * 100:.2f})")

    n_players = len(sorted_players)
    off_coef = model.coef_[:n_players]
    def_coef = model.coef_[n_players:]

    rows = []
    for idx, pid in enumerate(sorted_players):
        orapm = off_coef[idx] * 100
        drapm = -def_coef[idx] * 100
        rows.append(
            {
                "season": season,
                "player_id": pid,
                "ORAPM": orapm,
                "DRAPM": drapm,
                "RAPM": orapm + drapm,
                "RAPM_type": "single_season_split",
                "n_seasons_pooled": 1,
                "alpha": model.alpha_,
                "intercept": model.intercept_ * 100,
            }
        )
    return pd.DataFrame(rows)


def run_pooled_refined_rapm(full_df: pd.DataFrame, target_season: str, n_prior_seasons: int = 2) -> pd.DataFrame:
    seasons = sorted(full_df["season"].unique())
    seasons_to_use = _extract_target_and_prior_seasons(seasons, target_season, n_prior_seasons)
    if not seasons_to_use:
        return pd.DataFrame()

    pooled_df = full_df[full_df["season"].isin([s for s, _ in seasons_to_use])].copy()
    target_df = full_df[full_df["season"] == target_season].copy()

    sorted_players, player_to_idx = collect_player_index(pooled_df)
    print(f"   [POOLED RAPM] {target_season}: {len(sorted_players)} players, seasons={[(s, f'{w:.0%}') for s, w in seasons_to_use]}")
    print(f"   [POOLED RAPM] {target_season}: assembling pooled/target matrices...")

    pooled_parts_x = []
    pooled_parts_y = []
    pooled_parts_w = []
    for season, season_weight in seasons_to_use:
        part = full_df[full_df["season"] == season]
        x_part, y_part = build_rapm_matrix(part, player_to_idx)
        w_part = get_sample_weights(part) * season_weight
        pooled_parts_x.append(x_part)
        pooled_parts_y.append(y_part)
        pooled_parts_w.append(w_part)

    x_pooled = vstack(pooled_parts_x)
    y_pooled = np.concatenate(pooled_parts_y)
    w_pooled = np.concatenate(pooled_parts_w)

    x_target, y_target = build_rapm_matrix(target_df, player_to_idx)
    w_target = get_sample_weights(target_df)

    pooled_model, refined_model = _fit_prior_then_refine(
        x_pooled=x_pooled,
        y_pooled=y_pooled,
        w_pooled=w_pooled,
        x_target=x_target,
        y_target=y_target,
        w_target=w_target,
        pooled_alphas=RAPM_ALPHAS_POOLED,
        refine_alphas=RAPM_ALPHAS_SINGLE,
        prior_strength=PRIOR_REFINEMENT_STRENGTH,
    )

    print(
        "   [POOLED RAPM] "
        f"alpha_pooled={pooled_model.alpha_}, "
        f"alpha_refined={refined_model.alpha_}, "
        f"diagnostic_in_sample_r2={refined_model.score(x_target, y_target, sample_weight=w_target):.4f}"
    )

    rows = []
    for pid, coef in zip(sorted_players, refined_model.coef_):
        rows.append(
            {
                "season": target_season,
                "player_id": pid,
                "RAPM": coef * 100,
                "RAPM_type": "pooled",
                "n_seasons_pooled": len(seasons_to_use),
                "alpha": refined_model.alpha_,
                "alpha_pooled": pooled_model.alpha_,
                "intercept": refined_model.intercept_ * 100,
            }
        )
    return pd.DataFrame(rows)


def run_pooled_refined_split(full_df: pd.DataFrame, target_season: str, n_prior_seasons: int = 2) -> pd.DataFrame:
    seasons = sorted(full_df["season"].unique())
    seasons_to_use = _extract_target_and_prior_seasons(seasons, target_season, n_prior_seasons)
    if not seasons_to_use:
        return pd.DataFrame()

    pooled_df = full_df[full_df["season"].isin([s for s, _ in seasons_to_use])].copy()
    target_df = full_df[full_df["season"] == target_season].copy()

    sorted_players, player_to_idx = collect_player_index(pooled_df)
    n_players = len(sorted_players)
    print(f"   [POOLED SPLIT] {target_season}: {n_players} players, seasons={[(s, f'{w:.0%}') for s, w in seasons_to_use]}")
    print(f"   [POOLED SPLIT] {target_season}: assembling pooled/target matrices...")

    pooled_parts_x = []
    pooled_parts_y = []
    pooled_parts_w = []
    for season, season_weight in seasons_to_use:
        part = full_df[full_df["season"] == season]
        x_part, y_part = build_split_matrix(part, player_to_idx)
        w_part = get_sample_weights(part) * season_weight
        pooled_parts_x.append(x_part)
        pooled_parts_y.append(y_part)
        pooled_parts_w.append(w_part)

    x_pooled = vstack(pooled_parts_x)
    y_pooled = np.concatenate(pooled_parts_y)
    w_pooled = np.concatenate(pooled_parts_w)

    x_target, y_target = build_split_matrix(target_df, player_to_idx)
    w_target = get_sample_weights(target_df)

    pooled_model, refined_model = _fit_prior_then_refine(
        x_pooled=x_pooled,
        y_pooled=y_pooled,
        w_pooled=w_pooled,
        x_target=x_target,
        y_target=y_target,
        w_target=w_target,
        pooled_alphas=RAPM_ALPHAS_POOLED,
        refine_alphas=RAPM_ALPHAS_SINGLE,
        prior_strength=PRIOR_REFINEMENT_STRENGTH,
    )

    print(
        "   [POOLED SPLIT] "
        f"alpha_pooled={pooled_model.alpha_}, "
        f"alpha_refined={refined_model.alpha_}, "
        f"diagnostic_in_sample_r2={refined_model.score(x_target, y_target, sample_weight=w_target):.4f}"
    )

    off_coef = refined_model.coef_[:n_players]
    def_coef = refined_model.coef_[n_players:]

    rows = []
    for idx, pid in enumerate(sorted_players):
        orapm = off_coef[idx] * 100
        drapm = -def_coef[idx] * 100
        rows.append(
            {
                "season": target_season,
                "player_id": pid,
                "ORAPM": orapm,
                "DRAPM": drapm,
                "RAPM": orapm + drapm,
                "RAPM_type": "pooled_split",
                "n_seasons_pooled": len(seasons_to_use),
                "alpha": refined_model.alpha_,
                "alpha_pooled": pooled_model.alpha_,
                "intercept": refined_model.intercept_ * 100,
            }
        )
    return pd.DataFrame(rows)


def calculate_player_possessions(full_df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for col in ["off_lineup", "def_lineup"]:
        exploded = full_df[["season", col]].explode(col).dropna(subset=[col])
        exploded[col] = exploded[col].apply(clean_id)
        exploded = exploded[exploded[col] != "0"]
        counts = exploded.groupby(["season", col]).size().reset_index(name="_count")
        counts.columns = ["season", "player_id", "_count"]
        parts.append(counts)

    if not parts:
        return pd.DataFrame(columns=["season", "player_id", "possessions_played"])

    combo = pd.concat(parts, ignore_index=True)
    poss = combo.groupby(["season", "player_id"])["_count"].sum().reset_index()
    poss.columns = ["season", "player_id", "possessions_played"]
    return poss


def enrich_names(df: pd.DataFrame) -> pd.DataFrame:
    try:
        players_path = os.path.join(DATA_DIR, "players.parquet")
        if not os.path.exists(players_path):
            df["player_name"] = df["player_id"]
            return df

        players = pd.read_parquet(players_path)
        if "id" not in players.columns:
            df["player_name"] = df["player_id"]
            return df

        players = players.copy()
        players["id"] = players["id"].astype(str).apply(clean_id)
        name_map = players.set_index("id")["full_name"].to_dict()
        df["player_name"] = df["player_id"].map(name_map).fillna("Unknown")
    except Exception as exc:
        print(f"⚠️ Name enrichment failed: {exc}")
        df["player_name"] = df["player_id"]
    return df


def build_all_results(full_df: pd.DataFrame) -> pd.DataFrame:
    seasons = sorted(full_df["season"].unique())
    all_parts: List[pd.DataFrame] = []

    for season in seasons:
        print(f"\n{'=' * 60}")
        print(f"SEASON: {season}")
        print(f"{'=' * 60}")

        season_df = full_df[full_df["season"] == season].copy()
        sorted_players, player_to_idx = collect_player_index(season_df)

        all_parts.append(run_single_season_rapm(season_df, season, player_to_idx, sorted_players))
        all_parts.append(run_single_season_split(season_df, season, player_to_idx, sorted_players))

        if len(seasons) > 1:
            pooled = run_pooled_refined_rapm(full_df, season, n_prior_seasons=2)
            if not pooled.empty:
                all_parts.append(pooled)

            pooled_split = run_pooled_refined_split(full_df, season, n_prior_seasons=2)
            if not pooled_split.empty:
                all_parts.append(pooled_split)

    return pd.concat(all_parts, ignore_index=True)


def print_summary(final_df: pd.DataFrame, seasons: Sequence[str]) -> None:
    print("\n" + "=" * 60)
    print("RAPM SUMMARY")
    print("=" * 60)

    for season in seasons:
        print(f"\n📅 {season}")
        pooled = final_df[(final_df["season"] == season) & (final_df["RAPM_type"] == "pooled")]
        if not pooled.empty:
            top = pooled.nlargest(5, "RAPM")[["player_name", "RAPM", "possessions_played"]]
            print("   Top 5 pooled RAPM:")
            for _, row in top.iterrows():
                print(f"      {row['player_name']:25s}: {row['RAPM']:+6.2f} ({int(row['possessions_played']):,} poss)")

        split = final_df[(final_df["season"] == season) & (final_df["RAPM_type"] == "pooled_split")]
        if not split.empty:
            top_o = split.nlargest(5, "ORAPM")[["player_name", "ORAPM"]]
            top_d = split.nlargest(5, "DRAPM")[["player_name", "DRAPM"]]
            print("   Top 5 pooled ORAPM:")
            for _, row in top_o.iterrows():
                print(f"      {row['player_name']:25s}: {row['ORAPM']:+6.2f}")
            print("   Top 5 pooled DRAPM:")
            for _, row in top_d.iterrows():
                print(f"      {row['player_name']:25s}: {row['DRAPM']:+6.2f}")


def main() -> None:
    print("=" * 60)
    print("RAPM / ORAPM / DRAPM MODELING")
    print("=" * 60)
    print(f"Weight mode: {POSSESSION_WEIGHT_MODE}")

    full_df = load_clean_possessions()
    if full_df.empty:
        print("❌ No possession data found")
        return

    print(f"\n📊 Loaded {len(full_df):,} possessions")
    seasons = sorted(full_df["season"].unique())
    print(f"📅 Seasons: {seasons}")

    final_df = build_all_results(full_df)

    poss_df = calculate_player_possessions(full_df)
    final_df = final_df.merge(poss_df, on=["season", "player_id"], how="left")
    final_df["possessions_played"] = final_df["possessions_played"].fillna(0)

    final_df = enrich_names(final_df)

    sort_metric = final_df["RAPM"].fillna(-9999)
    final_df = final_df.assign(_sort_metric=sort_metric).sort_values(
        ["season", "RAPM_type", "_sort_metric"], ascending=[True, True, False]
    )
    final_df = final_df.drop(columns=["_sort_metric"])

    out_parquet = os.path.join(OUTPUT_DIR, "player_rapm.parquet")
    out_csv = os.path.join(OUTPUT_DIR, "player_rapm.csv")
    final_df.to_parquet(out_parquet, index=False)
    final_df.to_csv(out_csv, index=False)

    print(f"\n✅ Saved {out_parquet}")
    print(f"✅ Saved {out_csv}")

    print_summary(final_df, seasons)


if __name__ == "__main__":
    main()
