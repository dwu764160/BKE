"""
src/data_compute/compute_position_estimate.py
=============================================================================

Estimate player position distribution (PG/SG/SF/PF/C) from on-court lineups.

Method (BRef-inspired, self-contained fallback):
1) Use possession-level lineups (`possessions_clean_{season}.parquet`).
2) For each 5-man lineup stint, sort players by height ascending.
3) Assign slots by rank: shortest->PG, 2nd->SG, 3rd->SF, 4th->PF, tallest->C.
4) Weight by stint duration seconds and aggregate to season-level percentages.

Outputs:
- data/processed/player_position_estimates_{season}.parquet
- data/processed/player_position_estimates_{season}.csv
- data/processed/player_position_estimates.parquet (combined compatibility output)
- data/processed/player_position_estimates.csv (combined compatibility output)
=============================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data")
HISTORICAL_DIR = DATA_DIR / "historical"
OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_DIR.mkdir(exist_ok=True)

SEASONS = ["2022-23", "2023-24", "2024-25"]
POSITION_SLOTS = ["PG", "SG", "SF", "PF", "C"]


def _clock_to_seconds(clock_val):
    if pd.isna(clock_val):
        return np.nan
    text = str(clock_val).strip()
    if not text or ":" not in text:
        return np.nan
    try:
        minutes, seconds = text.split(":", 1)
        return float(minutes) * 60.0 + float(seconds)
    except Exception:
        return np.nan


def _normalize_player_id(val):
    try:
        return int(float(val))
    except Exception:
        return None


def _position_group_defaults(pos_text: str):
    pos = str(pos_text or "").strip()
    if pos == "Guard":
        return np.array([0.55, 0.35, 0.08, 0.02, 0.00])
    if pos in ("Guard-Forward", "Forward-Guard"):
        return np.array([0.20, 0.35, 0.30, 0.12, 0.03])
    if pos == "Forward":
        return np.array([0.05, 0.15, 0.40, 0.30, 0.10])
    if pos in ("Forward-Center", "Center-Forward"):
        return np.array([0.01, 0.05, 0.18, 0.41, 0.35])
    if pos == "Center":
        return np.array([0.00, 0.02, 0.08, 0.30, 0.60])
    return np.array([0.20, 0.20, 0.20, 0.20, 0.20])


def _position_prior_for_tie_break(pos_text: str) -> int:
    pos = str(pos_text or "").strip()
    if pos == "Guard":
        return 0
    if pos in ("Guard-Forward", "Forward-Guard"):
        return 1
    if pos == "Forward":
        return 2
    if pos in ("Forward-Center", "Center-Forward"):
        return 3
    if pos == "Center":
        return 4
    return 2


def _derive_primary_position(row):
    pg = row.get("pct_pg", 0.0)
    sg = row.get("pct_sg", 0.0)
    sf = row.get("pct_sf", 0.0)
    pf = row.get("pct_pf", 0.0)
    c = row.get("pct_c", 0.0)

    guard_share = pg + sg
    forward_share = sf + pf
    big_share = pf + c

    if c >= 0.58:
        return "Center"
    if big_share >= 0.62 and c >= 0.18:
        return "Forward-Center"
    if guard_share >= 0.65 and pg >= 0.20:
        return "Guard"
    if (
        (sg + sf) >= 0.52
        and 0.05 <= pg <= 0.35
        and c <= 0.18
        and guard_share >= 0.45
        and forward_share >= 0.25
    ):
        return "Guard-Forward"
    if forward_share >= 0.58 and guard_share <= 0.42 and c <= 0.32:
        return "Forward"
    if c >= 0.30 and big_share >= 0.55:
        return "Forward-Center"
    if guard_share >= 0.55:
        return "Guard"
    return "Forward"


def load_player_bios() -> pd.DataFrame:
    path = HISTORICAL_DIR / "players.parquet"
    if not path.exists():
        return pd.DataFrame()

    bios = pd.read_parquet(path)
    bios = bios.rename(columns={
        "player_id": "PLAYER_ID",
        "full_name": "PLAYER_NAME",
    })
    if "PLAYER_ID" not in bios.columns:
        return pd.DataFrame()

    bios["PLAYER_ID"] = pd.to_numeric(bios["PLAYER_ID"], errors="coerce")
    return bios[[c for c in ["PLAYER_ID", "PLAYER_NAME", "primary_position", "height_inches"] if c in bios.columns]].drop_duplicates("PLAYER_ID")


def load_player_season_index() -> pd.DataFrame:
    path = HISTORICAL_DIR / "complete_player_season_stats.parquet"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path)
    cols = ["PLAYER_ID", "PLAYER_NAME", "SEASON", "GP", "MIN"]
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.DataFrame()

    idx = df[cols].copy()
    idx["PLAYER_ID"] = pd.to_numeric(idx["PLAYER_ID"], errors="coerce")
    return idx.dropna(subset=["PLAYER_ID", "SEASON"]).drop_duplicates(["PLAYER_ID", "SEASON"])


def _rank_lineup_players(player_ids, height_map, pos_map):
    unique_ids = []
    seen = set()
    for raw_id in player_ids:
        pid = _normalize_player_id(raw_id)
        if pid is None or pid in seen:
            continue
        seen.add(pid)
        unique_ids.append(pid)

    if len(unique_ids) != 5:
        return []

    lineup_rows = []
    for pid in unique_ids:
        pos_text = pos_map.get(pid, "")
        height_val = height_map.get(pid, np.nan)
        if pd.isna(height_val):
            default_dist = _position_group_defaults(pos_text)
            expected_slot = int(np.argmax(default_dist))
            height_val = 74.0 + expected_slot * 2.0
        lineup_rows.append((pid, float(height_val), _position_prior_for_tie_break(pos_text)))

    lineup_rows = sorted(lineup_rows, key=lambda x: (x[1], x[2], x[0]))
    return [(pid, POSITION_SLOTS[idx]) for idx, (pid, _, _) in enumerate(lineup_rows)]


def compute_position_estimate_for_season(season: str, bios: pd.DataFrame) -> pd.DataFrame:
    path = HISTORICAL_DIR / f"possessions_clean_{season}.parquet"
    if not path.exists():
        return pd.DataFrame()

    poss = pd.read_parquet(path)
    if poss.empty:
        return pd.DataFrame()

    required = {"off_lineup", "def_lineup", "start_clock", "end_clock"}
    if not required.issubset(set(poss.columns)):
        return pd.DataFrame()

    poss = poss.copy()
    poss["start_sec"] = poss["start_clock"].apply(_clock_to_seconds)
    poss["end_sec"] = poss["end_clock"].apply(_clock_to_seconds)
    poss["duration_seconds"] = (poss["start_sec"] - poss["end_sec"]).clip(lower=0)

    positive_durations = poss.loc[poss["duration_seconds"] > 0, "duration_seconds"]
    fallback_duration = float(positive_durations.median()) if len(positive_durations) else 14.0
    poss["duration_seconds"] = poss["duration_seconds"].fillna(fallback_duration)
    poss["duration_seconds"] = poss["duration_seconds"].clip(lower=1.0, upper=60.0)

    bio_map = bios.set_index("PLAYER_ID") if not bios.empty else pd.DataFrame()
    height_map = bio_map["height_inches"].to_dict() if "height_inches" in bio_map.columns else {}
    pos_map = bio_map["primary_position"].to_dict() if "primary_position" in bio_map.columns else {}

    records = []
    for row in poss.itertuples(index=False):
        seconds = float(getattr(row, "duration_seconds", 0.0) or 0.0)
        if seconds <= 0:
            continue

        for lineup_name in ("off_lineup", "def_lineup"):
            lineup = getattr(row, lineup_name, None)
            if lineup is None or not isinstance(lineup, (list, np.ndarray)):
                continue

            ranked = _rank_lineup_players(lineup, height_map, pos_map)
            if not ranked:
                continue

            for pid, slot in ranked:
                records.append((pid, season, slot, seconds))

    if not records:
        return pd.DataFrame()

    raw = pd.DataFrame(records, columns=["PLAYER_ID", "SEASON", "slot", "seconds"])
    agg = raw.groupby(["PLAYER_ID", "SEASON", "slot"], as_index=False)["seconds"].sum()

    pivot = (
        agg.pivot_table(index=["PLAYER_ID", "SEASON"], columns="slot", values="seconds", fill_value=0.0)
        .reset_index()
    )

    for slot in POSITION_SLOTS:
        if slot not in pivot.columns:
            pivot[slot] = 0.0

    pivot["total_position_seconds"] = pivot[POSITION_SLOTS].sum(axis=1)
    for slot in POSITION_SLOTS:
        pivot[f"pct_{slot.lower()}"] = pivot[slot] / pivot["total_position_seconds"].replace(0, np.nan)
        pivot[f"pct_{slot.lower()}"] = pivot[f"pct_{slot.lower()}"] .fillna(0.0)

    pivot["pct_guards_own"] = pivot["pct_pg"] + pivot["pct_sg"]
    pivot["pct_forwards_own"] = pivot["pct_sf"] + pivot["pct_pf"]
    pivot["pct_centers_own"] = pivot["pct_c"]

    pos_cols = ["pct_pg", "pct_sg", "pct_sf", "pct_pf", "pct_c"]
    probs = pivot[pos_cols].to_numpy(dtype=float)
    entropy_terms = np.zeros_like(probs)
    positive_mask = probs > 0
    entropy_terms[positive_mask] = probs[positive_mask] * np.log2(probs[positive_mask])
    pivot["position_entropy"] = -entropy_terms.sum(axis=1)
    pivot["primary_position_estimate"] = pivot.apply(_derive_primary_position, axis=1)
    pivot["position_estimate_method"] = "lineup_height_rank_v1"

    return pivot


def apply_bio_fallbacks(position_df: pd.DataFrame, season_index: pd.DataFrame, bios: pd.DataFrame) -> pd.DataFrame:
    if season_index.empty:
        return position_df

    merged = season_index.merge(position_df, on=["PLAYER_ID", "SEASON"], how="left")

    bios_c = bios[[c for c in ["PLAYER_ID", "primary_position", "height_inches"] if c in bios.columns]].copy()
    merged = merged.merge(bios_c, on="PLAYER_ID", how="left", suffixes=("", "_bio"))

    if "height_inches" not in merged.columns:
        merged["height_inches"] = np.nan
    if "height_inches_bio" in merged.columns:
        merged["height_inches"] = merged["height_inches"].fillna(merged["height_inches_bio"])

    if "primary_position" not in merged.columns:
        merged["primary_position"] = np.nan

    for idx, row in merged[merged["total_position_seconds"].isna()].iterrows():
        base_pos = row.get("primary_position", "")
        dist = _position_group_defaults(base_pos)
        merged.at[idx, "pct_pg"] = float(dist[0])
        merged.at[idx, "pct_sg"] = float(dist[1])
        merged.at[idx, "pct_sf"] = float(dist[2])
        merged.at[idx, "pct_pf"] = float(dist[3])
        merged.at[idx, "pct_c"] = float(dist[4])
        merged.at[idx, "pct_guards_own"] = float(dist[0] + dist[1])
        merged.at[idx, "pct_forwards_own"] = float(dist[2] + dist[3])
        merged.at[idx, "pct_centers_own"] = float(dist[4])
        pos_mask = dist > 0
        entropy_val = float(-(dist[pos_mask] * np.log2(dist[pos_mask])).sum())
        merged.at[idx, "position_entropy"] = entropy_val
        merged.at[idx, "primary_position_estimate"] = _derive_primary_position({
            "pct_pg": dist[0],
            "pct_sg": dist[1],
            "pct_sf": dist[2],
            "pct_pf": dist[3],
            "pct_c": dist[4],
        })
        merged.at[idx, "position_estimate_method"] = "bio_prior_fallback"
        merged.at[idx, "total_position_seconds"] = 0.0

    if "primary_position_estimate" in merged.columns and "primary_position" in merged.columns:
        merged["primary_position_estimate"] = merged["primary_position_estimate"].fillna(merged["primary_position"])

    merged["height_inches"] = pd.to_numeric(merged["height_inches"], errors="coerce")
    merged["height_inches"] = merged["height_inches"].fillna(78)

    final_cols = [
        "PLAYER_ID", "PLAYER_NAME", "SEASON", "GP", "MIN",
        "primary_position", "height_inches",
        "primary_position_estimate", "position_estimate_method",
        "total_position_seconds", "position_entropy",
        "pct_pg", "pct_sg", "pct_sf", "pct_pf", "pct_c",
        "pct_guards_own", "pct_forwards_own", "pct_centers_own",
    ]
    existing = [c for c in final_cols if c in merged.columns]
    out = merged[existing].copy()
    out = out.drop_duplicates(["PLAYER_ID", "SEASON"]).sort_values(["SEASON", "PLAYER_NAME"])
    return out


def main():
    print("=" * 72)
    print("POSITION ESTIMATE (LINEUP HEIGHT RANKING)")
    print("=" * 72)

    bios = load_player_bios()
    season_index = load_player_season_index()

    all_seasons = []
    for season in SEASONS:
        print(f"\nProcessing {season}...")
        season_est = compute_position_estimate_for_season(season, bios)
        season_idx = season_index[season_index["SEASON"] == season].copy() if not season_index.empty else pd.DataFrame()
        season_final = apply_bio_fallbacks(season_est, season_idx, bios)
        print(f"  Computed rows: {len(season_est)} | Final rows (with fallbacks): {len(season_final)}")

        season_out_parquet = OUTPUT_DIR / f"player_position_estimates_{season}.parquet"
        season_out_csv = OUTPUT_DIR / f"player_position_estimates_{season}.csv"
        season_final.to_parquet(season_out_parquet, index=False)
        season_final.to_csv(season_out_csv, index=False)
        print(f"  - {season_out_parquet}")
        print(f"  - {season_out_csv}")
        all_seasons.append(season_final)

    final_df = pd.concat([df for df in all_seasons if not df.empty], ignore_index=True) if all_seasons else pd.DataFrame()

    # Keep combined compatibility artifact for downstream scripts that still expect one file.
    out_parquet = OUTPUT_DIR / "player_position_estimates.parquet"
    out_csv = OUTPUT_DIR / "player_position_estimates.csv"
    final_df.to_parquet(out_parquet, index=False)
    final_df.to_csv(out_csv, index=False)

    print(f"\nSaved combined {len(final_df)} rows")
    print(f"  - {out_parquet}")
    print(f"  - {out_csv}")

    if not final_df.empty:
        print("\nPrimary position estimate distribution:")
        print(final_df["primary_position_estimate"].value_counts().to_string())


if __name__ == "__main__":
    main()
