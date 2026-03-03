"""
Normalize player names across core pipeline artifacts.

Fixes:
  - player_id values with trailing .0
  - Unknown names where id-based names are available
  - diacritics/punctuation/name variants (e.g., Dončić vs Doncic, P.J. vs PJ)

Usage:
  python3 src/data_normalize/normalize_player_names.py
"""

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.player_eval.constants import HISTORICAL_DIR, PROCESSED_DIR, ROOT_DIR
from src.utils.player_name_normalizer import (
    apply_player_name_normalization,
    build_player_name_maps,
)


def _apply_to_file(
    path: Path,
    id_col_candidates,
    name_col_candidates,
    id_to_name,
    key_to_name,
) -> dict:
    if not path.exists():
        return {"file": str(path), "updated": False, "reason": "missing"}

    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        return {"file": str(path), "updated": False, "reason": f"read_error: {exc}"}

    id_col = next((c for c in id_col_candidates if c in df.columns), None)
    name_col = next((c for c in name_col_candidates if c in df.columns), None)
    if not id_col or not name_col:
        return {"file": str(path), "updated": False, "reason": "id_or_name_col_not_found"}

    before_unknown = int(df[name_col].astype(str).str.strip().str.lower().eq("unknown").sum())
    before_unique = int(df[name_col].astype(str).nunique())

    norm = apply_player_name_normalization(
        df=df,
        player_id_col=id_col,
        player_name_col=name_col,
        id_to_name=id_to_name,
        key_to_name=key_to_name,
    )
    norm.to_parquet(path, index=False)

    after_unknown = int(norm[name_col].astype(str).str.strip().str.lower().eq("unknown").sum())
    after_unique = int(norm[name_col].astype(str).nunique())

    return {
        "file": str(path),
        "updated": True,
        "id_col": id_col,
        "name_col": name_col,
        "rows": int(len(norm)),
        "unknown_before": before_unknown,
        "unknown_after": after_unknown,
        "unique_names_before": before_unique,
        "unique_names_after": after_unique,
    }


def main() -> None:
    source_specs = [
        (HISTORICAL_DIR / "players.parquet", ["id", "player_id"], ["full_name", "player_name"], 1),
        (PROCESSED_DIR / "player_archetypes.parquet", ["PLAYER_ID", "player_id"], ["PLAYER_NAME", "player_name"], 2),
        (HISTORICAL_DIR / "complete_player_season_stats.parquet", ["PLAYER_ID", "player_id"], ["PLAYER_NAME", "player_name"], 2),
        (PROCESSED_DIR / "player_rapm.parquet", ["player_id", "PLAYER_ID"], ["player_name", "PLAYER_NAME"], 3),
        (PROCESSED_DIR / "bke" / "bke_v28_decomposition.parquet", ["player_id", "PLAYER_ID"], ["player_name", "PLAYER_NAME"], 3),
        (PROCESSED_DIR / "player_profiles_advanced.parquet", ["player_id", "PLAYER_ID"], ["player_name", "PLAYER_NAME"], 3),
    ]

    id_to_name, key_to_name = build_player_name_maps(source_specs)

    outputs_dir = PROCESSED_DIR
    pd.DataFrame(
        [{"player_id": pid, "player_name": pname} for pid, pname in sorted(id_to_name.items())]
    ).to_parquet(outputs_dir / "player_name_id_map.parquet", index=False)
    pd.DataFrame(
        [{"name_key": key, "preferred_name": val} for key, val in sorted(key_to_name.items())]
    ).to_parquet(outputs_dir / "player_name_alias_map.parquet", index=False)

    targets = [
        PROCESSED_DIR / "bke" / "bke_v28_decomposition.parquet",
        PROCESSED_DIR / "player_rapm.parquet",
        PROCESSED_DIR / "player_archetypes.parquet",
        PROCESSED_DIR / "player_profiles_advanced.parquet",
        PROCESSED_DIR / "modeling_inputs_all.parquet",
        HISTORICAL_DIR / "complete_player_season_stats.parquet",
        PROCESSED_DIR / "player_eval" / "player_impact_profiles.parquet",
        PROCESSED_DIR / "player_eval" / "minute_model_predictions_v2.parquet",
        ROOT_DIR / "aggregate" / "player_profile_aggregate.parquet",
    ]

    results = []
    for target in targets:
        result = _apply_to_file(
            path=target,
            id_col_candidates=["player_id", "PLAYER_ID", "id"],
            name_col_candidates=["player_name", "PLAYER_NAME", "full_name", "display_first_last"],
            id_to_name=id_to_name,
            key_to_name=key_to_name,
        )
        results.append(result)
        print(result)

    report = {
        "id_map_size": int(len(id_to_name)),
        "alias_map_size": int(len(key_to_name)),
        "targets": results,
    }

    report_path = ROOT_DIR / "reports" / "player_name_normalization_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
