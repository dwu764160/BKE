"""
src/utils/player_name_normalizer.py
=============================================================================
Player name normalization utilities. Handles diacritics, suffixes, punctuation,
and common name variants for cross-source matching.
=============================================================================
"""
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


UNKNOWN_TOKENS = {
    "",
    "unknown",
    "nan",
    "none",
    "null",
    "na",
    "n/a",
}


def normalize_player_id_value(value) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def normalize_player_id_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def _strip_diacritics(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def canonical_name_key(name: str) -> str:
    if name is None:
        return ""
    text = _strip_diacritics(str(name)).lower().strip()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def _is_unknown_name(name: Optional[str]) -> bool:
    if name is None:
        return True
    return str(name).strip().lower() in UNKNOWN_TOKENS


_MANUAL_ALIAS_PREFERRED = {
    "pjtucker": "P.J. Tucker",
    "pjwashington": "P.J. Washington",
    "tjmcconnell": "T.J. McConnell",
    "ajgreen": "AJ Green",
    "ajgriffin": "AJ Griffin",
    "cjmccollum": "CJ McCollum",
    "oganunoby": "OG Anunoby",
    "herbertjones": "Herb Jones",
    "nikolajokic": "Nikola Jokić",
    "lukadoncic": "Luka Dončić",
}


def _extract_id_name_rows(
    df: pd.DataFrame,
    id_candidates: Sequence[str],
    name_candidates: Sequence[str],
) -> pd.DataFrame:
    id_col = next((c for c in id_candidates if c in df.columns), None)
    name_col = next((c for c in name_candidates if c in df.columns), None)
    if not id_col or not name_col:
        return pd.DataFrame(columns=["player_id", "player_name"])

    out = df[[id_col, name_col]].copy()
    out.columns = ["player_id", "player_name"]
    out["player_id"] = normalize_player_id_series(out["player_id"])
    out["player_name"] = out["player_name"].astype(str).str.strip()
    out = out[~out["player_name"].str.lower().isin(UNKNOWN_TOKENS)]
    out = out[out["player_id"].ne("")]
    return out.drop_duplicates()


def build_player_name_maps(
    source_specs: Iterable[Tuple[Path, Sequence[str], Sequence[str], int]],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Build (id_to_name, canonical_key_to_name) maps from multiple sources.

    source_specs: iterable of tuples
        (path, id_column_candidates, name_column_candidates, priority)
    Lower priority value wins when conflicts exist.
    """
    id_to_name: Dict[str, Tuple[int, str]] = {}
    key_to_name: Dict[str, Tuple[int, str]] = {}

    for path, id_candidates, name_candidates, priority in source_specs:
        if not Path(path).exists():
            continue
        try:
            df = pd.read_parquet(path)
        except Exception:
            continue

        rows = _extract_id_name_rows(df, id_candidates, name_candidates)
        if rows.empty:
            continue

        for _, row in rows.iterrows():
            player_id = str(row["player_id"])
            player_name = str(row["player_name"]).strip()
            if not player_id or _is_unknown_name(player_name):
                continue

            prev = id_to_name.get(player_id)
            if prev is None or priority < prev[0]:
                id_to_name[player_id] = (priority, player_name)

            key = canonical_name_key(player_name)
            if not key:
                continue
            prev_key = key_to_name.get(key)
            if prev_key is None or priority < prev_key[0]:
                key_to_name[key] = (priority, player_name)

    # Manual alias overrides
    for key, preferred in _MANUAL_ALIAS_PREFERRED.items():
        key_to_name[key] = (0, preferred)

    id_map_final = {k: v for k, (_, v) in id_to_name.items()}
    key_map_final = {k: v for k, (_, v) in key_to_name.items()}
    return id_map_final, key_map_final


def apply_player_name_normalization(
    df: pd.DataFrame,
    player_id_col: str,
    player_name_col: str,
    id_to_name: Dict[str, str],
    key_to_name: Dict[str, str],
) -> pd.DataFrame:
    out = df.copy()
    if player_id_col not in out.columns:
        return out

    out[player_id_col] = normalize_player_id_series(out[player_id_col])
    if player_name_col not in out.columns:
        out[player_name_col] = ""

    out[player_name_col] = out[player_name_col].astype(str).str.strip()

    # First pass: canonicalize existing names by alias key.
    existing_keys = out[player_name_col].map(canonical_name_key)
    out[player_name_col] = existing_keys.map(key_to_name).fillna(out[player_name_col])

    # Second pass: authoritative id-based mapping.
    mapped = out[player_id_col].map(id_to_name)
    use_mapped = mapped.notna() & (
        out[player_name_col].str.lower().isin(UNKNOWN_TOKENS)
        | out[player_name_col].eq("")
        | (mapped.map(canonical_name_key) != out[player_name_col].map(canonical_name_key))
    )
    out.loc[use_mapped, player_name_col] = mapped[use_mapped]

    # Final cleanup.
    out[player_name_col] = out[player_name_col].astype(str).str.strip()
    out.loc[out[player_name_col].str.lower().isin(UNKNOWN_TOKENS), player_name_col] = "Unknown"
    return out
