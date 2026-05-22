"""
src/simulation/train_minute_model.py
=============================================================================
Step 2 — Minutes Per Game (MPG) Prediction Model (v3 - Temporal)

Architecture:
  - Reads from the player profile aggregate (aggregate/player_profile_aggregate.parquet)
  - Target: Season N+1 actual MPG = MIN / GP
  - Features: Season N impact quality, behavioral rates, archetypes,
    player context (age, bio, salary), team/scheme context.
  - Model: sklearn RidgeCV (temporal split)
  - Rookie Model: Separate lookup table based on draft position
  - Validation: Temporal holdout on latest season transition
  - Post-processing: optional per-team normalization (team MPG sum ≈ 240)

Output:
  data/processed/player_eval/minute_model_v2.pkl
  data/processed/player_eval/minute_model_predictions_v2.parquet
  reports/player_eval_step2_minute_model_validation.json

Usage:
  python3 src/simulation/train_minute_model.py
=============================================================================
"""

import json
import pickle
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GridSearchCV
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.player_eval.constants import (
    MINUTE_MODEL_PATH,
    MINUTE_PREDICTIONS_PATH,
    PROFILE_AGGREGATE_PATH,
    RANDOM_SEED,
    STEP2_VALIDATION_REPORT,
)

# Team constraint: 5 players × 48 min = 240 min per game
TEAM_MPG_TARGET = 240.0
MPG_CAP = 42.0    # no player should exceed ~42 MPG
MPG_FLOOR = 0.0


def _season_key(series: pd.Series) -> pd.Series:
    """Convert season string '2023-24' to integer 2023."""
    return series.astype(str).str.slice(0, 4).astype(int)


def _first_existing_numeric(df: pd.DataFrame, candidates: List[str]) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def _normalize_team_mpg(frame: pd.DataFrame, pred_col: str) -> pd.Series:
    """
    Soft team normalization: scale predicted MPGs so the team sum
    approximates 240 minutes per game.
    """
    output = pd.Series(index=frame.index, dtype=float)

    team_col = None
    for c in frame.columns:
        if "team_abbreviation" in c.lower() and "rank" not in c.lower():
            team_col = c
            break
    if not team_col:
        return frame[pred_col].copy()

    # use season_y since that is the target season
    season_col = "season_y" if "season_y" in frame.columns else "season"

    for (_, _), idx in frame.groupby([season_col, team_col]).groups.items():
        vals = frame.loc[idx, pred_col].astype(float).clip(lower=MPG_FLOOR).copy()
        total = vals.sum()
        if total <= 0:
            vals[:] = TEAM_MPG_TARGET / len(vals)
        else:
            scale = TEAM_MPG_TARGET / total
            vals = vals * scale

        for _ in range(5):
            capped = vals.clip(upper=MPG_CAP)
            if np.allclose(capped.values, vals.values):
                vals = capped
                break
            excess = vals.sum() - capped.sum()
            free_mask = capped < MPG_CAP
            if free_mask.sum() == 0 or excess <= 1e-12:
                vals = capped
                break
            free_vals = capped[free_mask]
            capped.loc[free_mask] = free_vals + excess * (free_vals / free_vals.sum())
            vals = capped

        output.loc[idx] = vals
    return output


def build_temporal_training_set(agg: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Builds the Season N -> N+1 temporal training set."""
    df = agg.copy()
    df["season_int"] = _season_key(df["season"])
    df["_min"] = _first_existing_numeric(df, ["MIN", "min"])
    df["_gp"] = _first_existing_numeric(df, ["GP", "gp"])
    df["mpg"] = df["_min"] / df["_gp"].replace(0, np.nan)

    # Team context for N
    team_col = None
    for c in df.columns:
        if "team_abbreviation" in c.lower() and "rank" not in c.lower() and "pec" not in c.lower():
            team_col = c
            break
    if not team_col and "pec_team_abbreviation" in df.columns:
        team_col = "pec_team_abbreviation"

    if team_col and "pec_impact_bke" in df.columns:
        df["team_ctx_bke_rank"] = df.groupby(["season", team_col])["pec_impact_bke"].transform(
            lambda x: x.rank(ascending=False, method="min")
        )
    else:
        df["team_ctx_bke_rank"] = np.nan

    # Identify rookies
    df["is_rookie"] = (df["pec_experience_years"].fillna(0) == 0)

    # Split into N and N+1
    df_n = df.copy()
    df_n1 = df.copy()

    df_n["join_season"] = df_n["season_int"] + 1
    df_n1["join_season"] = df_n1["season_int"]

    temporal = pd.merge(
        df_n, df_n1,
        left_on=["player_id", "join_season"],
        right_on=["player_id", "join_season"],
        suffixes=("_x", "_y")
    )

    temporal["target_mpg"] = temporal["mpg_y"]

    # Filter transitions
    valid = temporal["target_mpg"].notna() & (temporal["_min_y"] > 0) & (temporal["_gp_y"] >= 5) & \
            (temporal["_min_x"] > 0) & (temporal["_gp_x"] >= 5)
    temporal = temporal[valid].copy()

    # Rookies
    valid_rookies = df["is_rookie"] & df["mpg"].notna() & (df["_min"] > 0) & (df["_gp"] >= 5)
    rookies = df[valid_rookies].copy()

    return temporal, rookies


def _build_rookie_model(rookies: pd.DataFrame) -> Dict[str, float]:
    """Fit a lookup table for rookie expected MPG based on draft position."""
    if "draft_round" not in rookies.columns or "draft_pick_overall" not in rookies.columns:
        return {"default": 12.0}

    rookies["draft_round_num"] = pd.to_numeric(rookies["draft_round"], errors="coerce")
    rookies["pick_overall_num"] = pd.to_numeric(rookies["draft_pick_overall"], errors="coerce")

    def categorize(row):
        r = row["draft_round_num"]
        p = row["pick_overall_num"]
        if pd.isna(r) or pd.isna(p):
            return "undrafted"
        if r == 1:
            if p <= 5: return "r1_1_5"
            elif p <= 15: return "r1_6_15"
            else: return "r1_16_30"
        elif r == 2:
            return "r2"
        return "undrafted"

    rookies["draft_cat"] = rookies.apply(categorize, axis=1)
    means = rookies.groupby("draft_cat")["mpg"].mean()

    return {
        "r1_1_5": round(float(means.get("r1_1_5", 24.0)), 2),
        "r1_6_15": round(float(means.get("r1_6_15", 20.0)), 2),
        "r1_16_30": round(float(means.get("r1_16_30", 16.0)), 2),
        "r2": round(float(means.get("r2", 8.0)), 2),
        "undrafted": round(float(means.get("undrafted", 5.0)), 2),
        "default": round(float(means.mean()) if not means.empty else 12.0, 2)
    }


def _extract_features(temporal: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
    temporal["prior_mpg"] = temporal["mpg_x"]
    temporal["prior_bke"] = temporal["pec_impact_bke_x"]
    temporal["prior_orapm"] = temporal["pec_impact_orapm_x"]
    temporal["prior_drapm"] = temporal["pec_impact_drapm_x"]
    temporal["age"] = temporal["pec_age_y"]
    temporal["prior_salary"] = temporal["pec_salary_x"]
    temporal["prior_usage"] = temporal["pec_behavioral_usage_x"]
    temporal["prior_3pt_rate"] = temporal["pec_behavioral_three_point_rate_x"]
    temporal["prior_ast_rate"] = temporal["pec_behavioral_assist_rate_x"]
    temporal["draft_position"] = pd.to_numeric(temporal["draft_pick_overall_x"], errors="coerce").fillna(61)
    temporal["experience_years"] = temporal["pec_experience_years_y"]
    temporal["team_ctx_bke_rank"] = temporal["team_ctx_bke_rank_x"]
    temporal["prior_gp_fraction"] = temporal["_gp_x"] / 82.0

    arche_cols = [c for c in temporal.columns if c.startswith("pec_off_prob_emb_") and c.endswith("_x")]
    if arche_cols:
        top_3 = temporal[arche_cols].mean().nlargest(3).index.tolist()
        for i, c in enumerate(top_3):
            temporal[f"prior_archetype_emb_{i+1}"] = temporal[c]

    features = [
        "prior_mpg", "prior_bke", "prior_orapm", "prior_drapm", "age",
        "prior_salary", "prior_usage", "prior_3pt_rate", "prior_ast_rate",
        "draft_position", "experience_years", "team_ctx_bke_rank", "prior_gp_fraction"
    ]
    if arche_cols:
        features.extend([f"prior_archetype_emb_{i+1}" for i in range(len(top_3))])

    return features, temporal


def main() -> None:
    if not PROFILE_AGGREGATE_PATH.exists():
        raise FileNotFoundError(f"Missing aggregate: {PROFILE_AGGREGATE_PATH}")

    agg = pd.read_parquet(PROFILE_AGGREGATE_PATH)
    
    temporal, rookies = build_temporal_training_set(agg)
    print(f"Temporal transitions: {len(temporal)}")
    print(f"Rookies identified: {len(rookies)}")

    rookie_lookup = _build_rookie_model(rookies)
    print(f"Rookie Lookup Table: {rookie_lookup}")

    feature_cols, X_full = _extract_features(temporal)

    for c in feature_cols:
        med = X_full[c].median()
        X_full[c] = X_full[c].fillna(med if not pd.isna(med) else 0.0)

    X = X_full[feature_cols].copy()
    y = X_full["target_mpg"]
    groups = X_full["season_x"].astype(str)

    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    n_splits = min(3, len(groups.unique()))
    cv = GroupKFold(n_splits=n_splits)

    # ── Temporal holdout (latest season) ──
    seasons_sorted = sorted(groups.unique())
    latest_season = seasons_sorted[-1] if seasons_sorted else "N/A"

    train_mask = groups != latest_season
    test_mask = groups == latest_season

    pipeline = make_pipeline(StandardScaler(), Ridge())
    holdout_model = GridSearchCV(pipeline, param_grid={'ridge__alpha': alphas}, cv=cv, scoring='neg_mean_absolute_error')
    holdout_model.fit(X[train_mask], y[train_mask], groups=groups[train_mask])

    holdout_preds = holdout_model.predict(X[test_mask]).clip(min=MPG_FLOOR, max=MPG_CAP)
    holdout_mae = mean_absolute_error(y[test_mask], holdout_preds)
    holdout_rmse = np.sqrt(mean_squared_error(y[test_mask], holdout_preds))
    holdout_r2 = r2_score(y[test_mask], holdout_preds)

    print(f"\nHoldout ({latest_season} predicting N+1): MAE={holdout_mae:.2f}, RMSE={holdout_rmse:.2f}, R²={holdout_r2:.3f}")

    # ── Full fit ──
    final_model = GridSearchCV(pipeline, param_grid={'ridge__alpha': alphas}, cv=cv, scoring='neg_mean_absolute_error')
    final_model.fit(X, y, groups=groups)
    print(f"Selected Alpha: {final_model.best_params_['ridge__alpha']}")
    final_estimator = final_model.best_estimator_

    preds_all = X_full.copy()
    preds_all["pred_mpg_raw"] = final_estimator.predict(X).clip(min=MPG_FLOOR, max=MPG_CAP)
    preds_all["pred_mpg_team_norm"] = _normalize_team_mpg(preds_all, "pred_mpg_raw")

    # ── Export ──
    export_cols = ["player_id", "season_y", "target_mpg", "pred_mpg_raw", "pred_mpg_team_norm"]
    if "player_name_y" in preds_all.columns:
        preds_all["player_name"] = preds_all["player_name_y"]
        export_cols.insert(1, "player_name")
    
    export_df = preds_all[[c for c in export_cols if c in preds_all.columns]].copy()
    export_df = export_df.rename(columns={"season_y": "season"})
    
    MINUTE_PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_parquet(MINUTE_PREDICTIONS_PATH, index=False)

    MINUTE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MINUTE_MODEL_PATH, "wb") as f:
        pickle.dump({
            "model": final_model,
            "feature_cols": feature_cols,
            "rookie_lookup": rookie_lookup,
            "version": "v3",
            "type": "temporal_ridge"
        }, f)

    # ── Validation report ──
    importances = {k: round(float(v), 4) for k, v in zip(feature_cols, final_estimator.named_steps['ridge'].coef_)}
    sorted_importances = dict(sorted(importances.items(), key=lambda item: abs(item[1]), reverse=True))

    report = {
        "model_version": "v3_temporal_ridge",
        "target": "season_N+1_MPG",
        "training_transitions": int(len(X)),
        "rookie_lookup": rookie_lookup,
        "holdout_latest_season": {
            "prior_season": latest_season,
            "n_test": int(test_mask.sum()),
            "mae_mpg": round(float(holdout_mae), 3),
            "rmse_mpg": round(float(holdout_rmse), 3),
            "r2": round(float(holdout_r2), 3)
        },
        "model_params": {
            "type": "RidgeCV",
            "selected_alpha": float(final_model.best_params_['ridge__alpha']),
            "features": feature_cols
        },
        "feature_coefficients": sorted_importances
    }
    
    STEP2_VALIDATION_REPORT.write_text(json.dumps(report, indent=2))
    print(f"\nSaved model: {MINUTE_MODEL_PATH}")
    print(f"Saved validation: {STEP2_VALIDATION_REPORT}")

if __name__ == "__main__":
    main()
