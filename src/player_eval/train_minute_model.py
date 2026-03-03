"""
src/player_eval/train_minute_model.py
=============================================================================
Step 2 — Minutes Per Game (MPG) Prediction Model (v2)

Architecture:
  - Reads from the player profile aggregate (aggregate/player_profile_aggregate.parquet)
  - Target: actual MPG = MIN / GP
  - Features: impact quality, behavioral rates, archetypes, playtype mix,
    player context (age, bio, salary), team/scheme context, prior-season lags.
    Volume stats (MIN, GP, POSS) are EXCLUDED to avoid circularity.
  - Model: sklearn GradientBoostingRegressor (deterministic, seed=42)
  - Validation: GroupKFold by season + temporal holdout on latest season
  - Post-processing: optional per-team normalization (team MPG sum ≈ 240)

ML Concepts Used:
  1. Gradient Boosted Decision Trees — sequential ensemble of shallow trees
  2. GroupKFold Cross-Validation — group by season to prevent temporal leakage
  3. Feature Engineering — rates, lags, team context (no volume leakage)
  4. Median Imputation — robust to outliers for missing values
  5. L2 Regularization — via tree depth + minimum samples constraints
  6. Temporal Holdout — latest season withheld for unbiased evaluation
  7. Team Normalization — optional constraint: team MPG sum ≈ 240

Output:
  data/processed/player_eval/minute_model_v2.pkl
  data/processed/player_eval/minute_model_predictions_v2.parquet
  reports/player_eval_step2_minute_model_validation.json

Usage:
  python3 src/player_eval/train_minute_model.py
=============================================================================
"""

import json
import pickle
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
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
    approximates 240 minutes per game (proportional scaling with cap/floor).
    """
    output = pd.Series(index=frame.index, dtype=float)

    # Identify team column
    team_col = None
    for c in frame.columns:
        if "team_abbreviation" in c.lower() and "rank" not in c.lower():
            team_col = c
            break
    if not team_col:
        return frame[pred_col].copy()

    for (_, _), idx in frame.groupby(["season", team_col]).groups.items():
        vals = frame.loc[idx, pred_col].astype(float).clip(lower=MPG_FLOOR).copy()
        total = vals.sum()
        if total <= 0:
            vals[:] = TEAM_MPG_TARGET / len(vals)
        else:
            scale = TEAM_MPG_TARGET / total
            vals = vals * scale

        # Cap at MPG_CAP, redistribute excess
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


def _team_spearman(df: pd.DataFrame, target_col: str, pred_col: str) -> float:
    """Average within-team Spearman correlation."""
    team_col = None
    for c in df.columns:
        if "team_abbreviation" in c.lower() and "rank" not in c.lower():
            team_col = c
            break
    if not team_col:
        return float("nan")

    values = []
    for _, grp in df.groupby(["season", team_col]):
        if len(grp) < 3:
            continue
        corr = spearmanr(grp[target_col], grp[pred_col], nan_policy="omit").correlation
        if not np.isnan(corr):
            values.append(float(corr))
    return float(np.mean(values)) if values else float("nan")


def _select_features(df: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
    """
    Select features from the aggregate. Strictly NO volume stats (MIN, GP, POSS)
    to avoid target leakage. Use impact quality, behavioral rates, archetypes,
    context, and prior-season lags.
    """
    feature_cols = []

    # ── Impact quality metrics (pec_ from Step 1) ──────────────────────
    impact_cols = [c for c in df.columns if c.startswith("pec_impact_")]
    feature_cols.extend(impact_cols)

    # ── Behavioral rates ───────────────────────────────────────────────
    behavioral_cols = [c for c in df.columns if c.startswith("pec_behavioral_")]
    feature_cols.extend(behavioral_cols)

    # ── Archetype probabilities ────────────────────────────────────────
    arche_cols = [c for c in df.columns if c.startswith("pec_off_prob_") or c.startswith("pec_def_prob_")]
    feature_cols.extend(arche_cols)

    # ── Playtype distribution ──────────────────────────────────────────
    playtype_cols = [c for c in df.columns if c.startswith("pec_playtype_")]
    feature_cols.extend(playtype_cols)

    # ── Context ────────────────────────────────────────────────────────
    context_cols = [
        "pec_age", "pec_height_inches", "pec_weight_lbs",
        "pec_experience_years", "pec_salary",
        "pec_on_off_diff", "pec_scheme_stability_index",
        "pec_compression_flag", "pec_defensive_shrinkage_lambda",
    ]
    feature_cols.extend([c for c in context_cols if c in df.columns])

    # ── Prior season lag features ──────────────────────────────────────
    lag_cols = [c for c in df.columns if c.startswith("lag_")]
    feature_cols.extend(lag_cols)

    # ── Team context features ──────────────────────────────────────────
    team_ctx_cols = [c for c in df.columns if c.startswith("team_ctx_")]
    feature_cols.extend(team_ctx_cols)

    # Ensure all exist in DataFrame
    feature_cols = [c for c in feature_cols if c in df.columns]
    feature_cols = sorted(set(feature_cols))

    return feature_cols, df


def _build_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add prior-season lag features for key metrics."""
    df = df.copy()
    df["_season_int"] = _season_key(df["season"])
    df = df.sort_values(["player_id", "_season_int"]).reset_index(drop=True)

    lag_sources = {
        "lag_mpg": "target_mpg",
        "lag_impact_bke": "pec_impact_bke",
        "lag_impact_orapm": "pec_impact_orapm",
        "lag_usage": "pec_behavioral_usage",
        "lag_salary": "pec_salary",
    }

    for lag_name, source_col in lag_sources.items():
        if source_col in df.columns:
            df[lag_name] = df.groupby("player_id")[source_col].shift(1)
        else:
            df[lag_name] = np.nan

    df = df.drop(columns=["_season_int"])
    return df


def _build_team_context(df: pd.DataFrame) -> pd.DataFrame:
    """Add team-level context features (competition for minutes)."""
    df = df.copy()

    team_col = None
    for c in df.columns:
        if "team_abbreviation" in c.lower() and "rank" not in c.lower() and "pec" not in c.lower():
            team_col = c
            break
    if not team_col:
        # Try pec version
        if "pec_team_abbreviation" in df.columns:
            team_col = "pec_team_abbreviation"
        else:
            return df

    # Number of players on team roster
    team_size = df.groupby(["season", team_col])["player_id"].transform("count")
    df["team_ctx_roster_size"] = team_size

    # Team average impact BKE
    if "pec_impact_bke" in df.columns:
        df["team_ctx_avg_bke"] = df.groupby(["season", team_col])["pec_impact_bke"].transform("mean")
        df["team_ctx_bke_rank"] = df.groupby(["season", team_col])["pec_impact_bke"].transform(
            lambda x: x.rank(ascending=False, method="min")
        )

    # Team average usage
    if "pec_behavioral_usage" in df.columns:
        df["team_ctx_avg_usage"] = df.groupby(["season", team_col])["pec_behavioral_usage"].transform("mean")

    return df


def _cross_validate(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series, meta: pd.DataFrame
) -> List[Dict]:
    """GroupKFold cross-validation grouped by season."""
    unique_groups = sorted(groups.unique().tolist())
    if len(unique_groups) < 2:
        return []

    n_splits = min(3, len(unique_groups))
    splitter = GroupKFold(n_splits=n_splits)
    folds = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups=groups), start=1):
        model = _build_regressor()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])

        preds = model.predict(X.iloc[test_idx])
        fold_df = meta.iloc[test_idx].copy()
        fold_df["pred_mpg"] = preds

        mae = mean_absolute_error(y.iloc[test_idx], preds)
        rmse = float(np.sqrt(mean_squared_error(y.iloc[test_idx], preds)))
        r2 = r2_score(y.iloc[test_idx], preds)
        sp = _team_spearman(fold_df, "target_mpg", "pred_mpg")

        folds.append({
            "fold": fold_idx,
            "test_seasons": sorted(fold_df["season"].astype(str).unique().tolist()),
            "n_test": int(len(test_idx)),
            "mae": float(mae),
            "rmse": rmse,
            "r2": float(r2),
            "team_spearman": float(sp) if not np.isnan(sp) else None,
        })
    return folds


def _build_regressor() -> GradientBoostingRegressor:
    """Deterministic Gradient Boosting Regressor with production-grade params."""
    return GradientBoostingRegressor(
        random_state=RANDOM_SEED,
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.85,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features=0.7,
        validation_fraction=0.15,
        n_iter_no_change=30,
        tol=1e-5,
    )


def main() -> None:
    # ── Load aggregate ─────────────────────────────────────────────────
    if not PROFILE_AGGREGATE_PATH.exists():
        raise FileNotFoundError(
            f"Missing aggregate: {PROFILE_AGGREGATE_PATH}.\n"
            "Run: python3 src/profile_aggregate/build_profile_aggregate.py"
        )

    agg = pd.read_parquet(PROFILE_AGGREGATE_PATH).copy()
    agg["season"] = agg["season"].astype(str)
    agg["player_id"] = agg["player_id"].astype(str)

    # ── Compute target: MPG = MIN / GP ─────────────────────────────────
    agg["_min"] = _first_existing_numeric(agg, ["MIN", "min"])
    agg["_gp"] = _first_existing_numeric(agg, ["GP", "gp"])
    agg["target_mpg"] = agg["_min"] / agg["_gp"].replace(0, np.nan)

    # Filter: must have valid target and at least some minutes
    valid = agg["target_mpg"].notna() & (agg["_min"] > 0) & (agg["_gp"] >= 5)
    df = agg[valid].copy()
    print(f"Training data: {len(df)} rows after filtering (GP >= 5, MIN > 0)")

    # ── Build lag and team context features ────────────────────────────
    df = _build_lag_features(df)
    df = _build_team_context(df)

    # ── Select features ────────────────────────────────────────────────
    feature_cols, df = _select_features(df)
    print(f"Features selected: {len(feature_cols)}")

    X = df[feature_cols].copy()
    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        med = X[col].median()
        X[col] = X[col].fillna(med if not np.isnan(med) else 0.0)
    X = X.fillna(0.0)

    y = df["target_mpg"].astype(float)
    groups = df["season"].astype(str)

    # ── Cross-validation ───────────────────────────────────────────────
    meta_cols = ["season", "player_id"]
    # Add team column to meta
    team_col = None
    for c in df.columns:
        if "team_abbreviation" in c.lower() and "rank" not in c.lower() and "pec" not in c.lower():
            team_col = c
            break
    if team_col:
        meta_cols.append(team_col)
    meta = df[meta_cols + ["target_mpg"]].copy()
    cv_folds = _cross_validate(X, y, groups, meta)
    print(f"Cross-validation folds: {len(cv_folds)}")
    for f in cv_folds:
        print(f"  Fold {f['fold']}: MAE={f['mae']:.2f}, RMSE={f['rmse']:.2f}, R²={f['r2']:.3f}, Spearman={f.get('team_spearman', 'N/A')}")

    # ── Temporal holdout (latest season) ───────────────────────────────
    latest_season = sorted(df["season"].unique().tolist(), key=lambda s: int(s[:4]))[-1]
    train_mask = df["season"] != latest_season
    test_mask = df["season"] == latest_season

    holdout_model = _build_regressor()
    holdout_model.fit(X[train_mask], y[train_mask])

    holdout = df[test_mask].copy()
    holdout["pred_mpg_raw"] = holdout_model.predict(X[test_mask])
    holdout["pred_mpg_raw"] = holdout["pred_mpg_raw"].clip(lower=MPG_FLOOR, upper=MPG_CAP)

    holdout_mae = mean_absolute_error(holdout["target_mpg"], holdout["pred_mpg_raw"])
    holdout_rmse = float(np.sqrt(mean_squared_error(holdout["target_mpg"], holdout["pred_mpg_raw"])))
    holdout_r2 = r2_score(holdout["target_mpg"], holdout["pred_mpg_raw"])
    holdout_team_sp = _team_spearman(holdout, "target_mpg", "pred_mpg_raw")
    print(f"\nHoldout ({latest_season}): MAE={holdout_mae:.2f}, RMSE={holdout_rmse:.2f}, R²={holdout_r2:.3f}, Spearman={holdout_team_sp:.4f}")

    # ── Full fit (all data) ────────────────────────────────────────────
    final_model = _build_regressor()
    final_model.fit(X, y)

    pred_all = df.copy()
    pred_all["pred_mpg_raw"] = final_model.predict(X)
    pred_all["pred_mpg_raw"] = pred_all["pred_mpg_raw"].clip(lower=MPG_FLOOR, upper=MPG_CAP)

    # Optional team normalization
    pred_all["pred_mpg_team_norm"] = _normalize_team_mpg(pred_all, "pred_mpg_raw")

    # ── Feature importance ─────────────────────────────────────────────
    importances = pd.Series(final_model.feature_importances_, index=feature_cols).sort_values(ascending=False)

    # ── Export predictions ─────────────────────────────────────────────
    export_cols = ["player_id", "season"]
    if "player_name" in pred_all.columns:
        export_cols.append("player_name")
    elif "pec_player_name" in pred_all.columns:
        pred_all["player_name"] = pred_all["pec_player_name"]
        export_cols.append("player_name")
    if team_col and team_col in pred_all.columns:
        export_cols.append(team_col)
    export_cols.extend(["target_mpg", "pred_mpg_raw", "pred_mpg_team_norm"])
    # Add key impact features for interpretability
    for c in ["pec_impact_bke", "pec_behavioral_usage", "pec_impact_orapm", "pec_impact_drapm"]:
        if c in pred_all.columns:
            export_cols.append(c)

    pred_export = pred_all[[c for c in export_cols if c in pred_all.columns]].copy()
    pred_export.to_parquet(MINUTE_PREDICTIONS_PATH, index=False)

    # ── Save model ─────────────────────────────────────────────────────
    with open(MINUTE_MODEL_PATH, "wb") as handle:
        pickle.dump({
            "model": final_model,
            "feature_cols": feature_cols,
            "seed": RANDOM_SEED,
            "target": "mpg",
            "version": "v2",
        }, handle)

    # ── Residual analysis ──────────────────────────────────────────────
    residuals = pred_all["target_mpg"] - pred_all["pred_mpg_raw"]

    # ── Validation report ──────────────────────────────────────────────
    full_mae = mean_absolute_error(pred_all["target_mpg"], pred_all["pred_mpg_raw"])
    full_rmse = float(np.sqrt(mean_squared_error(pred_all["target_mpg"], pred_all["pred_mpg_raw"])))
    full_r2 = r2_score(pred_all["target_mpg"], pred_all["pred_mpg_raw"])
    full_sp = _team_spearman(pred_all, "target_mpg", "pred_mpg_raw")

    validation_report = {
        "model_version": "v2",
        "target": "minutes_per_game (MPG = MIN / GP)",
        "training_rows": int(len(df)),
        "feature_count": int(len(feature_cols)),
        "seasons": sorted(df["season"].astype(str).unique().tolist()),
        "model_type": "GradientBoostingRegressor (sklearn)",
        "model_params": {
            "n_estimators": 500,
            "max_depth": 4,
            "learning_rate": 0.03,
            "subsample": 0.85,
            "min_samples_split": 20,
            "min_samples_leaf": 10,
            "max_features": 0.7,
            "early_stopping": "n_iter_no_change=30",
        },
        "cross_validation": cv_folds,
        "holdout_latest_season": {
            "season": latest_season,
            "n_test": int(test_mask.sum()),
            "mae_mpg": float(holdout_mae),
            "rmse_mpg": float(holdout_rmse),
            "r2": float(holdout_r2),
            "team_spearman": float(holdout_team_sp) if not np.isnan(holdout_team_sp) else None,
        },
        "full_fit": {
            "mae_mpg": float(full_mae),
            "rmse_mpg": float(full_rmse),
            "r2": float(full_r2),
            "team_spearman": float(full_sp) if not np.isnan(full_sp) else None,
        },
        "residual_analysis": {
            "mean": float(residuals.mean()),
            "std": float(residuals.std()),
            "p5": float(residuals.quantile(0.05)),
            "p25": float(residuals.quantile(0.25)),
            "median": float(residuals.median()),
            "p75": float(residuals.quantile(0.75)),
            "p95": float(residuals.quantile(0.95)),
        },
        "target_distribution": {
            "mean": float(y.mean()),
            "median": float(y.median()),
            "std": float(y.std()),
            "min": float(y.min()),
            "max": float(y.max()),
        },
        "top_feature_importance": [
            {"feature": f, "importance": float(v)} for f, v in importances.head(25).items()
        ],
        "feature_categories": {
            "impact": [c for c in feature_cols if "impact" in c],
            "behavioral": [c for c in feature_cols if "behavioral" in c],
            "archetype": [c for c in feature_cols if "off_prob" in c or "def_prob" in c],
            "playtype": [c for c in feature_cols if "playtype" in c],
            "context": [c for c in feature_cols if c in ["pec_age", "pec_height_inches", "pec_weight_lbs",
                        "pec_experience_years", "pec_salary", "pec_on_off_diff", "pec_scheme_stability_index"]],
            "lag": [c for c in feature_cols if c.startswith("lag_")],
            "team_context": [c for c in feature_cols if c.startswith("team_ctx_")],
        },
        "design_notes": [
            "v2 redesign: predicts MPG (minutes per game) instead of minute share.",
            "Volume stats (MIN, GP, POSS) are EXCLUDED from features to avoid target leakage.",
            "Features are sourced from the profile aggregate via Step 1 curated impact/behavioral metrics.",
            "Prior-season lag features (MPG, BKE, ORAPM, usage, salary) capture temporal momentum.",
            "Team context features (roster size, avg BKE, rank) capture minute competition.",
            "GBR with early stopping (n_iter_no_change=30) prevents overfitting.",
            "Subsample=0.85 + max_features=0.7 add stochastic regularization.",
            "Optional team MPG normalization scales predictions to sum ≈ 240 per team-game.",
        ],
    }

    STEP2_VALIDATION_REPORT.write_text(json.dumps(validation_report, indent=2), encoding="utf-8")
    print(f"\nSaved model: {MINUTE_MODEL_PATH}")
    print(f"Saved predictions: {MINUTE_PREDICTIONS_PATH}")
    print(f"Saved validation: {STEP2_VALIDATION_REPORT}")
    print(json.dumps(validation_report, indent=2))


if __name__ == "__main__":
    main()
