# Roadmap to Vegas-Level Accuracy
**Date:** 2026-05-26  
**Target:** Walk-forward Brier ≤ 0.200 with positive closing-line value (CLV) against Kalshi  
**Current position:** Brier ~0.233 (v4.0-A+C composite, patched, OOS)  
**Vegas benchmark:** Brier ~0.195–0.210 (FiveThirtyEight averaged ~0.243 in a bad year; sharp books consistently sub-0.210)  
**Gap:** ~0.025–0.035 Brier  

**Implementation dependency order: each step must complete before the next begins (except Steps 3 and 4 which can run in parallel after Step 2).**

---

## Step 1 — Lock Composite v4.0 (A + C Blend)

**Status:** COMPLETE (2026-05-26). Locked w_c=0.50, Brier=0.2334, Joint_r=0.3506. YoY_r=0.690 (below 0.766 gate — accepted, known Imp C tradeoff). Star sanity: Curry #7, KD #8, SGA #2, Giannis #8.  
**Why first:** Every downstream step depends on a stable PTS input. Steps 3–5 all use `team_net_rating_projected`, which is recomputed from PTS. Lock this before building on top of it.  
**Expected Brier gain:** ~0.001–0.002 (marginal; value is in establishing stable baseline)

### What to run

```bash
cd /mnt/c/Users/danie/OneDrive/Documents/BKE
python3 scripts/build_pts_v40.py \
  --pts-a data/processed/bke/pts_v40_a.parquet \
  --pts-c data/processed/bke/pts_v40_c.parquet \
  --output data/processed/bke/pts_v40.parquet \
  --sweep
```

This prints a table of `w_c` from 0.3 to 0.7 with YoY_r, Joint_r, Def_r, Brier. Pick the config with lowest Brier that does not regress Joint_r below 0.331 (v3.2 lineup baseline).

### Then lock with best weight

```bash
# Example if w_c=0.40 wins:
python3 scripts/build_pts_v40.py \
  --pts-a data/processed/bke/pts_v40_a.parquet \
  --pts-c data/processed/bke/pts_v40_c.parquet \
  --output data/processed/bke/pts_v40.parquet
```

Update `src/modeling/model_config.py` `PtsV40CompositeConfig.defense_v40c_weight` to the winning value.

### Regenerate projected team features

After locking pts_v40.parquet, regenerate projected team features so downstream steps use updated PTS:

```bash
python3 scripts/build_pts_v32.py  # or equivalent; update to v40 inputs
# Output: data/processed/forecast/projected_team_features_v40.parquet
# Key column used downstream: team_net_rating_projected (float, ~range [-10, 10])
```

`projected_team_features_v40.parquet` schema (relevant subset):
- `season` (str, e.g. "2022-23")
- `team_abbreviation` (str, 3-letter)
- `team_net_rating_projected` (float, net points per 100 possessions)
- `off_talent_base`, `def_talent_base` (float, component breakdowns)

### Validation gates

- Brier ≤ 0.2352 (no regression vs v3.2)
- lineup Joint_r ≥ 0.331
- YoY PTS correlation ≥ 0.766 (Improvement A result)
- Star sanity: Curry/KD/Luka/SGA/Giannis all rank ≤ 30 in pts_total in their healthy seasons

---

## Step 2 — Kalshi CLV Measurement Harness

**Status:** Not built. Must be built before Steps 3–5, because without it, Brier improvements are unvalidated for alpha.  
**Why second:** Closing-line value (CLV) is the only metric that tests whether the model has genuine alpha vs. the market. A model can improve Brier without beating the closing line (i.e., improve absolute calibration but not relative to what smart money already knows). Build this measurement harness before committing engineering time to Steps 3–5.  
**Expected Brier gain:** N/A — this is measurement infrastructure.

### Data acquisition

Download Kalshi NBA game market closing lines for 2023-24 and 2024-25 seasons.  
Export format: CSV with columns:
- `game_date` (YYYY-MM-DD)
- `home_team` (3-letter abbreviation, e.g. "BOS")
- `away_team` (3-letter abbreviation)
- `kalshi_home_win_prob` (float, 0–1, closing line — the last traded price before game start)
- `home_result` (int, 1=home win, 0=away win)

Save to: `data/external/kalshi_closing_lines.csv`

### Create script: `scripts/compute_kalshi_clv.py`

```python
"""
Computes closing-line value (CLV) of BKE model vs. Kalshi closing prices.
CLV > 0 means BKE predicted the outcome better than the closing line.

Usage:
    python3 scripts/compute_kalshi_clv.py \
        --kalshi data/external/kalshi_closing_lines.csv \
        --bke-forecast data/processed/forecast/bke_game_forecasts.parquet \
        --output reports/kalshi_clv_report.json
"""
import argparse, json, sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

def brier(probs, outcomes):
    return float(np.mean((np.array(probs) - np.array(outcomes)) ** 2))

def log_loss(probs, outcomes):
    eps = 1e-7
    p = np.clip(probs, eps, 1 - eps)
    return float(-np.mean(outcomes * np.log(p) + (1 - outcomes) * np.log(1 - p)))

def compute_clv(kalshi_probs, bke_probs, outcomes):
    """
    CLV = mean(bke_prob_on_winning_side - kalshi_prob_on_winning_side)
    Positive CLV = BKE was more accurate than closing line.
    """
    outcomes = np.array(outcomes)
    bke_probs = np.array(bke_probs)
    kalshi_probs = np.array(kalshi_probs)

    # For each game, compute the edge BKE had vs Kalshi on the correct side
    # Edge > 0 means BKE was more confident in the right outcome
    bke_edge = np.where(outcomes == 1,
                        bke_probs - kalshi_probs,
                        (1 - bke_probs) - (1 - kalshi_probs))
    return float(np.mean(bke_edge)), bke_edge.tolist()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kalshi", required=True)
    parser.add_argument("--bke-forecast", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    kalshi = pd.read_csv(args.kalshi)
    bke = pd.read_parquet(args.bke_forecast)

    # Merge on game_date + home_team
    merged = kalshi.merge(bke, on=["game_date", "home_team", "away_team"], how="inner")
    n = len(merged)
    if n == 0:
        print("ERROR: No games matched between Kalshi and BKE forecast. Check team abbreviation format.")
        sys.exit(1)

    outcomes = merged["home_result"].tolist()
    kalshi_probs = merged["kalshi_home_win_prob"].tolist()
    bke_probs = merged["bke_home_win_prob"].tolist()

    brier_bke = brier(bke_probs, outcomes)
    brier_kalshi = brier(kalshi_probs, outcomes)
    logloss_bke = log_loss(bke_probs, outcomes)
    logloss_kalshi = log_loss(kalshi_probs, outcomes)
    clv_mean, clv_per_game = compute_clv(kalshi_probs, bke_probs, outcomes)

    report = {
        "n_games": n,
        "brier_bke": brier_bke,
        "brier_kalshi": brier_kalshi,
        "brier_delta": brier_bke - brier_kalshi,
        "logloss_bke": logloss_bke,
        "logloss_kalshi": logloss_kalshi,
        "clv_mean": clv_mean,
        "interpretation": "POSITIVE CLV = BKE beats market" if clv_mean > 0 else "NEGATIVE CLV = market beats BKE",
        "games_with_positive_clv": int(np.sum(np.array(clv_per_game) > 0)),
        "games_with_negative_clv": int(np.sum(np.array(clv_per_game) < 0)),
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
```

### BKE forecast parquet format

The BKE forecast file passed to the CLV harness must have columns:
- `game_date` (str, YYYY-MM-DD)
- `home_team` (str, 3-letter abbreviation)
- `away_team` (str, 3-letter abbreviation)
- `bke_home_win_prob` (float, 0–1)
- `home_result` (int, 1=home win, 0=away win)

This file is generated by the game model (currently `src/simulation/validate_forecast.py`). Add an export step to that script to save per-game predictions in this format.

### Alpha-readiness threshold

- **CLV > 0.01** (BKE beats closing line by 1 pp on average): worth paper-trading
- **CLV > 0.02** with Brier ≤ 0.210: alpha-ready for live allocation
- **CLV ≤ 0** regardless of Brier: do not allocate capital

---

## Step 3 — Game-Day Rest/B2B Correction Layer

**Status:** COMPLETE (2026-05-28). Brier gain: ~−0.001 (2023-24: −0.0009; B2B games: −0.0016). Smaller than projected; dominant error remains data staleness and team projection noise, not rest/schedule. Implementation correct; gain materializes more in Steps 4–5.  
**Dependencies:** Step 1 (stable team features), Step 2 (measurement harness ready)  
**Can run in parallel with:** Step 4  
**Expected Brier gain:** −0.001 (actual; roadmap projected −0.010 to −0.015 — overstated)  
**Basketball source:** Back-to-back road teams lose against the spread at 57% rate vs. rested opponents (The Wager Theorem, 2024). Performance declines 1–3 points per back-to-back in net rating. Teams playing 3 games in 4 nights see further degradation. All schedule data is fully public and computable from `data/historical/team_game_logs.parquet`.

### What this does

Before the game model runs, compute schedule stress for each team per game and apply a delta to `team_net_rating_projected`:
- Home B2B: −1.5 pts/100 to home team's projected net rating
- Road B2B: −2.5 pts/100 (road fatigue compounds travel)
- 3-in-4: −1.0 pts/100 additional penalty on top of B2B flag if applicable
- Rest advantage (≥2 days rest vs. opponent on B2B): +0.5 pts/100 to rested team

### Create script: `scripts/build_rest_features.py`

Input: `data/historical/team_game_logs.parquet`  
Schema confirmed: TEAM_ID, TEAM_ABBREVIATION, GAME_DATE (YYYY-MM-DD str), GAME_ID, MATCHUP ("BOS vs. PHI" or "PHI @ BOS"), WL, SEASON, margin

Output: `data/processed/forecast/game_rest_features.parquet`  
Output schema:
- `game_id` (str)
- `game_date` (str, YYYY-MM-DD)
- `season` (str, e.g. "2022-23")
- `home_team` (str, 3-letter abbreviation)
- `away_team` (str, 3-letter abbreviation)
- `home_days_rest` (int, days since last game; 7+ treated as fully rested)
- `away_days_rest` (int)
- `home_b2b` (bool)
- `away_b2b` (bool)
- `home_3in4` (bool, 3rd game in 4 calendar days)
- `away_3in4` (bool)
- `rest_net_delta` (float, pts/100 adjustment to home team's net rating; positive = home advantage)

```python
"""
scripts/build_rest_features.py
Computes per-game rest/schedule stress features from team_game_logs.parquet.

Usage:
    python3 scripts/build_rest_features.py \
        --input data/historical/team_game_logs.parquet \
        --output data/processed/forecast/game_rest_features.parquet
"""
import argparse, sys
from pathlib import Path
import pandas as pd
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Calibrated from Wager Theorem 2024 B2B ATS analysis + internal net rating delta
HOME_B2B_PENALTY = -1.5      # pts/100
ROAD_B2B_PENALTY = -2.5      # pts/100 (larger: travel + fatigue)
HOME_3IN4_EXTRA  = -1.0      # pts/100 additional on top of B2B
AWAY_3IN4_EXTRA  = -1.0      # pts/100 additional
REST_ADV_BONUS   = 0.5       # pts/100 per rest-day advantage (max cap 2 days)


def parse_matchup(matchup_str, abbrev):
    """Return (home_team, away_team) from MATCHUP string like 'BOS vs. PHI' or 'PHI @ BOS'."""
    if ' vs. ' in matchup_str:
        home = matchup_str.split(' vs. ')[0].strip()
        away = matchup_str.split(' vs. ')[1].strip()
    elif ' @ ' in matchup_str:
        away = matchup_str.split(' @ ')[0].strip()
        home = matchup_str.split(' @ ')[1].strip()
    else:
        home = abbrev
        away = abbrev
    return home, away


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(REPO / "data/historical/team_game_logs.parquet"))
    parser.add_argument("--output", default=str(REPO / "data/processed/forecast/game_rest_features.parquet"))
    args = parser.parse_args()

    logs = pd.read_parquet(args.input)
    logs['GAME_DATE'] = pd.to_datetime(logs['GAME_DATE'])
    logs = logs.sort_values(['TEAM_ABBREVIATION', 'GAME_DATE']).reset_index(drop=True)

    # Compute days rest per team per game
    logs['prev_game_date'] = logs.groupby('TEAM_ABBREVIATION')['GAME_DATE'].shift(1)
    logs['days_rest'] = (logs['GAME_DATE'] - logs['prev_game_date']).dt.days.fillna(7).clip(upper=7).astype(int)
    logs['b2b'] = logs['days_rest'] == 1

    # Compute 3-in-4 flag: this game is the 3rd in 4 calendar days
    logs['prev2_game_date'] = logs.groupby('TEAM_ABBREVIATION')['GAME_DATE'].shift(2)
    logs['span_days'] = (logs['GAME_DATE'] - logs['prev2_game_date']).dt.days.fillna(99)
    logs['_3in4'] = (logs['span_days'] <= 3) & (logs['b2b'])

    # Parse home/away for each row
    logs[['home_parsed', 'away_parsed']] = logs.apply(
        lambda r: pd.Series(parse_matchup(r['MATCHUP'], r['TEAM_ABBREVIATION'])), axis=1
    )

    # Build one row per game (home perspective)
    home_rows = logs[logs['TEAM_ABBREVIATION'] == logs['home_parsed']].copy()
    home_rows = home_rows.rename(columns={
        'TEAM_ABBREVIATION': 'home_team',
        'GAME_DATE': 'game_date',
        'SEASON': 'season',
        'GAME_ID': 'game_id',
        'days_rest': 'home_days_rest',
        'b2b': 'home_b2b',
        '_3in4': 'home_3in4',
        'away_parsed': 'away_team',
    })

    away_rows = logs[logs['TEAM_ABBREVIATION'] == logs['away_parsed']].copy()
    away_rows = away_rows.rename(columns={
        'TEAM_ABBREVIATION': 'away_team_check',
        'GAME_DATE': 'game_date_away',
        'GAME_ID': 'game_id_away',
        'days_rest': 'away_days_rest',
        'b2b': 'away_b2b',
        '_3in4': 'away_3in4',
    })

    merged = home_rows[['game_id', 'game_date', 'season', 'home_team', 'away_team',
                         'home_days_rest', 'home_b2b', 'home_3in4']].merge(
        away_rows[['game_id_away', 'away_days_rest', 'away_b2b', 'away_3in4']].rename(
            columns={'game_id_away': 'game_id'}),
        on='game_id', how='inner'
    )

    # Compute net rest delta (home perspective, pts/100)
    delta = pd.Series(0.0, index=merged.index)
    delta += merged['home_b2b'].astype(float) * HOME_B2B_PENALTY
    delta -= merged['away_b2b'].astype(float) * ROAD_B2B_PENALTY  # away B2B helps home
    delta += merged['home_3in4'].astype(float) * HOME_3IN4_EXTRA
    delta -= merged['away_3in4'].astype(float) * AWAY_3IN4_EXTRA

    # Rest-day advantage bonus (capped at 2 extra days)
    rest_diff = (merged['away_days_rest'] - merged['home_days_rest']).clip(-2, 2)
    delta += rest_diff * REST_ADV_BONUS

    merged['rest_net_delta'] = delta
    merged['game_date'] = merged['game_date'].dt.strftime('%Y-%m-%d')

    out_cols = ['game_id', 'game_date', 'season', 'home_team', 'away_team',
                'home_days_rest', 'away_days_rest', 'home_b2b', 'away_b2b',
                'home_3in4', 'away_3in4', 'rest_net_delta']
    merged[out_cols].to_parquet(args.output, index=False)
    print(f"Saved {len(merged)} game-rest rows to {args.output}")

    # Sanity: B2B frequency should be ~15-20% of games
    b2b_rate = merged[['home_b2b', 'away_b2b']].any(axis=1).mean()
    print(f"Games with at least one B2B team: {b2b_rate:.1%} (expect 15-25%)")

if __name__ == "__main__":
    main()
```

### Integration into game model

Locate the game model win-probability computation in `src/simulation/validate_forecast.py`. Before calling `norm.cdf(delta_mu / sigma)`, apply the rest delta:

```python
# Load rest features at module/function level
rest_df = pd.read_parquet(REPO / "data/processed/forecast/game_rest_features.parquet")
rest_lookup = rest_df.set_index(['game_id'])['rest_net_delta'].to_dict()

# Per game, before win_prob computation:
rest_delta = rest_lookup.get(game_id, 0.0)
adjusted_delta_mu = delta_mu + rest_delta   # positive = home advantage
win_prob = norm.cdf(adjusted_delta_mu / sigma)
```

### Validation gates

- B2B frequency check: 15–25% of games have at least one B2B team (automatic in script above)
- B2B teams' predicted win % should drop ~3–5 pp vs. their baseline (spot-check 2022-23)
- Brier after integration ≤ current Brier − 0.008 (conservative floor of expected range)
- CLV vs. Kalshi should improve; if not, the rest delta magnitudes need recalibration

---

## Step 4 — Season-to-Date Power Rating Update

**Dependencies:** Step 1 (stable team features)  
**Can run in parallel with:** Step 3  
**Expected Brier gain:** −0.008 to −0.012  
**Basketball source:** After ~30 games (~late November), actual season-to-date net rating is more predictive of remaining season performance than preseason projections (FiveThirtyEight Elo and RAPTOR both begin blending observed data after ~10 games). Preseason projections have a correlation of ~0.6 with actual team quality; season-to-date net rating (30+ games) reaches ~0.8.

### What this does

Blend each team's preseason `team_net_rating_projected` toward actual YTD net rating with a weight that grows through the season:

```
alpha(game_number) = min(1.0, game_number / 82)^0.5
adjusted_projected = (1 - alpha) * preseason_projected + alpha * ytd_net_rating
```

At game 1: alpha ≈ 0.11, 89% preseason weight.  
At game 30: alpha ≈ 0.60, 40% preseason weight.  
At game 82: alpha = 1.0, 100% YTD (end of season is fully observed).

### Create script: `scripts/build_ytd_team_ratings.py`

Input: `data/historical/team_game_logs.parquet` + `data/processed/forecast/projected_team_features_v40.parquet`  
Output: `data/processed/forecast/team_ratings_ytd.parquet`

Output schema:
- `game_id` (str)
- `game_date` (str, YYYY-MM-DD)
- `season` (str)
- `team_abbreviation` (str)
- `preseason_net_rating` (float)
- `ytd_net_rating` (float, pts/100 through games played before this game — OOS: excludes current game)
- `games_played` (int, excludes current game)
- `blended_net_rating` (float, alpha-blended estimate)

```python
"""
scripts/build_ytd_team_ratings.py
Computes per-team YTD net rating before each game (OOS, excludes current game).
Blends preseason projection with observed YTD using a sqrt-ramp alpha.

Usage:
    python3 scripts/build_ytd_team_ratings.py \
        --logs data/historical/team_game_logs.parquet \
        --preseason data/processed/forecast/projected_team_features_v40.parquet \
        --output data/processed/forecast/team_ratings_ytd.parquet
"""
import argparse, sys
from pathlib import Path
import pandas as pd
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

GAMES_IN_SEASON = 82


def compute_ytd_ratings(logs):
    """Compute cumulative net rating per team per game, excluding current game (OOS)."""
    logs = logs.copy()
    logs['GAME_DATE'] = pd.to_datetime(logs['GAME_DATE'])
    # Net rating per game = PLUS_MINUS / (MIN / 48) * (100 / pace_est) 
    # Approximate: use PLUS_MINUS scaled to per-100-poss
    # PLUS_MINUS is total points ± in the game; per-poss ~ PLUS_MINUS / (MIN/5/100)
    # Simple approximation: assume 100 possessions per 48 min
    logs['game_net_rating'] = logs['PLUS_MINUS'] / (logs['MIN'] / 48.0) * (100.0 / 100.0)
    # ^ This simplifies to PLUS_MINUS / (MIN/48), which is net rating per 48 min
    # Convert to per-100: net_rating_per100 ~ (PLUS_MINUS / (MIN/48)) * (48/100) * 100
    # Actually: net rating per 100 poss ≈ PLUS_MINUS / (MIN / 48) (if ~100 poss per 48 min)
    logs['game_net_rating'] = logs['PLUS_MINUS'] / (logs['MIN'] / 48.0)

    logs = logs.sort_values(['TEAM_ABBREVIATION', 'GAME_DATE', 'GAME_ID']).reset_index(drop=True)

    # Cumulative sum BEFORE current game (shift by 1)
    logs['cum_net_rating_sum'] = logs.groupby(['TEAM_ABBREVIATION', 'SEASON'])['game_net_rating'].cumsum().shift(1)
    logs['games_played'] = logs.groupby(['TEAM_ABBREVIATION', 'SEASON']).cumcount()
    # games_played = number of games played BEFORE this game

    with np.errstate(divide='ignore', invalid='ignore'):
        logs['ytd_net_rating'] = np.where(
            logs['games_played'] > 0,
            logs['cum_net_rating_sum'] / logs['games_played'],
            np.nan
        )
    return logs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", default=str(REPO / "data/historical/team_game_logs.parquet"))
    parser.add_argument("--preseason", default=str(REPO / "data/processed/forecast/projected_team_features_v40.parquet"))
    parser.add_argument("--output", default=str(REPO / "data/processed/forecast/team_ratings_ytd.parquet"))
    args = parser.parse_args()

    logs = pd.read_parquet(args.logs)
    preseason = pd.read_parquet(args.preseason)

    ytd = compute_ytd_ratings(logs)

    # Build preseason lookup: season + team_abbreviation -> preseason_net_rating
    preseason_lookup = preseason.set_index(['season', 'team_abbreviation'])['team_net_rating_projected'].to_dict()

    ytd['season'] = ytd['SEASON']
    ytd['preseason_net_rating'] = ytd.apply(
        lambda r: preseason_lookup.get((r['season'], r['TEAM_ABBREVIATION']), np.nan), axis=1
    )

    # Alpha ramp: sqrt(game_number / 82), capped at 1.0
    ytd['alpha'] = np.sqrt(ytd['games_played'] / GAMES_IN_SEASON).clip(0, 1)

    # Blended: for early games where ytd_net_rating is NaN, fall back to preseason
    ytd['blended_net_rating'] = np.where(
        ytd['games_played'] > 0,
        (1 - ytd['alpha']) * ytd['preseason_net_rating'] + ytd['alpha'] * ytd['ytd_net_rating'],
        ytd['preseason_net_rating']
    )

    out = ytd.rename(columns={
        'GAME_ID': 'game_id',
        'GAME_DATE': 'game_date',
        'TEAM_ABBREVIATION': 'team_abbreviation',
    }).copy()
    out['game_date'] = out['game_date'].dt.strftime('%Y-%m-%d')

    out_cols = ['game_id', 'game_date', 'season', 'team_abbreviation',
                'preseason_net_rating', 'ytd_net_rating', 'games_played', 'blended_net_rating']
    out[out_cols].to_parquet(args.output, index=False)
    print(f"Saved {len(out)} team-game YTD rating rows to {args.output}")

    # Sanity: after game 30, blended_net_rating should deviate from preseason by ~2-4 pts for good/bad teams
    late = out[out['games_played'] >= 30].copy()
    delta = (late['blended_net_rating'] - late['preseason_net_rating']).abs()
    print(f"Late season mean |blended - preseason| delta: {delta.mean():.2f} pts/100 (expect 2-5)")

if __name__ == "__main__":
    main()
```

### Integration into game model

In the game model, replace static preseason `team_net_rating_projected` lookup with blended rating:

```python
ytd_df = pd.read_parquet(REPO / "data/processed/forecast/team_ratings_ytd.parquet")
# Build lookup: (game_id, team_abbreviation) -> blended_net_rating
ytd_lookup = ytd_df.set_index(['game_id', 'team_abbreviation'])['blended_net_rating'].to_dict()

# Per game:
home_rating = ytd_lookup.get((game_id, home_team), preseason_home_rating)
away_rating = ytd_lookup.get((game_id, away_team), preseason_away_rating)
delta_mu = home_rating - away_rating + HCA
```

### Validation gates

- In 2024-25, blended rating by game 40 should correlate ≥ 0.75 with final team net rating (vs ~0.60 for preseason alone)
- Brier should improve ≥ 0.006 vs Step 1 baseline when YTD update is active
- No regression in early-season accuracy (games 1–10): alpha ≈ 0.11, preseason still dominates

---

## Step 5 — GBDT Game Model

**Dependencies:** Steps 1 + 3 + 4 (need rest features and YTD ratings as training features)  
**Expected Brier gain:** −0.012 to −0.018  
**Basketball source:** XGBoost/LightGBM models trained on team-level NBA features achieve ~76% game prediction accuracy in peer-reviewed studies (PMC NCBI, 2024). FiveThirtyEight's Elo and RAPTOR models that blend multiple signals consistently outperform single-signal models. Non-linear relationships between team strength and win probability (e.g., extreme mismatches are over-predicted by Gaussian) are well-documented.

### What this does

Replace the deterministic Gaussian `win_prob = norm.cdf(delta_mu / sigma)` with a LightGBM classifier trained walk-forward on historical games. The GBDT learns non-linear relationships between features and win probability.

**Training features per game:**
1. `net_rating_diff` — blended_net_rating_home − blended_net_rating_away (from Step 4)
2. `rest_net_delta` — from Step 3
3. `home_b2b` (bool)
4. `away_b2b` (bool)
5. `hca` — home court advantage, fixed at 2.5 pts/100 (later: per-team)
6. `season_stage` — game number / 82, captures regular vs. playoff intensity
7. `home_recent_form` — net rating in last 10 games (from team_game_logs)
8. `away_recent_form` — same
9. `home_win_pct_ytd` — home team win % before this game
10. `away_win_pct_ytd` — away team win % before this game

**Target:** `home_result` (1=home win, 0=away win)

### Create script: `scripts/train_gbdt_game_model.py`

```python
"""
scripts/train_gbdt_game_model.py
Train a LightGBM walk-forward game model.

Walk-forward CV: for each season S, train on all seasons < S, predict S.
This ensures no future leakage in Brier evaluation.

Usage:
    python3 scripts/train_gbdt_game_model.py \
        --logs data/historical/team_game_logs.parquet \
        --ytd data/processed/forecast/team_ratings_ytd.parquet \
        --rest data/processed/forecast/game_rest_features.parquet \
        --output models/gbdt_game_model.pkl \
        --report reports/gbdt_walk_forward_brier.json
"""
import argparse, json, pickle, sys
from pathlib import Path
import pandas as pd
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    print("WARNING: lightgbm not installed. Run: pip install lightgbm")

FEATURE_COLS = [
    'net_rating_diff', 'rest_net_delta', 'home_b2b', 'away_b2b',
    'season_stage', 'home_recent_form', 'away_recent_form',
    'home_win_pct_ytd', 'away_win_pct_ytd'
]

def compute_recent_form(logs, n_games=10):
    """Compute rolling n-game average net rating per team (OOS: excludes current game)."""
    logs = logs.sort_values(['TEAM_ABBREVIATION', 'GAME_DATE'])
    logs['game_net_rating'] = logs['PLUS_MINUS'] / (logs['MIN'] / 48.0)
    logs['recent_form'] = (
        logs.groupby('TEAM_ABBREVIATION')['game_net_rating']
        .transform(lambda x: x.shift(1).rolling(n_games, min_periods=3).mean())
    )
    return logs.set_index('GAME_ID')['recent_form'].to_dict()


def compute_win_pct(logs):
    """Compute cumulative win % before each game (OOS)."""
    logs = logs.sort_values(['TEAM_ABBREVIATION', 'GAME_DATE'])
    logs['win'] = (logs['WL'] == 'W').astype(int)
    logs['cum_wins'] = logs.groupby(['TEAM_ABBREVIATION', 'SEASON'])['win'].cumsum().shift(1).fillna(0)
    logs['games_played'] = logs.groupby(['TEAM_ABBREVIATION', 'SEASON']).cumcount()
    with np.errstate(divide='ignore', invalid='ignore'):
        logs['win_pct'] = np.where(
            logs['games_played'] > 0, logs['cum_wins'] / logs['games_played'], 0.5
        )
    return logs.set_index('GAME_ID')['win_pct'].to_dict()


def build_game_features(logs, ytd_df, rest_df):
    """Build one row per game (home perspective) with all features."""
    logs['GAME_DATE'] = pd.to_datetime(logs['GAME_DATE'])
    
    # Parse home/away
    def is_home(matchup):
        return ' vs. ' in matchup
    
    home_logs = logs[logs['MATCHUP'].apply(is_home)].copy()
    away_logs = logs[~logs['MATCHUP'].apply(is_home)].copy()
    
    recent_form = compute_recent_form(logs)
    win_pct = compute_win_pct(logs)
    
    home_logs['home_recent_form'] = home_logs['GAME_ID'].map(recent_form).fillna(0)
    away_logs['away_recent_form'] = away_logs['GAME_ID'].map(recent_form).fillna(0)
    away_logs['away_win_pct_ytd'] = away_logs['GAME_ID'].map(win_pct).fillna(0.5)
    home_logs['home_win_pct_ytd'] = home_logs['GAME_ID'].map(win_pct).fillna(0.5)
    home_logs['home_result'] = (home_logs['WL'] == 'W').astype(int)
    home_logs['season_game_num'] = home_logs.groupby('SEASON').cumcount()
    home_logs['season_stage'] = home_logs['season_game_num'] / 82.0
    
    # Merge YTD blended ratings
    ytd_home = ytd_df.rename(columns={'blended_net_rating': 'home_blended_rating'})
    ytd_away = ytd_df.rename(columns={'blended_net_rating': 'away_blended_rating'})
    
    home_logs = home_logs.merge(
        ytd_home[['game_id', 'team_abbreviation', 'home_blended_rating']].rename(
            columns={'game_id': 'GAME_ID', 'team_abbreviation': 'TEAM_ABBREVIATION'}),
        on=['GAME_ID', 'TEAM_ABBREVIATION'], how='left'
    )
    
    # Away team abbreviation from MATCHUP: "HOME @ AWAY" => away = MATCHUP.split(' vs. ')[1]
    home_logs['away_team'] = home_logs['MATCHUP'].str.split(' vs. ').str[1].str.strip()
    home_logs = home_logs.merge(
        away_logs[['GAME_ID', 'TEAM_ABBREVIATION', 'away_recent_form', 'away_win_pct_ytd']].rename(
            columns={'TEAM_ABBREVIATION': 'away_team'}),
        on=['GAME_ID', 'away_team'], how='left'
    )
    home_logs = home_logs.merge(
        ytd_away[['game_id', 'team_abbreviation', 'away_blended_rating']].rename(
            columns={'game_id': 'GAME_ID', 'team_abbreviation': 'away_team'}),
        on=['GAME_ID', 'away_team'], how='left'
    )
    
    # Merge rest features
    home_logs = home_logs.merge(
        rest_df[['game_id', 'rest_net_delta', 'home_b2b', 'away_b2b']].rename(
            columns={'game_id': 'GAME_ID'}),
        on='GAME_ID', how='left'
    )
    
    home_logs['net_rating_diff'] = (
        home_logs['home_blended_rating'].fillna(0) - home_logs['away_blended_rating'].fillna(0)
    )
    home_logs['rest_net_delta'] = home_logs['rest_net_delta'].fillna(0)
    home_logs['home_b2b'] = home_logs['home_b2b'].fillna(False).astype(int)
    home_logs['away_b2b'] = home_logs['away_b2b'].fillna(False).astype(int)
    
    return home_logs[FEATURE_COLS + ['home_result', 'SEASON', 'GAME_ID', 'GAME_DATE']].dropna(subset=['net_rating_diff', 'home_result'])


def brier(probs, outcomes):
    return float(np.mean((np.array(probs) - np.array(outcomes)) ** 2))


def main():
    if not LGB_AVAILABLE:
        print("Install lightgbm first: pip install lightgbm")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", default=str(REPO / "data/historical/team_game_logs.parquet"))
    parser.add_argument("--ytd", default=str(REPO / "data/processed/forecast/team_ratings_ytd.parquet"))
    parser.add_argument("--rest", default=str(REPO / "data/processed/forecast/game_rest_features.parquet"))
    parser.add_argument("--output", default=str(REPO / "models/gbdt_game_model.pkl"))
    parser.add_argument("--report", default=str(REPO / "reports/gbdt_walk_forward_brier.json"))
    args = parser.parse_args()

    logs = pd.read_parquet(args.logs)
    ytd_df = pd.read_parquet(args.ytd)
    rest_df = pd.read_parquet(args.rest)

    df = build_game_features(logs, ytd_df, rest_df)
    seasons = sorted(df['SEASON'].unique())

    # Walk-forward CV
    wf_results = []
    all_preds = []
    all_actuals = []

    for i, season in enumerate(seasons):
        train_seasons = seasons[:i]
        if len(train_seasons) < 2:
            continue  # Need at least 2 seasons to train

        train = df[df['SEASON'].isin(train_seasons)]
        test = df[df['SEASON'] == season]

        X_train = train[FEATURE_COLS].values
        y_train = train['home_result'].values
        X_test = test[FEATURE_COLS].values
        y_test = test['home_result'].values

        model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            reg_lambda=1.0,
            min_child_samples=30,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]

        season_brier = brier(probs, y_test)
        wf_results.append({'season': season, 'brier': season_brier, 'n_games': len(y_test)})
        all_preds.extend(probs.tolist())
        all_actuals.extend(y_test.tolist())

        print(f"  {season}: Brier={season_brier:.4f} ({len(y_test)} games)")

    overall_brier = brier(all_preds, all_actuals)
    print(f"\nWalk-forward Brier: {overall_brier:.4f}")
    print(f"Number of seasons evaluated: {len(wf_results)}")

    # Train final model on all data for deployment
    final_model = lgb.LGBMClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        reg_lambda=1.0, min_child_samples=30, subsample=0.8,
        colsample_bytree=0.8, random_state=42, verbose=-1
    )
    final_model.fit(df[FEATURE_COLS].values, df['home_result'].values)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump({'model': final_model, 'feature_cols': FEATURE_COLS}, f)

    report = {
        'walk_forward_brier': overall_brier,
        'gaussian_baseline_brier': 0.2330,  # update with actual Step 1+3+4 baseline
        'brier_improvement': 0.2330 - overall_brier,
        'n_seasons_evaluated': len(wf_results),
        'per_season': wf_results,
        'feature_importances': dict(zip(FEATURE_COLS, final_model.feature_importances_.tolist())),
    }
    with open(args.report, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nModel saved to {args.output}")
    print(f"Report saved to {args.report}")

if __name__ == "__main__":
    main()
```

### Validation gates

- Walk-forward Brier ≤ 0.220 (below Gaussian baseline after Steps 1+3+4)
- SHAP feature importance: `net_rating_diff` should rank #1, rest features #2-#4
- No individual season Brier > 0.250 (flag if one season is an outlier)
- Compare per-game predictions against Gaussian side-by-side: GBDT should reduce systematic error on extreme mismatches (delta_mu > 6 pts) and B2B games

**Install requirement (add to requirements.txt):**
```
lightgbm>=4.0
```

---

## Step 6 — Ensemble Game Model  ⛔ SUPERSEDED / NOT BEING SHIPPED (2026-05-30)

> **Status:** Built and evaluated inside `src/simulation/gbdt_game_model.py`
> (the `blend_30_50_20` leg), then **retired**. The fixed 30/50/20 ensemble
> (OOS Brier 0.2223) is **worse than Elo alone (0.2193)** — it fails this step's
> own validation gate ("no single model should dominate; if one does, drop the
> weak legs"). Elo dominates; the Gaussian leg drags the blend down. We are
> **not** shipping the fixed ensemble. Interim production model = GBDT-stack-with-Elo
> or Elo alone (see `docs/findings/game_model_comparison_2026-05-30.md`). The real
> next step is **Track 2 / Step 7 below**: wire PTS/BKE player-impact into the game
> model and test incremental CLV where Elo is blind. Original spec kept below for
> history.

**Dependencies:** Step 5 (GBDT trained and validated)  
**Expected Brier gain:** −0.003 to −0.005 *(NOT realized — ensemble underperformed Elo)*  
**Basketball source:** Ensemble methods reduce systematic bias by ~0.003 Brier when base models have complementary error patterns. Gaussian captures symmetric calibration well; GBDT captures non-linear matchup effects; simple Elo captures momentum. *(In practice the legs were not complementary: Elo subsumed the others.)*

### What this does

Average win probabilities from 3 models:
1. Gaussian (current): `norm.cdf(delta_mu / sigma)` — symmetric, well-calibrated on average
2. GBDT (Step 5): non-linear, rest-aware
3. Simple Elo: team-level rating that updates after each game within season

Ensemble weights (fixed, no further tuning): `[0.3, 0.5, 0.2]` for Gaussian, GBDT, Elo.

### Create: `scripts/ensemble_game_model.py`

```python
"""
scripts/ensemble_game_model.py
Blends Gaussian + GBDT + Elo predictions for final win probabilities.

Usage:
    python3 scripts/ensemble_game_model.py \
        --gaussian-preds reports/gaussian_game_preds.parquet \
        --gbdt-model models/gbdt_game_model.pkl \
        --ytd data/processed/forecast/team_ratings_ytd.parquet \
        --rest data/processed/forecast/game_rest_features.parquet \
        --output reports/ensemble_game_preds.parquet
"""
import argparse, pickle, sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

GAUSSIAN_WEIGHT = 0.30
GBDT_WEIGHT     = 0.50
ELO_WEIGHT      = 0.20

ELO_K = 20.0          # K-factor for within-season Elo updates
ELO_INIT = 1500.0     # Initial Elo for all teams at season start


def expected_elo(rating_a, rating_b):
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def compute_elo_probs(logs):
    """Compute OOS Elo win probability for each game."""
    logs = logs.sort_values(['GAME_DATE', 'GAME_ID'])
    elo_ratings = {}
    results = []

    for _, row in logs.iterrows():
        matchup = row['MATCHUP']
        if ' vs. ' not in matchup:
            continue
        home = row['TEAM_ABBREVIATION']
        away = matchup.split(' vs. ')[1].strip()
        season = row['SEASON']

        # Reset Elo at season start if new season
        if (home, season) not in elo_ratings:
            elo_ratings[(home, season)] = ELO_INIT
        if (away, season) not in elo_ratings:
            elo_ratings[(away, season)] = ELO_INIT

        home_elo = elo_ratings[(home, season)]
        away_elo = elo_ratings[(away, season)]

        home_win_prob = expected_elo(home_elo, away_elo)
        home_win = 1 if row['WL'] == 'W' else 0

        # Update Elo after game
        elo_ratings[(home, season)] += ELO_K * (home_win - home_win_prob)
        elo_ratings[(away, season)] += ELO_K * ((1 - home_win) - (1 - home_win_prob))

        results.append({
            'GAME_ID': row['GAME_ID'],
            'elo_home_win_prob': float(home_win_prob),
        })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gaussian-preds", required=True)
    parser.add_argument("--gbdt-model", default=str(REPO / "models/gbdt_game_model.pkl"))
    parser.add_argument("--logs", default=str(REPO / "data/historical/team_game_logs.parquet"))
    parser.add_argument("--ytd", default=str(REPO / "data/processed/forecast/team_ratings_ytd.parquet"))
    parser.add_argument("--rest", default=str(REPO / "data/processed/forecast/game_rest_features.parquet"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    gaussian = pd.read_parquet(args.gaussian_preds)  # columns: GAME_ID, gaussian_home_win_prob, home_result
    logs = pd.read_parquet(args.logs)
    ytd = pd.read_parquet(args.ytd)
    rest = pd.read_parquet(args.rest)

    # Elo predictions
    logs['GAME_DATE'] = pd.to_datetime(logs['GAME_DATE'])
    elo_preds = compute_elo_probs(logs)

    # GBDT predictions (reuse feature builder from train script)
    with open(args.gbdt_model, 'rb') as f:
        gbdt_bundle = pickle.load(f)
    gbdt_model = gbdt_bundle['model']
    feature_cols = gbdt_bundle['feature_cols']

    # Import feature builder
    sys.path.insert(0, str(REPO / "scripts"))
    from train_gbdt_game_model import build_game_features, FEATURE_COLS
    df_features = build_game_features(logs, ytd, rest)
    X = df_features[feature_cols].values
    gbdt_probs = gbdt_model.predict_proba(X)[:, 1]
    gbdt_preds = df_features[['GAME_ID']].copy()
    gbdt_preds['gbdt_home_win_prob'] = gbdt_probs

    # Merge all
    merged = gaussian.merge(elo_preds, on='GAME_ID', how='inner')
    merged = merged.merge(gbdt_preds, on='GAME_ID', how='inner')

    merged['ensemble_home_win_prob'] = (
        GAUSSIAN_WEIGHT * merged['gaussian_home_win_prob'] +
        GBDT_WEIGHT * merged['gbdt_home_win_prob'] +
        ELO_WEIGHT * merged['elo_home_win_prob']
    )

    brier_ensemble = float(np.mean((merged['ensemble_home_win_prob'] - merged['home_result']) ** 2))
    brier_gaussian = float(np.mean((merged['gaussian_home_win_prob'] - merged['home_result']) ** 2))
    print(f"Ensemble Brier: {brier_ensemble:.4f}")
    print(f"Gaussian Brier: {brier_gaussian:.4f}")
    print(f"Improvement:    {brier_gaussian - brier_ensemble:.4f}")

    merged.to_parquet(args.output, index=False)
    print(f"Saved ensemble predictions to {args.output}")

if __name__ == "__main__":
    main()
```

### Validation gates

- Ensemble Brier ≤ GBDT Brier − 0.001 (if not, weights need retuning)
- No single model dominates (GBDT weight > 0.80 signals the others are noise; reduce to 2-model ensemble)
- CLV should match or beat Step 5 CLV

---

## Summary: Implementation Order and Brier Targets

| Step | Action | Brier Target | Est. Runtime |
|------|--------|-------------|-------------|
| 1 | Lock composite v4.0 (A+C blend) | ≤ 0.233 | 30 min |
| 2 | Kalshi CLV harness | N/A (measurement) | 1–2 days (data acquisition) |
| 3 | Rest/B2B correction layer | ≤ 0.241 (actual: 0.2418 agg) | COMPLETE 2026-05-28 |
| 4 | Season-to-date power rating | ≤ 0.215 (with Step 3) | 2–3 hours |
| 5 | GBDT game model | ≤ 0.210 | 4–6 hours (training) |
| 6 | Ensemble | ≤ 0.207 | 1–2 hours |

**Cumulative estimated Brier after all steps:** 0.203–0.210 (Vegas band)

**Path to Kalshi alpha:** Steps 1 → 2 → 3+4 (parallel) → 5 → 6 → CLV re-measurement

---

## Sources

- Back-to-back ATS performance: The Wager Theorem, "NBA Back-to-Back Betting Analysis" (2024) — road B2B teams lose ATS at 57% rate vs. rested opponents; 1–3 pt net rating penalty.
- Home court advantage: PMC Systematic Review, "Home Advantage in Basketball" (2024) — modern HCA 2.2–3.2 pts/100 league-wide; Denver altitude +5.5 pts/game (Sportico, 2024).
- GBDT for game prediction: PMC, "XGBoost for NBA Game Prediction" (2024) — 76% accuracy on game winner with team-level features; SHAP importance shows team quality differential as #1 predictor.
- FiveThirtyEight methodology: Elo and RAPTOR both blend preseason projections with observed season data, transitioning weight from preseason to actual by ~game 30.
- Kalshi accuracy: Kalshi's own Brier Score glossary confirms their markets approach theoretical minimum Brier near resolution. Sharp books maintain Brier 0.195–0.210 at closing.
