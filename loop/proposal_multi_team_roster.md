# Multi-Team Roster Solution — Philosophy & Design Proposal

**Date:** 2026-03-07  
**Status:** PROPOSAL — Awaiting approval before implementation  
**Author:** Copilot (Beast Mode)

---

## 1. Problem Statement

Currently, the BKE pipeline assigns each player to **exactly one team per season** — their end-of-season (last) team as reported by the NBA's `leaguedashplayerstats` endpoint. This creates three distinct problems:

### 1.1 Backtest Accuracy Problem
In any given season, **70–81 players** (~15% of all players) play for multiple teams. Examples:

| Player | Season | Actual Teams (Games) | Pipeline Assignment |
|--------|--------|---------------------|-------------------|
| Mikal Bridges | 2022-23 | PHX (56), BKN (27) | BKN only |
| D'Angelo Russell | 2022-23 | MIN (54), LAL (17) | LAL only |
| D'Angelo Russell | 2024-25 | LAL (29), BKN (29) | BKN only |
| Anthony Davis | 2024-25 | LAL (~40), DAL (~11) | DAL only |

**Impact:** PHX's 2022-23 backtest is missing Bridges' 56-game contribution. BKN's backtest overcounts him (gets his full 83-game stats instead of 27 games). This distorts both teams' talent estimates and undermines team-level calibration.

### 1.2 Forecast Contamination Problem
When forecasting season N+1 from season N data, we use season N's end-of-season rosters. But the forecast is supposed to represent **preseason expectations**. End-of-season rosters include:
- Mid-season trades (Bridges to NYK, Davis to DAL)
- Buyout signings (e.g., a player bought out in February and signed by a contender)
- Seasonal roster churn that hasn't happened yet at forecast time

This makes our "forecast" partially retrospective — we're using information unavailable at the forecast's hypothetical evaluation point.

### 1.3 BKE Impact Portability Problem
The BKE decomposition computes **season-level** impact scores (OBKE, DBKE) that blend a player's performance across all teams they played for. For most players, these are reasonably portable (the BKE decomposition is designed to isolate individual contribution). However:
- A player's **behavioral profile** (playtype distribution, usage rate, minutes) can differ significantly between stints
- These behavioral differences affect team structure metrics (archetype interaction, playmaking depth, etc.)
- The current system applies the same behavioral fingerprint to all teams, which is inaccurate for the pre-trade team

---

## 2. Proposed Solution: Team-Stint Architecture

### 2.1 Core Concept: Split by Stint, Not by Season

Instead of one row per player per season, the pipeline should produce **one row per player per team-stint per season**. A "stint" is a continuous period where a player is on the same team's roster.

**Data Source:** `final_player_game_logs.parquet` already contains per-game team information via the `MATCHUP` field (extractable as the first token: `BKN vs. PHI` → `BKN`). This gives us exact game-by-game team affiliation.

### 2.2 New Intermediate Data Product: Player-Team Stints

**File:** `data/processed/player_team_stints.parquet`

Schema:
```
player_id, player_name, season, team_abbreviation, stint_number,
games_played, total_minutes, mpg,
first_game_date, last_game_date,
is_primary_stint (bool — true for the stint with most minutes)
```

**Construction:** For each player-season, group game logs by team (extracted from MATCHUP), order by date, and assign stint numbers. A player can have stints like:
- Stint 1: MIN (54 games, Oct–Feb)
- Stint 2: LAL (17 games, Feb–Apr)

### 2.3 Impact Allocation Strategy

The BKE decomposition produces **one** impact score per player per season. We do NOT propose re-running BKE per stint (this would require per-stint RAPM computation which is data-starved for short stints). Instead:

**Portable impact, stint-specific behavior:**
- **OBKE / DBKE scores**: Same for all stints (they measure portable individual impact)
- **Minutes / MPG**: Stint-specific (from game logs)
- **Behavioral stats** (usage, playtype distribution, shooting): Ideally stint-specific if derivable from game logs; otherwise use season-wide values with a flag indicating "blended"
- **Archetype assignment**: Use season-level archetype (behavioral archetype is role-based and generally stable across stints)
- **Team assignment**: Stint-specific (the whole point)

This means for team aggregation, Mikal Bridges' 2022-23 profile would appear:
- Under PHX with 56 games, ~35 MPG, his OBKE/DBKE scores → PHX gets his 56-game contribution
- Under BKN with 27 games, ~35 MPG, his OBKE/DBKE scores → BKN gets his 27-game contribution

### 2.4 Backtest Mode Changes

**`build_player_impact_profiles.py`:**
- After building season-level profiles, merge with `player_team_stints.parquet`
- For multi-team players (TEAM_COUNT > 1), split the single profile row into N stint rows
- Each stint row gets: stint-specific (team_abbreviation, games, minutes, mpg), shared (impact scores, archetype, behavioral profile)
- Output: `player_impact_profiles.parquet` now has multiple rows per player-season for traded players

**`team_feature_aggregation.py`:**
- No changes needed — it already groups by (season, team_abbreviation)
- Each team now correctly gets only the players who actually played for them, with proportional minutes

**Validation considerations:**
- Minute shares within each team should still sum to ~1.0 (240 minutes per team-game)
- A player's total minutes across stints should match their season total
- This should improve backtest accuracy for teams heavily affected by trades

### 2.5 Forecast Mode Changes

**The key philosophy question: What roster should we use for forecasting?**

**Proposed answer: Use the preseason roster (opening day), not the end-of-season roster.**

Implementation options (listed from simplest to most sophisticated):

#### Option A: Carry-Forward with Free Agency Awareness (Recommended for v1)
- Use the **prior season's final stint** team for each player (where they finished the season)
- This is the simplest approximation of "where will they be next year" and is correct for ~85% of players
- For the ~15% who get traded mid-season, they'll be on the "wrong" team, but this is the same error as before
- **Improvement over status quo:** None for forecast, but backtest improves significantly

#### Option B: Preseason Roster Snapshot (Recommended for v2)
- Add a new data fetch step: `fetch_preseason_rosters.py` that pulls team rosters as of a specified date (e.g., October 1)
- NBA API endpoint: `commonteamroster` with `LeagueID=00` and `Season=YYYY-YY`
- This captures free agency signings and known trades but not mid-season moves
- **Improvement:** Forecast uses correct preseason rosters, including summer free agent signings and offseason trades
- **Limitation:** Still doesn't predict mid-season trades

#### Option C: Roster + Transaction Timeline (Future Enhancement)
- Fetch historical transaction data (trades, signings, waivers)
- Build a timeline model that can reconstruct "who was on what team" at any point in the season
- Could potentially model trade probability for forecast scenarios
- **This is over-engineering for current needs and should remain a future goal**

### 2.6 Handling Edge Cases

**Waived/released players:** Players waived and not re-signed will have a stint that ends mid-season. Their contribution counts for the team where they played, with accurate game/minute counts.

**10-day contracts:** Short-stint players (< 5 games) should be handled normally but will contribute minimally to team aggregation due to low minutes.

**Season-ending injuries:** Not a multi-team problem per se, but the availability discount system (implemented in this session) already handles this via expected games-played fraction.

**Three or more teams:** A player on 3+ teams in a season simply gets 3+ stint rows. The system generalizes naturally.

---

## 3. Pipeline Integration Plan

### 3.1 New Pipeline Steps

```
┌─────────────────────────────────────────────────────────┐
│ CURRENT PIPELINE                                         │
│                                                         │
│ fetch_box_scores_complete → complete_player_season_stats │
│         ↓                                               │
│ build_player_impact_profiles → player_impact_profiles   │
│         ↓                                               │
│ team_feature_aggregation → team_feature_aggregation     │
│         ↓                                               │
│ season_sim → season results                             │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ PROPOSED PIPELINE                                        │
│                                                         │
│ fetch_box_scores_complete → complete_player_season_stats │
│ derive_player_team_stints → player_team_stints (NEW)    │
│         ↓                                               │
│ build_player_impact_profiles → player_impact_profiles   │
│   (now with stint-level rows for multi-team players)    │
│         ↓                                               │
│ team_feature_aggregation → team_feature_aggregation     │
│   (unchanged — groups by team_abbreviation naturally)   │
│         ↓                                               │
│ season_sim → season results                             │
└─────────────────────────────────────────────────────────┘
```

### 3.2 New Files

| File | Purpose |
|------|---------|
| `src/features/derive_player_team_stints.py` | Extract per-stint team assignments from game logs |
| `src/data_fetch/fetch_preseason_rosters.py` | (v2) Fetch roster snapshots for forecast mode |

### 3.3 Modified Files

| File | Change |
|------|--------|
| `src/player_eval/build_player_impact_profiles.py` | Merge stint data, split multi-team rows |
| `src/player_eval/project_next_season.py` | Use primary stint team for carry-forward (minor) |
| `readme.md` | Add new pipeline steps |

### 3.4 No Changes Needed

| File | Why |
|------|-----|
| `src/profile_aggregate/team_feature_aggregation.py` | Already groups by (season, team_abbreviation) |
| `src/simulation/season_sim.py` | Operates on team features, not player-level |
| `src/modeling/model_rapm.py` | RAPM operates on possessions, not team rosters |
| BKE decomposition scripts | Impact is player-level, not team-level |

---

## 4. Frontend Implications

### 4.1 Existing Viewers

**`player_bke_viewer.py`** — Currently shows one row per player-season. Should show stint breakdown for multi-team players (expandable row or separate stint rows).

**`player_eval_viewer.py`** — Same consideration. Player evaluation should indicate when a player had multiple stints with team labels.

**`simulation_viewer.py`** — Team-level projections should benefit from more accurate rosters. No UI change needed.

**`player_data_viewer.py`** — Should show all stints for a player when viewing their season data.

### 4.2 Proposed New Frontend Features (Future)

**Roster Timeline View:** An interactive team-season view showing:
- Which players were on the roster at each point in the season
- Trade markers showing incoming/outgoing players
- Minutes distribution pre/post trade

**Trade Impact Analysis:** For each mid-season trade, show:
- Pre-trade team talent estimate vs. post-trade
- Net impact of the trade on each team's projected wins

These are enhancement features and should NOT be implemented alongside the core multi-team fix. They should be separate initiatives after the data layer is correct.

---

## 5. Implementation Priority & Phasing

### Phase 1: Backtest Accuracy (Immediate — next implementation cycle)
1. Build `derive_player_team_stints.py` from game logs
2. Modify `build_player_impact_profiles.py` to use stint data
3. Validate backtest improvement (expect 0.5–1.0 MAE improvement for heavily-traded teams)

### Phase 2: Forecast Roster Improvement (Near-term)
1. Build `fetch_preseason_rosters.py` using NBA API
2. Wire into `project_next_season.py` as the default roster source for forecast mode
3. Keep manual roster CSV override as fallback

### Phase 3: Frontend Enhancements (When pipeline is stable)
1. Add stint indicators to player viewers
2. Build roster timeline view (new viewer)
3. Add trade impact analysis

---

## 6. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Short stints have noisy per-game stats | Low — we use season-level BKE, not stint-level | Season BKE is portable by design |
| Preseason rosters don't predict mid-season trades | Medium — ~15% of players trade | Already handled by replacement buffer and availability discount |
| Profile row count increases (~70-80 extra rows per season) | Low — ~5% increase | Marginal compute cost |
| Minute accounting across stints may not sum perfectly | Low | Normalize to actual season totals from aggregate stats |

---

## 7. Expected Impact on Forecast Quality

**Backtest:** Should improve MAE for teams heavily affected by mid-season trades. The Mikal Bridges BKN over-projection (previously erroneously giving BKN his full 83-game stats instead of 27) is a canonical example of what this fixes. We already fixed the dedup bug, but the underlying roster assignment issue remains.

**Forecast:** Option A (carry-forward) provides marginal improvement. Option B (preseason rosters) should meaningfully improve forecast accuracy by using correct opening-day rosters rather than end-of-season rosters.

---

## 8. Decision Required

Please review and approve:
1. ✅ Core concept: team-stint architecture with split rows for multi-team players
2. ✅ Impact strategy: portable BKE scores, stint-specific minutes/team
3. ✅ Phase 1 scope: backtest accuracy first, forecast roster second
4. ✅ Frontend: deferred to Phase 3 (data layer first)
5. ✅ Option A for forecast v1 (carry-forward), Option B for v2 (preseason rosters)

Any modifications or concerns before implementation begins?
