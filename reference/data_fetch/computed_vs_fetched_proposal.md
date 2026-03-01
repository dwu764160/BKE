# Computed vs Fetched Advanced Metrics — Proposal

## Executive Summary

After researching the available data sources and comparing our computed metrics against Basketball-Reference (BRef) ground truth, this proposal recommends **continuing to compute WS/BPM/VORP internally** while fixing the identified systematic biases, rather than attempting to fetch them from external sources.

---

## 1. What We Compute Today

| Metric | Source File | Status |
|--------|-----------|--------|
| WS (Win Shares) | `compute_linear_metrics.py` | OWS close (mean diff -0.071), DWS drifts +0.375 |
| OWS (Offensive WS) | same | Reasonably accurate (std 0.316) |
| DWS (Defensive WS) | same | **Systematic overestimate** (+0.375 mean) |
| BPM (Box Plus/Minus) | same | Outlier-prone (mean +0.404, max ±3.3) |
| OBPM / DBPM | same | Split from BPM |
| VORP | same | Downstream of BPM, inherits its errors |
| GMSC_AVG | same | Custom game score; no BRef equivalent |

## 2. What Can Be Fetched From Public APIs

### NBA.com (via `nba_api`)
| Endpoint | Provides |
|----------|----------|
| `LeagueDashPlayerStats` (Advanced) | OFF_RATING, DEF_RATING, NET_RATING, USG_PCT, TS_PCT, EFG_PCT, AST_PCT, PIE, PACE |
| `PlayerEstimatedMetrics` | E_OFF_RATING, E_DEF_RATING, E_NET_RATING, E_USG_PCT |
| `BoxScoreAdvancedV3` | Per-game ORTG, DRTG, USG%, TS%, eFG%, PACE, PIE |

**NBA.com does NOT provide:** WS, OWS, DWS, BPM, OBPM, DBPM, VORP, Game Score.

These are **Basketball-Reference proprietary metrics** — BRef computes them using their own implementations of Dean Oliver's Win Shares formula and Daniel Myers' BPM 2.0 regression.

### Basketball-Reference
- **No public API** exists.
- Their [Terms of Service](https://www.sports-reference.com/termsofuse.html) explicitly prohibit automated scraping.
- The `basketball_reference_web_scraper` Python package exists but violates BRef TOS and is unreliable (rate-limited, breaks on page changes).
- Stathead subscription provides query access but not bulk API access.

**Conclusion: WS/BPM/VORP cannot be reliably fetched from any public source.**

## 3. Root Causes of Our Drift

### DWS (systematic +0.375)
Three compounding issues traced to `compute_linear_metrics.py` lines 375-480:
1. **League-average constants for opponent stats** — We use `DFG_pct=0.47`, `DOR_pct=0.25`, `Opp_FGA_pg=88` etc. for all teams. BRef uses each team's actual opponent stats, meaning bad defensive teams properly get less DWS credit.
2. **Player on-court DRTG** — We use `df['DRTG']` (noisy PBP-derived on-court defensive rating) instead of the team's overall season defensive rating.
3. **Stops formula simplification** — Our `Stops2` "team credit" uses per-minute league rates rather than team-specific values.

Top DWS overestimates confirm this: Booker +1.028, Durant +0.964, Trae Young +0.937 — all on poor defensive teams that get inflated by league-average constants.

### BPM (outlier ±3.3)
Independent from DWS (correlation r=0.026):
1. **Ad-hoc adjustments** not in BRef's formula: `AST_BIG_BONUS=14`, efficiency penalties/bonuses, `COMPRESSION=0.89`, `OFFSET=0.60`.
2. **Team adjustment method** — We use `team_avg_net_rtg * 0.25`, while BRef constrains the sum of all player BPMs (weighted by minutes) to equal the team's adjusted efficiency differential. This is a critical difference.
3. **Position/role regression** — BRef uses a specific position spectrum (1-5) with per-position coefficients. Our implementation approximates this differently.

## 4. Advantages of Computing Internally

1. **Possession-level granularity** — Our pipeline operates at the play-by-play level. We can compute WS/BPM at the game level, lineup level, or stint level — not just season aggregates.
2. **No external dependency** — No API keys, rate limits, TOS violations, or breaking scraper changes.
3. **Pipeline integration** — These metrics feed directly into our archetype classification and player profiling pipeline with full internal consistency.
4. **Customizability** — We can tune formulas to emphasize what matters for our analysis (e.g., our RAPM/xRAPM already provide superior plus-minus estimation).
5. **Historical coverage** — We control what seasons are supported without relying on third-party availability.

## 5. Disadvantages of Computing Internally

1. **DWS systematic bias** — Fixable (see recommendations below).
2. **BPM outlier sensitivity** — Fixable by switching to BRef's published BPM 2.0 methodology.
3. **Not directly BRef-comparable** — Analysts who reference BRef values will find our numbers differ. Mitigated by labeling our values as "BKE-computed."
4. **Maintenance burden** — Must track methodology changes if BRef updates their formulas.

## 6. Recommendations

### Fix DWS (Priority: HIGH)
- Replace hardcoded league-average constants with **team-specific opponent stats** from `LeagueDashTeamStats` (already available via our fetch pipeline).
- Use **team overall DRTG** instead of player on-court DRTG for the baseline calculation.
- Expected improvement: reduce mean drift from +0.375 to ~±0.1.

### Fix BPM (Priority: HIGH)
- Rewrite to match **BPM 2.0 published methodology** (coefficients are fully published on BRef):
  - Implement proper position regression (TRB%, STL%, PF%, AST%, BLK%).
  - Implement offensive role regression.
  - Use published per-position coefficients for the 13-variable linear regression.
  - Implement proper team adjustment: sum of (player BPM × minutes%) must equal team adjusted efficiency.
  - Remove all ad-hoc bonuses/penalties/compression.
- Expected improvement: reduce mean drift from +0.404 to ~±0.2 and eliminate ±3.3 outliers.

### Keep VORP as-is
- Formula is trivial: `[BPM - (-2.0)] * (poss_pct) * (team_games/82)`. Once BPM is fixed, VORP follows.

### Keep GMSC_AVG
- This is our custom metric. No BRef equivalent to compare against.

### Label in UI
- Add "(BKE)" suffix or a tooltip in the HTML viewer to indicate these are internally computed, not fetched from BRef.

---

## 7. Decision Matrix

| Metric | Fetch from API? | Compute ourselves? | Recommendation |
|--------|:-:|:-:|---|
| WS/OWS/DWS | ❌ Not available | ✅ Yes | **Compute** (fix DWS constants) |
| BPM/OBPM/DBPM | ❌ Not available | ✅ Yes | **Compute** (rewrite to BPM 2.0 spec) |
| VORP | ❌ Not available | ✅ Yes | **Compute** (fix follows from BPM fix) |
| GMSC_AVG | ❌ N/A | ✅ Yes | **Compute** (custom metric) |
| USG%/TS%/ORTG/DRTG | ✅ NBA.com API | ✅ Yes | **Fetch** (already doing this via profiles) |
| RAPM/xRAPM | ❌ Proprietary | ✅ Yes | **Compute** (our flagship metrics) |

## 8. Implementation Priority

1. **Immediate (this session):** Surface existing WS/BPM/VORP/GMSC in the HTML viewer as-is with a "BKE-computed" note.
2. **Next sprint:** Fix DWS by integrating team-specific opponent stats.
3. **Following sprint:** Rewrite BPM to match BPM 2.0 published spec.
4. **Validation:** Re-run `compare_bref_advanced_metrics.py` after each fix to track drift reduction.
