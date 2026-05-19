# CLAUDE.md — BKE (Basketball KPI Engine)

Guidance for Claude Code sessions working in this repository.

## Repo essentials

- Possession-level NBA player-impact pipeline: fetch → normalize → features →
  compute metrics/archetypes → BKE modeling → simulation/forecast → viewers.
- **`data/`, `reports/`, `models/`, `aggregate/` are gitignored.** A fresh
  clone has code + reference docs only. Most scripts (`validate_sim.py`, app
  viewers) will `FileNotFoundError` until the data pipeline is run or data is
  restored from a `data_backup_*` snapshot.
- Recorded validation numbers live in `reference/**/*.md` and `loop/*.txt`
  (dated, append-only narrative snapshots), not in tracked JSON.
- Do not edit `loop/*DO_NOT_CHANGE*.txt` checkbox formats — automation parses
  them.
- Tests: `pytest -q` (currently only data-free `tests/test_simulation_core.py`,
  3 checks).

## Cross-Repo References

### Sister Repo: Robinhood Trading Bot

This repo's simulation and prediction output feeds Sleeve C
(Prediction Markets) in the Robinhood trading bot's alpha thesis.

Robinhood alpha thesis:   docs/alpha-thesis.md in the Robinhood repo
BKE performance handoff:  docs/for-alpha-thesis.md (this repo)

To fetch this repo's performance doc from a Robinhood CC session:

    gh api repos/Daniel-Wu-Github/BKE/contents/docs/for-alpha-thesis.md?ref=personal \
      | python3 -c "import sys,json,base64; d=json.load(sys.stdin); \
        print(base64.b64decode(d['content']).decode())"

To fetch the Robinhood alpha thesis from a BKE CC session:

    gh api repos/Daniel-Wu-Github/robinhood/contents/docs/alpha-thesis.md?ref=main \
      | python3 -c "import sys,json,base64; d=json.load(sys.stdin); \
        print(base64.b64decode(d['content']).decode())"

### When to Regenerate docs/for-alpha-thesis.md

Re-run the full investigation whenever:
  - A new BKE version is released (new bke_v* report appears in reports/)
  - validate_sim.py is re-run on new season data
  - The alpha thesis in the Robinhood repo updates Sleeve C parameters
  - Any loop/ file marks a major milestone complete

### Current Sleeve C status (as of 2026-05-19)

Per `docs/for-alpha-thesis.md`: **NOT alpha-ready.** Headline Brier ≈ 0.21 is
in-sample/leaky (repo docs admit same-season leakage). Leakage-free forecast is
MAE ≈ 9 wins (~3-4 worse than Vegas) with **no game-level Brier** and **zero
market-price comparison**. Recommended live allocation: **$0 / research-only**
until a walk-forward game-level test vs. Kalshi closing lines shows positive
closing-line value.
