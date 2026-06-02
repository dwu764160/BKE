# src/data/ — Schema Contract

Single module that enforces the canonical parquet schema across the entire pipeline.

## Files

| File | Purpose |
|------|---------|
| `schema_contract.py` | Public API: `canonicalize`, `load_standardized`, `save_standardized` |

## API

```python
from src.data.schema_contract import load_standardized, save_standardized, canonicalize

# Read any parquet and canonicalize in one call
df = load_standardized("data/historical/team_game_logs.parquet")

# Canonicalize an in-memory DataFrame (no I/O)
df = canonicalize(df)

# Write with canonicalization applied before save
save_standardized(df, "data/processed/output.parquet")
```

## Canonicalization rules (applied by `canonicalize()`)

1. Lowercase all column names (`SEASON` → `season`, `PLAYER_ID` → `player_id`)
2. Drop case-variant duplicate columns (keep first by column order)
3. If `season_id` present and `season` absent: rename to `season`
4. If both `season` and `season_id` present: drop `season_id`
5. Normalize `season` column: `'22024'` → `'2024-25'`
6. Normalize `*_id` columns: strip trailing `.0` from string representations

## Usage contract

Every script that reads a parquet must use `load_standardized` instead of `pd.read_parquet`.  
Every script that writes a parquet must use `save_standardized` instead of `df.to_parquet`.  
Raw external fetch scripts use `save_standardized` on their write path.

This ensures `load_standardized` is effectively a no-op on already-canonical files.
