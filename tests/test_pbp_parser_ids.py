"""
tests/test_pbp_parser_ids.py
Sanity-checks the PBP row normalization (ID extraction & field filling).
Input: sample rows (inline) or data/historical/pbp_normalized.parquet
Output: printed normalized rows and optional ID fill-rate statistics
"""

from pathlib import Path
import sys

# Ensure src is on sys.path
# Use parents[1] to reference the repository root, then append /src
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from data_normalize.pbp_parser import normalize_pbp_row

try:
    import pandas as pd
except Exception:
    pd = None


def main():
    rows = [
        {
            "GAME_ID": "1",
            "PERIOD": 1,
            "clock": "PT05M30.00S",
            "scoreHome": 10,
            "scoreAway": 12,
            "DESCRIPTION": "J. Doe 2 PTS",
            "PLAYER1_ID": 123,
            "player2Id": 456,
            "teamId": 789,
            "RAW_TEXT": ""
        },
        {
            "GAME_ID": "2",
            "PERIOD": 2,
            "clock": "PT02M10.00S",
            "DESCRIPTION": "J. Doe MISS",
            "person1Id": 111,
            "assistPlayerId": 222,
            "blockPlayerId": 333,
            "team_id": 444,
            "RAW_TEXT": ""
        },
        {
            "GAME_ID": "3",
            "PERIOD": 3,
            "clock": "PT01M00.00S",
            "RAW": "PT01M00.00S\n100-98\nSUB in: J. Smith",
            "player1_id": 555
        }
    ]

    for i, r in enumerate(rows, 1):
        out = normalize_pbp_row(r)
        print(f"--- Row {i} ---")
        print(out)


if __name__ == "__main__":
    main()
    # Sanity check: ID fill rates (optional - will skip if pandas or file unavailable)
    def check_id_fill_rates(parquet_path: str = "data/historical/pbp_normalized.parquet"):
        if pd is None:
            print("\n--- ID Fill Rates ---\nPandas not available; skipping fill-rate check.")
            return
        try:
            df = pd.read_parquet(parquet_path)
        except FileNotFoundError:
            print(f"\n--- ID Fill Rates ---\nParquet file not found: {parquet_path}; skipping check.")
            return
        except Exception as e:
            print(f"\n--- ID Fill Rates ---\nCould not read parquet ({parquet_path}): {e}")
            return

        print("\n--- ID Fill Rates ---")
        print(f"Rows with Player1 ID: {df['player1_id'].notna().mean():.1%}")
        print(f"Rows with Team ID:    {df['team_id'].notna().mean():.1%}")

        assists = df[df['raw_text'].str.contains("AST", na=False)]
        print(f"Assists captured (Player2): {assists['player2_id'].notna().mean():.1%}")

        blocks = df[df['event_type'] == "BLOCK"]
        print(f"Blocks captured (Player3):  {blocks['player3_id'].notna().mean():.1%}")

    check_id_fill_rates()
