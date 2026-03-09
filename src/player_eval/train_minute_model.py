"""
src/player_eval/train_minute_model.py
=============================================================================
Compatibility entrypoint for Step 2 minute-model training.

The canonical implementation currently lives in:
  src/simulation/train_minute_model.py

This wrapper preserves the documented command path in readme.md.
=============================================================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.simulation.train_minute_model import main


if __name__ == "__main__":
    main()
