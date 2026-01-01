import pandas as pd
import re

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def extract_event_line(raw_text: str) -> str:
    """
    Extract the actual play description from RAW_TEXT.
    NBA PBP format:
      clock
      score (optional)
      event description
    """
    if not isinstance(raw_text, str):
        return ""

    lines = [l.strip() for l in raw_text.split("\n") if l.strip()]

    # Walk backwards to find the event line
    for line in reversed(lines):
        # Skip score lines like "2 - 5"
        if re.match(r"^\d+\s*-\s*\d+$", line):
            continue
        # Skip clock lines like "11:38"
        if re.match(r"^\d{1,2}:\d{2}$", line):
            continue
        return line

    return ""


def classify_event(event: str) -> str:
    """
    Classify normalized NBA PBP event types.
    """
    e = event.upper()

    # -------------------------
    # Shot attempts
    # -------------------------
    if e.startswith("MISS"):
        return "SHOT_MISS"

    if (
        "JUMP SHOT" in e
        or "LAYUP" in e
        or "DUNK" in e
        or "HOOK SHOT" in e
        or "TIP" in e
        or "PUTBACK" in e
        or "FADEAWAY" in e
        or "ALLEY OOP" in e
    ):
        return "SHOT_MADE"

    # -------------------------
    # Free throws
    # -------------------------
    if "FREE THROW" in e:
        return "FREE_THROW"

    # -------------------------
    # Rebounds
    # -------------------------
    if "REBOUND" in e:
        return "REBOUND"

    # -------------------------
    # Turnovers
    # -------------------------
    if "TURNOVER" in e:
        return "TURNOVER"

    # -------------------------
    # Fouls
    # -------------------------
    if "FOUL" in e:
        return "FOUL"

    # -------------------------
    # Steals / Blocks
    # -------------------------
    if "STEAL" in e:
        return "STEAL"

    if "BLOCK" in e:
        return "BLOCK"

    # -------------------------
    # Substitutions
    # -------------------------
    if e.startswith("SUB:"):
        return "SUBSTITUTION"

    # -------------------------
    # Timeouts
    # -------------------------
    if "TIMEOUT" in e:
        return "TIMEOUT"

    # -------------------------
    # Jump balls
    # -------------------------
    if "JUMP BALL" in e:
        return "JUMP_BALL"

    # -------------------------
    # Violations
    # -------------------------
    if "VIOLATION" in e or "TRAVELING" in e:
        return "VIOLATION"

    # -------------------------
    # Instant replay
    # -------------------------
    if "INSTANT REPLAY" in e:
        return "INSTANT_REPLAY"

    return "UNKNOWN"


# ------------------------------------------------------------
# Main normalization step
# ------------------------------------------------------------

def normalize_step1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 1: Normalize RAW_TEXT into EVENT_TEXT + EVENT_TYPE
    """

    df = df.copy()

    df["EVENT_TEXT"] = df["RAW_TEXT"].apply(extract_event_line)
    df["EVENT_TYPE"] = df["EVENT_TEXT"].apply(classify_event)

    return df


# ------------------------------------------------------------
# CLI usage
# ------------------------------------------------------------

if __name__ == "__main__":
    df = pd.read_parquet("data/historical/pbp_raw.parquet")
    df = normalize_step1(df)

    print(df["EVENT_TYPE"].value_counts(dropna=False))

    df.to_parquet(
        "data/historical/pbp_normalized_step1.parquet",
        index=False,
    )
