"""
src/data_normalize/pbp_parser.py
Deterministic, regex-based parser for NBA Play-by-Play text.
"""

import re
from typing import Optional, Dict, Any, List

# -------------------------------------------------------------------------
# Regex Patterns
# -------------------------------------------------------------------------

# Matches "PTS" to signal a made basket (e.g., "25 PTS")
RE_PTS = re.compile(r"\b(\d+)\s*PTS\b", re.IGNORECASE)

# Matches "MISS" to signal a miss
RE_MISS = re.compile(r"\bMISS\b", re.IGNORECASE)

# Specific Event Keywords
PATTERNS = {
    "FREE_THROW": re.compile(r"Free Throw", re.IGNORECASE),
    "REBOUND": re.compile(r"REBOUND", re.IGNORECASE),
    "TURNOVER": re.compile(r"Turnover", re.IGNORECASE),
    "STEAL": re.compile(r"STEAL", re.IGNORECASE),
    "BLOCK": re.compile(r"BLOCK", re.IGNORECASE),
    "FOUL": re.compile(r"\bFOUL\b", re.IGNORECASE),
    "SUBSTITUTION": re.compile(r"^SUB:", re.IGNORECASE),
    "TIMEOUT": re.compile(r"Timeout", re.IGNORECASE),
    "JUMP_BALL": re.compile(r"Jump Ball", re.IGNORECASE),
    "VIOLATION": re.compile(r"Violation|Traveling|Palming|Goaltending", re.IGNORECASE),
    "INSTANT_REPLAY": re.compile(r"Instant Replay", re.IGNORECASE),
}

# Shot descriptors
RE_FIELD_GOAL_KEYWORDS = re.compile(
    r"\b(Shot|Layup|Dunk|Fadeaway|Tip|Putback|Alley Oop)\b", re.IGNORECASE
)

# -------------------------------------------------------------------------
# Extraction Helpers
# -------------------------------------------------------------------------

def parse_clock_and_score(raw_text: str):
    """Extracts clock, score, and the clean event description line."""
    if not isinstance(raw_text, str):
        return None, None, None, ""

    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    if not lines:
        return None, None, None, ""

    # Line 0 is usually clock
    clock = lines[0]
    
    # Line 1 might be score
    score_re = re.compile(r"^(\d{1,3})\s*[-–]\s*(\d{1,3})$")
    away_score = None
    home_score = None
    event_start_idx = 1

    if len(lines) > 1:
        m = score_re.match(lines[1])
        if m:
            away_score = int(m.group(1))
            home_score = int(m.group(2))
            event_start_idx = 2
    
    event_text = " ".join(lines[event_start_idx:])
    return clock, away_score, home_score, event_text

def determine_base_event_type(text: str) -> str:
    """Classifies the event text into a generic type."""
    for etype, pattern in PATTERNS.items():
        if pattern.search(text):
            return etype

    if RE_PTS.search(text) or RE_MISS.search(text) or RE_FIELD_GOAL_KEYWORDS.search(text):
        return "FIELD_GOAL"

    return "UNKNOWN"

def parse_shot_details(text: str):
    """Extracts shot details: Made/Miss, Points, 3PT status."""
    is_made = bool(RE_PTS.search(text))
    is_miss = bool(RE_MISS.search(text))
    
    points = 0
    if is_made:
        m = RE_PTS.search(text)
        if m:
            points = int(m.group(1))
    
    is_three = "3PT" in text.upper()

    return {
        "is_made": is_made,
        "is_miss": is_miss,
        "points": points,
        "is_three": is_three
    }

# -------------------------------------------------------------------------
# Main Normalizer
# -------------------------------------------------------------------------

def normalize_pbp_row(row: Dict[str, Any]) -> Dict[str, Any]:
    raw_text = row.get("RAW_TEXT") or row.get("RAW") or ""
    game_id = row.get("GAME_ID")

    clock, away, home, event_text = parse_clock_and_score(raw_text)
    base_event = determine_base_event_type(event_text)
    
    # Defaults
    final_event_type = base_event
    shot_details = {"is_made": False, "points": 0, "is_three": False}

    # Refine Field Goals into 2PT vs 3PT
    if base_event in ["FIELD_GOAL", "FREE_THROW"]:
        shot_details = parse_shot_details(event_text)
        
        if base_event == "FIELD_GOAL":
            if shot_details["is_three"]:
                final_event_type = "FIELD_GOAL_3PT"
            else:
                final_event_type = "FIELD_GOAL_2PT"

    # Construct Output
    normalized = {
        "game_id": game_id,
        "clock": clock,
        "away_score": away,
        "home_score": home,
        "event_text": event_text,
        "event_type": final_event_type,
        "raw_text": raw_text
    }
    
    # Merge shot details
    normalized.update(shot_details)

    return normalized