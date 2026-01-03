"""
src/data_normalize/pbp_parser.py
Deterministic, regex-based parser for NBA Play-by-Play text.
Handles both legacy text-only data and new JSON-derived structure.
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

# ISO Clock Format (PT12M00.00S)
RE_ISO_CLOCK = re.compile(r"PT(\d+)M(\d+)(\.\d+)?S")

# Specific Event Keywords
PATTERNS = {
    "FREE_THROW": re.compile(r"Free Throw", re.IGNORECASE),
    "REBOUND": re.compile(r"REBOUND", re.IGNORECASE),
    "TURNOVER": re.compile(r"Turnover", re.IGNORECASE),
    "STEAL": re.compile(r"STEAL", re.IGNORECASE),
    "BLOCK": re.compile(r"BLOCK", re.IGNORECASE),
    "FOUL": re.compile(r"\bFOUL\b", re.IGNORECASE),
    # UPDATED: Relaxed to match "SUB in" / "SUB out"
    "SUBSTITUTION": re.compile(r"^SUB", re.IGNORECASE), 
    "TIMEOUT": re.compile(r"Timeout", re.IGNORECASE),
    "JUMP_BALL": re.compile(r"Jump Ball", re.IGNORECASE),
    "VIOLATION": re.compile(r"Violation|Traveling|Palming|Goaltending", re.IGNORECASE),
    "INSTANT_REPLAY": re.compile(r"Instant Replay", re.IGNORECASE),
    # UPDATED: Added Period markers
    "PERIOD": re.compile(r"^Period", re.IGNORECASE),
}

# Shot descriptors
RE_FIELD_GOAL_KEYWORDS = re.compile(
    r"\b(Shot|Layup|Dunk|Fadeaway|Tip|Putback|Alley Oop)\b", re.IGNORECASE
)

# -------------------------------------------------------------------------
# Extraction Helpers
# -------------------------------------------------------------------------

def clean_clock(raw_clock: str) -> str:
    """Converts ISO format (PT12M00.00S) to standard MM:SS."""
    if not raw_clock or not isinstance(raw_clock, str):
        return None
    
    m = RE_ISO_CLOCK.search(raw_clock)
    if m:
        mm = int(m.group(1))
        ss = int(m.group(2))
        return f"{mm:02d}:{ss:02d}"
    
    return raw_clock.strip()

def parse_clock_and_score(raw_text: str):
    """Fallback text extractor."""
    if not isinstance(raw_text, str):
        return None, None, None, ""

    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    if not lines:
        return None, None, None, ""

    clock = clean_clock(lines[0])
    score_re = re.compile(r"^(\d{1,3})\s*[-–]\s*(\d{1,3})$")
    away = None
    home = None
    event_start_idx = 1

    if len(lines) > 1:
        m = score_re.match(lines[1])
        if m:
            away = int(m.group(1))
            home = int(m.group(2))
            event_start_idx = 2
    
    event_text = " ".join(lines[event_start_idx:])
    return clock, away, home, event_text

def determine_base_event_type(text: str) -> str:
    """
    Classifies the event text into a generic type using STRICT PRIORITY.
    Priority is critical to correctly classify 'Blocked Shots' as 'FIELD_GOAL'.
    """
    if not text:
        return "UNKNOWN"

    # 1. Structural & Game Flow (High confidence, rarely ambiguous)
    for etype in ["PERIOD", "TIMEOUT", "SUBSTITUTION", "VIOLATION", "JUMP_BALL", "INSTANT_REPLAY"]:
        if PATTERNS[etype].search(text):
            return etype

    # 2. Free Throws (Specific, distinct from Field Goals)
    if PATTERNS["FREE_THROW"].search(text):
        return "FREE_THROW"

    # 3. Field Goals (THE FIX)
    # Check for "MISS", "PTS", or Shot Keywords BEFORE checking for Block/Steal.
    # This ensures "MISS ... - blocked" is typed as FIELD_GOAL (Missed), not BLOCK.
    if RE_PTS.search(text) or RE_MISS.search(text) or RE_FIELD_GOAL_KEYWORDS.search(text):
        return "FIELD_GOAL"

    # 4. Other Stats (Block, Steal, Rebound, Turnover, Foul)
    # Only standalone block events (e.g. "J. Brown BLOCK") should fall through to here.
    for etype in ["BLOCK", "STEAL", "REBOUND", "TURNOVER", "FOUL"]:
        if PATTERNS[etype].search(text):
            return etype

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
    # 1. Extract Explicit Columns
    game_id = row.get("GAME_ID")
    period = row.get("PERIOD")
    clock = clean_clock(row.get("clock"))
    home_score = row.get("scoreHome")
    away_score = row.get("scoreAway")
    event_text = row.get("DESCRIPTION")
    
    # 2. Fallback to Raw Text
    raw_text = row.get("RAW_TEXT") or row.get("RAW") or ""
    if not clock or not event_text:
        parsed_clock, parsed_away, parsed_home, parsed_text = parse_clock_and_score(raw_text)
        if not clock: clock = parsed_clock
        if not event_text: event_text = parsed_text
        if home_score is None: home_score = parsed_home
        if away_score is None: away_score = parsed_away

    # 3. Determine Event Type & Details
    if not event_text: event_text = ""
    
    base_event = determine_base_event_type(event_text)
    
    final_event_type = base_event
    shot_details = {"is_made": False, "points": 0, "is_three": False}

    if base_event in ["FIELD_GOAL", "FREE_THROW"]:
        shot_details = parse_shot_details(event_text)
        
        if base_event == "FIELD_GOAL":
            if shot_details["is_three"]:
                final_event_type = "FIELD_GOAL_3PT"
            else:
                final_event_type = "FIELD_GOAL_2PT"

    # 4. Construct Output
    normalized = {
        "game_id": game_id,
        "period": period,
        "clock": clock,
        "away_score": away_score,
        "home_score": home_score,
        "event_text": event_text,
        "event_type": final_event_type,
        "raw_text": raw_text
    }
    normalized.update(shot_details)

    return normalized