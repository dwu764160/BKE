"""Play-by-play normalization utilities.

Implements a deterministic, lossless first-pass parser for RAW play-by-play
text. This module intentionally keeps parsing rule-based and conservative —
it extracts clock, optional score, event text, a coarse event type, and
lightweight shot parsing (distance, three, made/miss, assister).

Functions:
- parse_clock_and_score(raw_text)
- parse_shot(event_text)
- normalize_pbp_row(row)
- normalize_game_pbp(rows)

"""

import re
from typing import Optional, Dict, Any, List


EVENT_PATTERNS = {
	"SHOT_MADE": re.compile(r"\bPTS\b", re.IGNORECASE),
	"SHOT_MISS": re.compile(r"\bMISS\b", re.IGNORECASE),
	"FREE_THROW": re.compile(r"Free Throw", re.IGNORECASE),
	"REBOUND": re.compile(r"REBOUND", re.IGNORECASE),
	"TURNOVER": re.compile(r"Turnover", re.IGNORECASE),
	"STEAL": re.compile(r"STEAL", re.IGNORECASE),
	"BLOCK": re.compile(r"BLOCK", re.IGNORECASE),
	"FOUL": re.compile(r"\bFOUL\b", re.IGNORECASE),
	"SUBSTITUTION": re.compile(r"^SUB:", re.IGNORECASE),
	"TIMEOUT": re.compile(r"Timeout", re.IGNORECASE),
	"JUMP_BALL": re.compile(r"Jump Ball", re.IGNORECASE),
}


def parse_clock_and_score(raw_text: str):
	"""Parse clock and optional score from RAW_TEXT block.

	RAW_TEXT is expected to be lines where the first non-empty line is the
	clock, the second line may be a score like "7 - 5", and the remainder
	is the event description. Return (clock, away_score, home_score, event_line).
	"""
	if raw_text is None:
		return (None, None, None, "")

	lines = [l.strip() for l in str(raw_text).splitlines() if l.strip()]
	clock = lines[0] if lines else None
	home_score = away_score = None
	event_line = ""

	score_re = re.compile(r"^(\d{1,3})\s*[-–]\s*(\d{1,3})$")

	if len(lines) >= 2 and score_re.match(lines[1]):
		m = score_re.match(lines[1])
		away_score, home_score = int(m.group(1)), int(m.group(2))
		event_line = "\n".join(lines[2:]) if len(lines) > 2 else ""
	else:
		event_line = "\n".join(lines[1:]) if len(lines) > 1 else ""

	return clock, away_score, home_score, event_line


SHOT_DISTANCE_RE = re.compile(r"(\d+)'")
PLAYER_RE = re.compile(r"^([A-Za-z\.\s'-]+?)\s")


def parse_shot(event_text: str) -> Dict[str, Any]:
	"""Extract basic shot attributes from an event text.

	Returns a dict with any of: player, shot_distance, is_three, is_made, assister
	"""
	out: Dict[str, Any] = {}
	if not event_text:
		return out

	text = str(event_text)
	m = SHOT_DISTANCE_RE.search(text)
	if m:
		try:
			out["shot_distance"] = int(m.group(1))
		except Exception:
			out["shot_distance"] = None

	out["is_three"] = "3PT" in text or "3-PT" in text or "3 PT" in text
	out["is_made"] = not bool(re.search(r"\bMISS\b", text, re.IGNORECASE))

	# player name (best-effort): take leading token before first space
	pm = PLAYER_RE.match(text.replace("MISS ", ""))
	if pm:
		out["player"] = pm.group(1).strip()

	# assister: look for "(Name 2 AST)" or "(Name AST)"
	am = re.search(r"\(([^)]+?)\s+AST\)", text, re.IGNORECASE)
	if am:
		out["assister"] = am.group(1).strip()

	return out


def normalize_pbp_row(row: Dict[str, Any]) -> Dict[str, Any]:
	"""Normalize a single play-by-play row (expects dict-like with RAW_TEXT/RAW and GAME_ID).

	Produces a conservative normalized dict containing clock, scores, event_text,
	event_type, and any shot fields when relevant. Keeps original raw text.
	"""
	raw_text = None
	if isinstance(row, dict):
		raw_text = row.get("RAW_TEXT") or row.get("RAW") or row.get("DESCRIPTION")
		game_id = row.get("GAME_ID")
	else:
		# pandas Series-like
		raw_text = getattr(row, "RAW_TEXT", None) or getattr(row, "RAW", None) or getattr(row, "DESCRIPTION", None)
		game_id = getattr(row, "GAME_ID", None)

	clock, away_score, home_score, event_text = parse_clock_and_score(raw_text)

	event_type = "UNKNOWN"
	for etype, pattern in EVENT_PATTERNS.items():
		try:
			if pattern.search(event_text or ""):
				event_type = etype
				break
		except Exception:
			continue

	normalized: Dict[str, Any] = {
		"game_id": game_id,
		"clock": clock,
		"away_score": away_score,
		"home_score": home_score,
		"event_text": event_text,
		"event_type": event_type,
		"raw": raw_text,
	}

	if event_type in {"SHOT_MADE", "SHOT_MISS", "FREE_THROW"} or re.search(r"\bSHOT\b|3PT|PTS|MISS", event_text or "", re.IGNORECASE):
		normalized.update(parse_shot(event_text))

	return normalized


def normalize_game_pbp(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
	"""Normalize a list (or iterable) of rows.

	Each row may be a dict or pandas Series-like object. Returns list of dicts.
	"""
	out = []
	for r in rows:
		out.append(normalize_pbp_row(r))
	return out


if __name__ == "__main__":
	# simple smoke test
	sample = {
		"GAME_ID": "G1",
		"RAW_TEXT": "09:35\n5 - 5\nBrown 1' Driving Layup (5 PTS) (White 2 AST)",
	}
	print(normalize_pbp_row(sample))

