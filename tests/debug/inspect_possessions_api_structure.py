"""
tests/debug/inspect_possessions_api_structure.py
Debugs the 'Missing TeamStats' issue by dumping the raw API structure.
"""

import json
import os
import sys
from curl_cffi import requests

# Adjust path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

DATA_DIR = "data/historical"
HEADERS_FILE = "data/nba_headers.json"
GAME_ID = "0022301229" # One of the failing games

def load_captured_headers():
    if not os.path.exists(HEADERS_FILE):
        return None
    try:
        with open(HEADERS_FILE, 'r') as f:
            return json.load(f)
    except:
        return None

def inspect_game():
    url = "https://stats.nba.com/stats/boxscoreadvancedv2"
    
    params = {
        "GameID": GAME_ID,
        "StartPeriod": 0, "EndPeriod": 0,
        "StartRange": 0, "EndRange": 0,
        "RangeType": 0
    }
    
    headers = {
        'Accept': 'application/json, text/plain, */*',
        'Connection': 'keep-alive',
        'Origin': 'https://www.nba.com',
        'Referer': f'https://www.nba.com/game/{GAME_ID}/box-score',
        'x-nba-stats-origin': 'stats'
    }

    # Inject Captured Token
    captured = load_captured_headers()
    if captured and 'x-nba-stats-token' in captured:
        headers['x-nba-stats-token'] = captured['x-nba-stats-token']
        headers['User-Agent'] = captured.get('User-Agent', headers['User-Agent'])
        print("✅ Using Captured Headers")
    else:
        headers['x-nba-stats-token'] = 'true'
        print("⚠️ Using Fallback Headers (Might be the cause)")

    try:
        print(f"Fetching {GAME_ID}...", end=" ")
        response = requests.get(
            url, params=params, headers=headers, 
            impersonate="chrome110", timeout=15
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Response: {response.text[:200]}")
            return

        json_data = response.json()
        result_sets = json_data.get('resultSets', [])
        
        print(f"\n--- Available ResultSets ({len(result_sets)}) ---")
        for i, rs in enumerate(result_sets):
            print(f"[{i}] Name: '{rs['name']}' | Rows: {len(rs['rowSet'])}")
            if len(rs['rowSet']) > 0:
                print(f"    Columns: {rs['headers'][:5]}...")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    inspect_game()