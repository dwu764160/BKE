from playwright.sync_api import sync_playwright
import json
import time
from pathlib import Path

OUT_FILE = "data/nba_headers.json"


def main():
    Path("data").mkdir(exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=False,   # MUST be false for NBA
            slow_mo=50
        )
        context = browser.new_context()
        page = context.new_page()

        captured = {}

        def on_request(req):
            nonlocal captured
            if "stats.nba.com" in req.url and req.method == "GET":
                print(f"✓ Captured stats request: {req.url}")
                captured = req.headers

        page.on("request", on_request)

        print("Opening NBA homepage...")
        page.goto("https://www.nba.com", wait_until="domcontentloaded")

        # Force a stats request by navigating to stats page
        print("Navigating to NBA stats page...")
        page.goto(
            "https://www.nba.com/stats/players/traditional",
            wait_until="domcontentloaded",
            timeout=60000
        )

        # Wait for stats.nba.com requests
        for _ in range(30):
            if captured:
                break
            time.sleep(1)

        if not captured:
            print("❌ Failed to capture headers")
            browser.close()
            return

        # Keep only stable headers
        keep = {
            "accept",
            "accept-language",
            "origin",
            "referer",
            "user-agent",
            "x-nba-stats-origin",
            "x-nba-stats-token",
            "connection",
        }

        cleaned = {k: v for k, v in captured.items() if k.lower() in keep}

        with open(OUT_FILE, "w") as f:
            json.dump(cleaned, f, indent=2)

        print(f"\n✅ Headers saved → {OUT_FILE}")
        browser.close()


if __name__ == "__main__":
    main()
