import json
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT_PATH = Path("data/nba_session.json")
TEST_GAME = "0022300001"


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        print("Opening NBA site…")
        page.goto("https://www.nba.com", timeout=60000)
        page.wait_for_timeout(4000)

        print("Opening game page…")
        page.goto(
            f"https://www.nba.com/game/{TEST_GAME}/play-by-play",
            timeout=60000
        )
        page.wait_for_timeout(5000)

        cookies = context.cookies()
        ua = page.evaluate("() => navigator.userAgent")

        session = {
            "headers": {
                "user-agent": ua,
                "referer": "https://www.nba.com/",
            },
            "cookies": cookies,
        }

        with open(OUT_PATH, "w") as f:
            json.dump(session, f, indent=2)

        print(f"✅ Session saved → {OUT_PATH}")

        browser.close()


if __name__ == "__main__":
    main()
