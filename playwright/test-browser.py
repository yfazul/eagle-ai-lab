from playwright.sync_api import sync_playwright

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1366, "height": 768},
            locale="en-US",
            java_script_enabled=True
        )

        page = context.new_page()
        page.goto("https://www.google.com/search?q=cricbuzz", timeout=60000)
        page.wait_for_timeout(3000)
        print(page.title())
        browser.close()

run()
