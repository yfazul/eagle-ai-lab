from playwright.sync_api import sync_playwright
import csv

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=50)
        page = browser.new_page()

        # --- Step 1: Open Cricbuzz ---
        page.goto("https://www.cricbuzz.com")
        page.wait_for_timeout(2000)

        # --- Step 2: Hover Series and click ICC Women's World Cup 2025 ---
        page.hover("xpath=/html/body/div/div[4]/div[1]/div[5]/a")
        page.wait_for_timeout(1000)
        page.click("xpath=/html/body/div/div[4]/div[1]/div[5]/div/a[1]")
        page.wait_for_timeout(3000)

        # --- Step 3: Click Schedule & Results tab ---
        page.click("xpath=//*[@id='main-nav']//a[contains(text(), 'Schedule & Results')]")
        page.wait_for_timeout(3000)

        # --- Step 4: Scroll and click match on Sun, Nov 2 2025 ---
        match_xpath = "/html/body/div/main/div[2]/div[1]/div/div/div[31]/div/div/div/a/div[3]"
        page.eval_on_selector(f"xpath={match_xpath}", "el => el.scrollIntoView()")
        page.wait_for_timeout(1000)
        page.click(f"xpath={match_xpath}")
        page.wait_for_timeout(3000)

        # --- Step 5: Click Scorecard tab ---
        page.click("xpath=//*[@id='main-nav']//a[contains(text(), 'Scorecard')]")
        page.wait_for_timeout(5000)
        print("✅ Opened Scorecard tab")

        # --- Step 6: Select first innings container ---
        innings_div = page.query_selector(
            'xpath=//div[starts-with(@id,"scard-team-") and contains(@id,"-innings-1")]'
        )
        if not innings_div:
            print("⚠️ Could not find first innings scorecard")
            input("Press Enter to close the browser...")
            browser.close()
            return

        # --- Step 7: Select all batsman rows ---
        batsman_rows = innings_div.query_selector_all(
            'xpath=.//div[contains(@class,"scorecard-bat-grid")]'
        )

        # Limit to first 8 batsmen
        batsman_rows = batsman_rows[:8]

        scorecard_data = []
        for row in batsman_rows:
            children = row.query_selector_all("xpath=./*")
            if len(children) < 7:  # ensure row has enough columns
                continue

            # --- Batter and Dismissal ---
            batter_div = children[0]
            name_elem = batter_div.query_selector("a")
            name = name_elem.inner_text().strip() if name_elem else ""
            if not name:
                continue  # skip rows without a batter

            dismissal_div = batter_div.query_selector("div")
            dismissal = dismissal_div.inner_text().strip() if dismissal_div else ""

            # --- R, B, 4s, 6s, SR ---
            runs = children[1].inner_text().strip()
            balls = children[2].inner_text().strip()
            fours = children[3].inner_text().strip()
            sixes = children[4].inner_text().strip()
            sr = children[5].inner_text().strip()

            scorecard_data.append([name, dismissal, runs, balls, fours, sixes, sr])

        # --- Step 8: Write CSV ---
        with open("scorecard.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Batter", "Dismissal", "R", "B", "4s", "6s", "SR"])
            writer.writerows(scorecard_data)

        print("✅ Scorecard saved to scorecard.csv with first 8 batsmen")
        page.wait_for_timeout(2000)
        browser.close()

run()
