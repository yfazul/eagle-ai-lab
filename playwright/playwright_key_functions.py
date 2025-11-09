from playwright.async_api import async_playwright
import asyncio
async def playwright_function():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        pages = await browser.new_page()

        #navigation

        await pages.goto("https://www.google.com")

        await pages.wait_for_timeout(10000)
        await browser.close()

        #CSS Selector

if __name__ == "__main__":
    asyncio.run(playwright_function())
