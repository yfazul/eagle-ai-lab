import webbrowser
import time
import pyautogui
import pyperclip

def paste_text(text):
    # Use clipboard paste to avoid typing errors and speed
    pyperclip.copy(text)
    pyautogui.hotkey("ctrl", "v")  # on macOS use ("command", "v") if needed

def main():
    query = "South Africa vs Aus score"

    # 1. Open default browser to google
    webbrowser.open("https://www.google.com")
    time.sleep(3)  # wait for browser to open and page to load

    # 2. Click the address/search bar (this may vary by browser)
    # If your browser starts with focus in the address bar, you can skip this clicking step.
    # Otherwise, move mouse to the approximate search box area or use hotkey to focus.
    pyautogui.hotkey("ctrl", "l")  # focus address bar / search bar (works in most browsers)
    time.sleep(0.3)

    # 3. Paste the query and press ENTER
    paste_text(query)
    time.sleep(0.2)
    pyautogui.press("enter")
    time.sleep(2)  # wait for search results to load

    # 4. Press TAB a few times to reach first link and press ENTER (heuristic)
    # The number of tabs may vary; you can adjust. Here we press Tab 5 times then Enter.
    for _ in range(5):
        pyautogui.press("tab")
        time.sleep(0.2)
    pyautogui.press("enter")

    # Done
    print("Script finished. If it didn't click the first result, adjust TAB count or use Selenium.")

if __name__ == "__main__":
    main()
