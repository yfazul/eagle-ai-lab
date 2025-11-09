import pyautogui
import time

# ==== CONFIGURATION ====
group_name = "SE - AI-B3 - 2"
message = "One Week Completed - Pyautogui"

# Replace these with your actual coordinates
search_bar = (207, 234)      # Example only
message_box = (1034, 934)     # Example only

# ==== SCRIPT STARTS ====
print("Starting in 5 seconds... Please make sure WhatsApp Web tab is visible.")
time.sleep(5)

# Step 1: Click the search bar
pyautogui.click(search_bar)
time.sleep(30)

# Step 2: Type the group name
pyautogui.typewrite(group_name, interval=0.1)
time.sleep(30)

# Step 3: Press Enter to open chat
pyautogui.press('enter')
time.sleep(2)

# Step 4: Click the message input box
pyautogui.click(message_box)
time.sleep(10)

# Step 5: Type your message
pyautogui.typewrite(message, interval=0.05)

# Step 6: Press Enter to send
pyautogui.press('enter')

print("✅ Message sent successfully to:", group_name)
