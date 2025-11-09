import pyautogui
import time

time.sleep(10)
x,y=pyautogui.position()

print(f'x: {x}, y:{y}')