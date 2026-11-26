"""Capture center-screen examples while the calibrated controller aim is held."""
from datetime import datetime
from pathlib import Path
from time import sleep, monotonic
import ctypes

import aimi_local
import cv2
import mss
from controller_hold import calibrate
from utils.grab import screen

capture_root = Path.home() / "Downloads" / "ROB RUN detection test" / "live"
held = calibrate()
session = capture_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
session.mkdir(parents=True)
print(f"Saving up to 600 frames to {session}")
print("Switch to ROB RUN. Hold your aim trigger to save frames. Ctrl+C exits.")
user32 = ctypes.windll.user32
user32.GetForegroundWindow.restype = ctypes.c_void_p
with mss.mss() as capture:
    monitor = capture.monitors[1]
Wd, Hd = monitor["width"], monitor["height"]
region = (max(0, Wd//2-320), max(0, Hd//2-320),
          min(Wd-1, Wd//2+319), min(Hd-1, Hd//2+319))
count = 0
next_capture = 0.0
try:
    while count < 600:
        title = ctypes.create_unicode_buffer(512)
        user32.GetWindowTextW(ctypes.c_void_p(user32.GetForegroundWindow()), title, len(title))
        active_game = "rob run" in title.value.casefold() or "robrun" in title.value.casefold()
        if active_game and held() and monotonic() >= next_capture:
            bgr = cv2.cvtColor(screen(region=region), cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(session / f"frame-{count + 1:04d}.jpg"), bgr):
                raise OSError("Could not save capture frame.")
            count += 1
            next_capture = monotonic() + 0.5
        sleep(0.03)
except KeyboardInterrupt:
    pass
print(f"Saved {count} frames.")
