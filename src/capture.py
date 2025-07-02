#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from datetime import datetime
from pathlib import Path
import cv2
from picamera2 import Picamera2

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DIR_CAPTURES = ROOT / "gesture_data"

gesture   = "thumbs_up"
save_root = DIR_CAPTURES / gesture
save_root.mkdir(parents=True, exist_ok=True)
# -------------------------------------------------------------

gesture   = "background" # / "thumbs_up" / "v_sign" - change accordingly
save_root = Path("gesture_data") / gesture
save_root.mkdir(parents=True, exist_ok=True)

picam2 = Picamera2()

config = picam2.create_still_configuration(
    main  = {"size": (1640, 1232), "format": "RGB888"},  # saved images
    lores = {"size": (640,  480),  "format": "RGB888"},  # fast preview
    display="lores",
)
picam2.configure(config)
picam2.start()

print("[SPACE] snap   [Q] quit")
while True:
    preview = picam2.capture_array("lores")      # 30-fps, fits VNC
    # cv2.rectangle(preview, (160,60), (480,420), (0,255,0), 2)
    cv2.imshow("preview", preview)

    k = cv2.waitKey(1) & 0xFF
    if k == ord(" "):
        full = picam2.capture_array("main")      # 1640 � 1232
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        cv2.imwrite(str(save_root / f"{gesture}_{ts}.jpg"), full)
        print("  ? saved", ts)
    elif k == ord("q"):
        break

cv2.destroyAllWindows()
picam2.stop()
