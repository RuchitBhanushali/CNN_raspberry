#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Realtime gesture classifier
LED colours:
    thumbs_up  -> green
    v_sign     -> blue
    background / other -> red
"""

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2
from sense_hat import SenseHat

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--model",
               default=ROOT / "models" / "mobilenet_gestures.tflite", 
               type=Path)
args = p.parse_args()
MODEL_PATH = args.model
# "gestures.tflite" - for VGG-16
# "mobilenet_gestures.tflite" - for mobilenetv2

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
in_info  = interpreter.get_input_details()[0]
IMG_SIZE = (in_info["shape"][2], in_info["shape"][1])   # width, height

MAIN_RES     = (640, 480)          # camera capture resolution
LORES_RES    = (320, 240)          # preview window resolution (must be <= MAIN)

THRESH = 0.65                       # <- for VGG-16
#THRESH = 0.85                       # <- for Mobilenetv2

# --------------------------------------------------------------------
# 1.  Initialise hardware
# --------------------------------------------------------------------
sense = SenseHat()
sense.clear()

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
in_idx  = interpreter.get_input_details()[0]["index"]
out_idx = interpreter.get_output_details()[0]["index"]
num_classes = interpreter.get_output_details()[0]["shape"][-1]

picam2 = Picamera2()
cfg = picam2.create_still_configuration(
        main  = {"size": MAIN_RES,  "format": "RGB888"},
        lores = {"size": LORES_RES, "format": "RGB888"},
        display="lores")
picam2.configure(cfg)
picam2.start()

print("[q] quit")

# --------------------------------------------------------------------
# 2.  Main loop
# --------------------------------------------------------------------
WINDOW = "camera"                       # ?? name once, top of script
cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)   # ?? create window
try:
    while True:
        frame = picam2.capture_array("main")           # 640 x 480 RGB
        lores  = cv2.resize(frame, LORES_RES)
        #cv2.rectangle(lores, (80,60), (240,180), (0,255,0), 1)  # ?? helper box
        cv2.imshow(WINDOW, lores) 
        input224 = cv2.resize(frame, IMG_SIZE,         # 224 x 224 RGB
                              interpolation=cv2.INTER_AREA)
        tensor = np.expand_dims(input224 / 255.0, 0).astype(np.float32)

        interpreter.set_tensor(in_idx, tensor)
        interpreter.invoke()
        #pred = np.argmax(interpreter.get_tensor(out_idx)[0])

        #sense.clear(COLOURS.get(pred, (255, 0, 0)))    # default red
        
        probs = interpreter.get_tensor(out_idx)[0]
        pred  = np.argmax(probs)
        conf  = float(np.max(probs))     # 0.00 � 1.00

        if conf < THRESH:                # ? new: low-confidence branch
            colour = (0, 255, 0)         # green
        elif pred == 0:
            colour = (255, 0, 0)         # red
        else:                            # pred == 1
            colour = (0, 0, 255)         # blue

        sense.clear(colour)

        # Exit key (works over VNC, optional)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    sense.clear()
    picam2.stop()
    cv2.destroyAllWindows()
