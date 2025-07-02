#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live-stream evaluation of one or more TFLite gesture models.

Keys while running
------------------
    t  : label current frames as thumbs_up
    v  : label current frames as v_sign
    b  : label current frames as background
    q  : quit early

Usage
-----
    python3 live_eval.py vgg16_gestures.tflite mobilenet_gestures.tflite

The script grabs batches of live frames, applies centre-crop + resize
(the same as predict_live.py), records model predictions, and asks you
to press t / v / b to label the batch.  At the end it prints accuracy
and a confusion matrix for each model and stores them in live_acc.json.
"""

import sys, json, pathlib, cv2, numpy as np, tflite_runtime.interpreter as tflite
from picamera2 import Picamera2
from sklearn.metrics import accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
LABEL_KEYS  = {"t": 1, "v": 2, "b": 0}
LABEL_NAMES = ["background", "thumbs_up", "v_sign"]

# -------- CLI -------------------------------------------------------
import argparse, json, cv2, numpy as np
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("models", nargs="+", type=Path,
               help="TFLite model(s) relative to project root")
p.add_argument("--frames", type=int, default=120, help="total frames")
p.add_argument("--batch",  type=int, default=10,  help="frames per label press")
p.add_argument("--out",    type=Path, default=ROOT/"live_acc.json")
args = p.parse_args()



LABEL_KEYS  = {"t": 1, "v": 2, "b": 0}
LABEL_NAMES = ["background", "thumbs_up", "v_sign"]
BATCH       = 10     # frames per manual label
MAX_FRAMES  = 120    # stop after this many

WINDOW = "Live-Eval (click here, press t / v / b / q)"
cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

def load_model(path):
    inter = tflite.Interpreter(model_path=str(path))
    inter.allocate_tensors()
    inp = inter.get_input_details()[0]
    out = inter.get_output_details()[0]
    net_h, net_w = inp["shape"][1:3]
    return inter, inp["index"], out["index"], (net_w, net_h)

MODELS = [load_model(m) for m in args.models]

def preprocess(frame, net_size):
    h0, w0 = frame.shape[:2]
    s = min(h0, w0)
    crop = frame[(h0 - s)//2 : (h0 + s)//2,
                 (w0 - s)//2 : (w0 + s)//2]
    return cv2.resize(crop, net_size).astype(np.float32) / 255.0

def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python3 live_eval.py model1.tflite [model2.tflite …]")

    models = [load_model(pathlib.Path(p)) for p in sys.argv[1:]]
    labels, preds = [[] for _ in models], [[] for _ in models]

    picam2 = Picamera2()
    cfg = picam2.create_still_configuration(
            main = {"size": (640, 480), "format": "RGB888"},   # full-res RGB
            lores= {"size": (320, 240), "format": "RGB888"},   # preview RGB
            display = "lores")
    picam2.configure(cfg)
    picam2.start()

    current_label, n_frames = None, 0
    print("Press T / V / B to set label; Q to quit")

    try:
        while n_frames < MAX_FRAMES:
            frame = picam2.capture_array()

            # draw helper square (same crop)
            h0, w0 = frame.shape[:2]
            s = min(h0, w0)
            top, left = (h0-s)//2, (w0-s)//2
            cv2.rectangle(frame, (left, top), (left+s, top+s), (0,255,0), 1)

            # show preview
            preview = picam2.capture_array("lores")
            cv2.rectangle(preview, (80,60), (240,180), (0,255,0), 1)  # helper box
            cv2.imshow(WINDOW, preview)

            k = cv2.waitKey(1) & 0xFF
            if k in map(ord, "tvb"):
                current_label = LABEL_KEYS[chr(k)]
                print(f"\nLabel set to: {LABEL_NAMES[current_label]}")
            if k == ord("q"):
                break
            if current_label is None:
                continue

            for m, (inter, in_idx, out_idx, sz) in enumerate(models):
                inp = preprocess(frame, sz)
                inter.set_tensor(in_idx, np.expand_dims(inp,0))
                inter.invoke()
                preds[m].append(int(np.argmax(inter.get_tensor(out_idx)[0])))
                labels[m].append(current_label)

            n_frames += 1
            if n_frames % BATCH == 0:
                print(f"  captured {n_frames}/{MAX_FRAMES} frames")

    finally:
        picam2.stop()
        cv2.destroyAllWindows()

    results = []
    for m, path in enumerate(sys.argv[1:]):
        acc  = accuracy_score(labels[m], preds[m])
        conf = confusion_matrix(labels[m], preds[m]).tolist()
        results.append({"model": pathlib.Path(path).name,
                        "accuracy": round(float(acc), 3),
                        "frames": len(labels[m]),
                        "confusion": conf})

    print(json.dumps(results, indent=2))
    json.dump(results, open("live_acc.json", "w"), indent=2)
    print("? wrote live_acc.json")

if __name__ == "__main__":
    main()
