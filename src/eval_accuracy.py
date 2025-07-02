#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate one or more TFLite gesture models on a held-out image set.

Usage
-----
    python3 eval_accuracy.py vgg16_gestures.tflite mobilenet_gestures.tflite

Assumes directory structure:
    test_data/
        background/*.jpg
        thumbs_up/*.jpg
        v_sign/*.jpg
Outputs:
    acc_results.json  –  accuracy, confusion-matrix, per-class metrics
"""

import sys, json, pathlib
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tflite_runtime.interpreter as tflite

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DIR_TEST  = ROOT / "test_data"

TEST_DIR = DIR_TEST

p = argparse.ArgumentParser()
p.add_argument("--model", required=True, type=Path)
args = p.parse_args()

# --------------------------------------------------
def load_images():
    """Return X (images), y (labels), class_names (sorted)."""
    X, y, class_names = [], [], sorted(p.name for p in TEST_DIR.iterdir() if p.is_dir())
    for label, cname in enumerate(class_names):
        for path in (TEST_DIR / cname).glob("*.jpg"):
            img = Image.open(path).convert("RGB")
            X.append(np.asarray(img))
            y.append(label)
    return np.array(X), np.array(y), class_names

# --------------------------------------------------
def eval_model(model_path, X, y):
    """Run one TFLite model and compute metrics."""
    inter = tflite.Interpreter(model_path=str(model_path))
    inter.allocate_tensors()
    inp = inter.get_input_details()[0]
    out = inter.get_output_details()[0]
    h_net, w_net = inp["shape"][1:3]        # height, width

    preds = []
    for img in X:
        h0, w0 = img.shape[:2]
        crop = img[
            (h0 - min(h0, w0)) // 2 : (h0 + min(h0, w0)) // 2,
            (w0 - min(h0, w0)) // 2 : (w0 + min(h0, w0)) // 2,
        ]
        pil = Image.fromarray(crop).resize((w_net, h_net), Image.BILINEAR)
        tensor = np.expand_dims(np.asarray(pil, np.float32) / 255.0, 0)
        inter.set_tensor(inp["index"], tensor)
        inter.invoke()
        preds.append(np.argmax(inter.get_tensor(out["index"])[0]))

    preds = np.array(preds)
    return {
        "model": model_path.name,
        "accuracy": round(float(accuracy_score(y, preds)), 3),
        "confusion": confusion_matrix(y, preds).tolist(),
        "report": classification_report(y, preds, output_dict=True),
    }

# --------------------------------------------------
def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python3 eval_accuracy.py model.tflite [model2.tflite …]")
    X, y, classes = load_images()
    results = [eval_model(pathlib.Path(p), X, y) for p in sys.argv[1:]]
    out = {"classes": classes, "results": results}
    with open("acc_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print("? wrote acc_results.json")

if __name__ == "__main__":
    main()
