#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark TFLite models for FPS, mean latency and RAM use.

Usage:
  python3 benchmark_models.py vgg16_gestures.tflite mobilenet_gestures.tflite
"""

import sys, time, psutil, os, json
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image
from picamera2 import Picamera2

N_FRAMES = 120               # how many frames per model
WARMUP   = 10                # discarded warm-up runs

if len(sys.argv) < 3:
    sys.exit("python benchmark_models.py  <modelA.tflite>  <modelB.tflite>")

# Dummy frame (all zeros) � much faster than capturing real video
def dummy_input(shape):
    h, w = shape[1:3]
    return np.zeros((1, h, w, 3), dtype=np.float32)

results = []

for model_path in sys.argv[1:]:
    inter = tflite.Interpreter(model_path=model_path)
    inter.allocate_tensors()
    inp = inter.get_input_details()[0]["index"]
    shp = inter.get_input_details()[0]["shape"]

    # Warm-up
    tensor = dummy_input(shp)
    for _ in range(WARMUP):
        inter.set_tensor(inp, tensor)
        inter.invoke()

    t0 = time.perf_counter()
    for _ in range(N_FRAMES):
        inter.set_tensor(inp, tensor)
        inter.invoke()
    t1 = time.perf_counter()

    fps   = (N_FRAMES) / (t1 - t0)
    msec  = 1000 * (t1 - t0) / N_FRAMES
    rss   = psutil.Process().memory_info().rss / (1024*1024)

    results.append({
        "model": os.path.basename(model_path),
        "input": "{}x{}".format(shp[2], shp[1]),   # <- ascii only
        "fps":   round(fps, 1),
        "latency_ms": round(msec, 1),
        "rss_mb": round(rss, 1),
    })

print(json.dumps(results, indent=2))
with open("bench_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("? wrote bench_results.json")
