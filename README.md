# CNN_raspberry
This Repository is a part of Lab Report for Gesture Recognition using CNNs on Raspberry Pi.
This discribes the degisn, instuctions and evaluation of a real-time hand-gesture recognition system deplyed on a raspberry PI 4B.
Two transfer-learning approaches VGG-16 and MobileNet V2 are compared in the terms of accuracy, model size, and Inference Latency.

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![TensorFlow Lite](https://img.shields.io/badge/TF_Lite-2.21-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This repo contains all code, models, data, and plots for our Raspberry Pi
hand-gesture demo that detects

* **👍 thumbs-up**
* **✌︎ V-sign**
* **background / unknown**

and lights the Sense HAT in **green**, **blue**, or **red**.

| Backbone            | Val accuracy | FPS on Pi 4 | Model size |
|---------------------|--------------|-------------|------------|
| **VGG-16 + GAP**    | 0.92         | 11          | 55 MiB     |
| **MobileNet V2**    | 0.90         | 21          |  7 MiB     |

*(See `Lab06_Gesture_Report.pdf` for full methodology and results.)*


## 1  Clone & install

```bash
git clone https://github.com/▸yourusername◂/CNN_raspberry.git
cd CNN_raspberry

# create env (desktop or Pi)
python -m venv ml_lab
source ml_lab/bin/activate             # Windows: ml_lab\Scripts\activate

# deps — desktop: TF; Pi: tflite_runtime wheels are fine
python -m pip install -r requirements.txt


On Raspberry Pi OS you may also need:

sudo apt update
sudo apt install libatlas-base-dev libopenjp2-7 picamera2 sense-hat

