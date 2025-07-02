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

```

On Raspberry Pi OS you may also need:
```bash
sudo apt update
sudo apt install libatlas-base-dev libopenjp2-7 picamera2 sense-hat
```

2 Collect data (Pi only)
```bash
python src/capture_live.py  --gesture thumbs_up
python src/capture_live.py  --gesture v_sign
python src/capture_live.py  --gesture background
```
Images land in gesture_data/<gesture>/.


3 Train in Colab
Open the notebooks:
> notebooks/train_vgg16.ipynb
> notebooks/train_mobilenet.ipynb

Run all cells.
Download the generated .tflite and drop it into models/.


4 Real-time inference (Pi)
```bash
python src/predict_live.py  --model models/mobilenet_gestures.tflite
```
LED glows green (👍), blue (✌︎), or red (background).


5 Benchmark & accuracy
```bash
# speed / latency / RAM
python src/benchmark_models.py models/*.tflite

# offline accuracy on held-out set
python src/eval_accuracy.py   --model models/mobilenet_gestures.tflite
#    (add extra --model  ... for VGG)

# live accuracy with manual labelling (press t / v / b keys)
python src/live_eval.py models/*.tflite
```

Plots and CSV metrics are generated with:

```bash
python src/make_figures.py
```


Hardware & Software

Raspberry Pi 4B @1.5 GHz, 4 GB
Camera V2 (IMX-219) ● Sense HAT 8×8 LED
TensorFlow Lite 2.21 (CPU, XNNPACK)
Python 3.11 • OpenCV 4.10 • Picamera2 0.3.2


Installation gotchas

| Symptom                                        | Fix                                                     |
| ---------------------------------------------- | ------------------------------------------------------- |
| `ImportError: _ARRAY_API not found`            | `pip install "numpy<2"` before installing OpenCV on Pi. |
| Blurry preview over VNC                        | Request `format="RGB888"` and use lo-res stream.        |
