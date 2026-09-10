# CSE60685 FA26 Lab 1

Deploy pretrained MobileNetV3-Small and SqueezeNet 1.1 image classifiers on a Raspberry Pi 4B using ONNX Runtime.

## Quick start on Raspberry Pi

Use Raspberry Pi OS 64-bit with Python 3.13 or 3.11.
Download the public course repository directly on the Pi.

```bash
sudo apt update
sudo apt install -y git vim python3-venv python3-pip
cd ~
git clone --depth 1 "https://github.com/guoyb17/CSE60685-FA26-Lab-1.git"
cd ~/CSE60685-FA26-Lab-1
python3 -m venv env
source env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python check_environment.py
python classify.py --model mobilenet_v3_small --image samples/chelsea.png
```

The `.onnx` files, class names, and three sample images are included.
The environment check verifies their checksums and runs both models.
If a setup command fails, resolve it before continuing.
Reactivate `env` after starting a new terminal, and preserve an existing project folder if you are resuming earlier work.

Optional: compare predictions on all three images and save them:

```bash
python classify.py --model mobilenet_v3_small --image samples/chelsea.png samples/coffee.png samples/rocket.jpg --output results/mobilenet_predictions.json
python classify.py --model squeezenet1_1 --image samples/chelsea.png samples/coffee.png samples/rocket.jpg --output results/squeezenet_predictions.json
```

Scores are softmax outputs across ImageNet classes.

## Add the timer

Complete `time_inference` in `latency.py` using Tutorial Step 4: record `start = perf_counter_ns()` before `session.run`, record `end = perf_counter_ns()` immediately after it, and return `(end - start) / 1000000` instead of the placeholder `None`.
Then run:

```bash
python latency.py --model mobilenet_v3_small --image samples/chelsea.png
```

The starter intentionally stops with an instruction message until the timer is added.
Its surrounding code supplies 10 warm-ups, 100 timed calls, and the median calculation.
Make exercise edits in this file; use the supplied `benchmark.py` unchanged for report measurements.

## Latency measurements

Run these commands separately on an otherwise idle Pi, using comparable cooling and power conditions:

```bash
python benchmark.py --model mobilenet_v3_small --threads 1 --warmup 10 --runs 100 --output results/mobilenet_t1
python benchmark.py --model squeezenet1_1 --threads 1 --warmup 10 --runs 100 --output results/squeezenet_t1
```

Report `inference.median_ms` for these two runs.
This is the inference-only median in milliseconds, not `end_to_end.median_ms`.
Use the same input and Pi conditions for both.

Optional thread comparison:

```bash
python benchmark.py --model mobilenet_v3_small --threads 4 --warmup 10 --runs 100 --output results/mobilenet_t4
python benchmark.py --model squeezenet1_1 --threads 4 --warmup 10 --runs 100 --output results/squeezenet_t4
```

All commands use `samples/chelsea.png` by default.
Each writes a new directory containing `summary.json` and `raw_timings.csv`.
Each command performs 10 untimed warm-ups and 100 timed calls for each of two phases: `inference` and `end_to_end`.
The CSV therefore contains 200 measurement rows plus a header; warm-ups are not recorded.
Existing outputs are preserved;
use a new name for a repeat.

| Measurement | Timed work |
| --- | --- |
| `inference` | Synchronous CPU `session.run` with a prepared input tensor, including Python call overhead |
| `end_to_end` | Image open/decode, EXIF/RGB conversion, resize/crop/normalization, inference, softmax, and top five labels |

Both exclude session initialization, warm-ups, printing, file output, checksums, and telemetry.
Repeated reads may use the OS file cache.
Results include all timed samples, median and p95, environment details, model/image hashes, and available Pi telemetry.
`null` telemetry means unavailable.

## Camera check

Before attaching or reseating the camera, run `sudo poweroff`, wait for shutdown to finish, and disconnect power, following the hardware setup instructions.
With the virtual environment active:

```bash
rpicam-hello --list-cameras
python camera_check.py --output results/camera_check.jpg
python classify.py --model mobilenet_v3_small --image results/camera_check.jpg
```

Install `rpicam-apps` with APT if the OS camera commands are missing.
The helper uses `rpicam-still --nopreview`, so SSH does not require a preview window.
It checks that a JPEG can be captured and decoded.
It refuses to overwrite an existing file; use a new output filename each time you repeat the capture.

## Shut down safely

Save your work, then run `sudo poweroff` on the Pi.
Wait for the operating system to finish shutting down and SD-card activity to stop before turning off the power switch or unplugging the supply.
The physical switch does not perform an operating-system shutdown.

## Files

| File or directory | Purpose |
| --- | --- |
| `classify.py` | Student command for one or more images |
| `latency.py` | Student starter with three timing statements to add |
| `benchmark.py` | Student command for inference and image-to-result timing |
| `common.py` | Readable preprocessing, model loading, top five labels, and telemetry |
| `check_environment.py` | Runtime, asset, and inference checks |
| `camera_check.py` | CSI camera listing, headless JPEG capture, and decode check |
| `models/` | Two ONNX models, 1,000 labels, export details, and SHA-256 hashes |
| `samples/` | Three photographs, sources, and checksums |

Use `python classify.py --help` or `python benchmark.py --help` for options.
Explicit relative image paths are relative to the terminal's working directory;
bundled model paths and the default image are resolved relative to the scripts.
