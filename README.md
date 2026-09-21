# CSE 60685 - Machine Learning for Embedded Systems

## Lab 2: LeNet architecture comparison on the Raspberry Pi

The three required designs have completed training, full test-set evaluation and inference measurements on the supplied Pi 4B. [Lab 2 evidence and reproduction notes](Lab2/EVIDENCE.md) link the retained source, checkpoints, original JSON/CSV files and checks. **Canvas submission: [Lab2_Report.pdf](Lab2/results/Lab2_Report.pdf)**, containing one page of analysis plus the required AI Attribution Appendix. [LaTeX source](Lab2/report/Lab2_Report.tex) is retained. The report incorporates the student's preference for C's parameter/accuracy tradeoff and explains the architecture-dependent runtime results.

| Model | Parameters | Test accuracy | Mean training (s/epoch) | Median inference (ms) |
| --- | ---: | ---: | ---: | ---: |
| A: baseline | 44,426 | 79.72% | 13.8596 | 2.6602735 |
| B: conv-small | 27,180 | 78.05% | 10.1653 | 2.3970385 |
| C: FC-small (60/42) | 20,984 | 78.84% | 13.5567 | 2.4440160 |

## Lab 1: image classification and inference latency

This repository retains the completed timing exercise and actual measurements from the supplied Raspberry Pi 4B. All inference and camera runs were performed on the Pi. The OMEN computer was used for SSH, backups, report typesetting, and offline verification.

**Canvas submission: upload only [Lab1_Report.pdf](Lab1/results/Lab1_Report.pdf), which is one page.** The assignment asks that the code and JSON/CSV be kept for inspection; they are retained here. The deadline stated in the assignment is before class on Thursday, September 17, 2026.

The final report is written in [LaTeX](Lab1/report/Lab1_Report.tex), using a plain black-and-white Times New Roman layout. It covers the device/settings, required results table, model comparison, timing boundaries, and optional four-thread results.

### Required results

Both models used `samples/chelsea.png`, one CPU thread, 10 untimed warm-ups and 100 timed calls **per phase**, with the original `benchmark.py`. Runs were sequential, with the same Pi, power supply, software environment and cooling (open lid, running fan). The table reports inference-only medians.

| Model | ONNX size (MiB) | Median inference (ms) | Raw evidence |
| --- | ---: | ---: | --- |
| MobileNetV3-Small | 9.71 | 39.351 | [JSON](Lab1/results/mobilenet_t1/summary.json), [CSV](Lab1/results/mobilenet_t1/raw_timings.csv) |
| SqueezeNet 1.1 | 4.73 | 90.091 | [JSON](Lab1/results/squeezenet_t1/summary.json), [CSV](Lab1/results/squeezenet_t1/raw_timings.csv) |

Each CSV contains 200 observations plus a header: 100 `inference` and 100 `end_to_end` calls. No timed observations were discarded. The exported JSON contains the full-precision values, SHA-256 hashes, versions, thread settings and sampled device telemetry. Optional four-thread medians were 21.702 ms and 42.988 ms, respectively.

### Assignment and tutorial checklist

| Requirement | Retained evidence |
| --- | --- |
| Set up the Pi and project virtual environment; run environment check | [PASS log](Lab1/results/check_environment_final.log), [packages](Lab1/results/python_packages.txt), device/version metadata in each summary |
| Classify Chelsea with MobileNet and print five predictions | [Predictions](Lab1/results/mobilenet_sample_predictions.json), [top five in summary](Lab1/results/mobilenet_t1/summary.json) |
| Complete the timer immediately around `session.run`, returning milliseconds | [latency.py](Lab1/latency.py), [patch against starter](Lab1/results/latency_exercise.patch), [exercise output](Lab1/results/latency_exercise.txt) |
| Use unchanged reference benchmark for both required runs | [benchmark.py](Lab1/benchmark.py), [source hashes](Lab1/results/source_sha256.txt), [upstream commit](Lab1/results/repository_commit.txt) |
| Same image, Pi, one thread; 10 warm-ups and 100 calls per phase; sequential runs | Required JSON/CSV above, [run context](Lab1/results/run_context.json) |
| List camera and capture a readable JPEG with PASS | [camera listing](Lab1/results/camera_listing.txt), [PASS log](Lab1/results/camera_check.log), [captured image](Lab1/results/camera_check.jpg) |
| One-page PDF with settings, table and both discussion answers | [Final PDF](Lab1/results/Lab1_Report.pdf), [LaTeX source](Lab1/report/Lab1_Report.tex) |
| Optional Step 6: four threads and other sample images | [MobileNet T4](Lab1/results/mobilenet_t4/summary.json), [SqueezeNet T4](Lab1/results/squeezenet_t4/summary.json), both `*_sample_predictions.json` files |
| Step 8: retain work and perform operating-system shutdown | [Final files verified on Pi](Lab1/provenance/final_pi_verification.log); [`sudo poweroff` succeeded](Lab1/provenance/shutdown.log), exit code 0, followed by SSH disconnect |

### Source and provenance

The lab scripts, model files and sample images originate from [guoyb17/CSE60685-FA26-Lab-1](https://github.com/guoyb17/CSE60685-FA26-Lab-1), commit `5bb37f5c8718881a9728e3cc7567a560b4119920`. Its [original README](Lab1/README.md) and asset manifests are retained. Among the supplied Python scripts, only `latency.py` was changed. The SHA-256 of the unchanged reference `benchmark.py` is:

```text
a5968bcb910edc48d4c0d56fff8e7ff74d984cb10e66f4abe1752863496175ba
```

All original measurement files retain their exact exported bytes. The final PDF was typeset in LaTeX on OMEN after export. [Provenance notes](Lab1/provenance/README.md) explain that document-only revision and retain the original Pi export manifest. The final report and LaTeX source were copied back to the Pi and checked before `sudo poweroff` succeeded on September 10, 2026, at 02:05 UTC (September 9 at 22:05 Eastern Daylight Time). [verify_results.py](verify_results.py) checks source/evidence hashes, the model/image hashes, row counts, medians, thread settings and sequential run times without performing any inference:

```bash
python verify_results.py
```

The report's optional Conv-only MAC estimates can be reproduced with [inspect_model_graphs.py](Lab1/analysis/inspect_model_graphs.py), using the additional `onnx` package in an analysis environment. This reads graph structure and performs no inference.

### Repeat the experiment on the Raspberry Pi

Use a new output directory so the recorded runs stay intact. These commands run inside a Pi terminal, after copying/cloning this repository to the Pi:

```bash
cd ML-Embedded-Systems/Lab1
python3 -m venv env
source env/bin/activate
python -m pip install -r requirements.txt
python check_environment.py
python classify.py --model mobilenet_v3_small --image samples/chelsea.png
python latency.py --model mobilenet_v3_small --image samples/chelsea.png
python benchmark.py --model mobilenet_v3_small --image samples/chelsea.png --threads 1 --warmup 10 --runs 100 --output results/mobilenet_t1_repeat
# Let the board cool to a comparable starting temperature, then run the next model.
python classify.py --model squeezenet1_1 --image samples/chelsea.png
python benchmark.py --model squeezenet1_1 --image samples/chelsea.png --threads 1 --warmup 10 --runs 100 --output results/squeezenet_t1_repeat
rpicam-hello --list-cameras
python camera_check.py --output results/camera_check_repeat.jpg
sudo poweroff
```

Wait for operating-system shutdown and SD-card activity to stop before disconnecting power. For a repeated camera capture, use a fresh filename again.

To rebuild the report, use XeLaTeX with Times New Roman and Consolas installed:

```bash
cd Lab1/report
xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error -output-directory=../results Lab1_Report.tex
```

Compilation changes the PDF file hash; regenerate the document entry in the manifest if rebuilding. Measurement files must remain unchanged.
