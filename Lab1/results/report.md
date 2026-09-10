The final submission is `Lab1_Report.pdf`, compiled from `../report/Lab1_Report.tex`. The text below is retained as supporting report notes.

# CSE 60685 Lab 1 Report

On the supplied Raspberry Pi 4 Model B Rev 1.5 running 64-bit Raspberry Pi OS (Debian 13 trixie, release 13.5) and Python 3.13.5, I measured samples/chelsea.png with one CPU thread, 10 untimed warm-ups and 100 timed calls per phase, using the unchanged benchmark.py.

ONNX Runtime 1.23.2 used CPUExecutionProvider with FP32 batch-one inputs. Runs were sequential with the same CanaKit power supply, open case lid, running fan and ondemand CPU governor. The required runs started at 43.8-45.3 C; sampled temperatures stayed at or below 51.1 C, and all recorded power/throttling flags were 0x0.

| Model | ONNX size (MiB) | Median inference (ms) |
|---|---:|---:|
| MobileNetV3-Small | 9.71 | 39.351 |
| SqueezeNet 1.1 | 4.73 | 90.091 |

MobileNetV3-Small was 2.29 times faster, although its ONNX file is larger. The smaller SqueezeNet file did not yield lower latency. Parameter count and file size describe stored weights; latency also depends on spatial feature-map sizes, arithmetic, memory traffic and CPU kernel efficiency. Static inspection of the supplied graphs gives about 54.9 million convolution MACs for MobileNet versus 349.2 million for SqueezeNet (Conv nodes only). MobileNet includes 11 grouped/depthwise convolutions, whereas SqueezeNet uses dense convolutions and concatenations. Operator mix and optimized execution matter; these aggregate timings do not isolate each contribution.

In latency.py, I placed start = perf_counter_ns() immediately before session.run(output_names, feed), and end = perf_counter_ns() immediately after, returning (end - start) / 1,000,000 milliseconds. The interval includes the completed synchronous CPU model call and Python call overhead. Model/session loading and image decoding/preprocessing are outside it because the goal is repeated inference on a prepared tensor, not startup or full-pipeline cost. Softmax, printing and file output are also excluded.

With four threads, inference medians were 21.702 ms (MobileNet) and 42.988 ms (SqueezeNet), giving one-to-four-thread speedups of 1.81x and 2.10x, respectively.

check_environment.py passed for both models. Chelsea produced five predictions (MobileNet top result: tiger cat, 49.31%). The completed timer exercise ran 10 warm-ups and 100 calls. camera_check.py reported PASS for a readable 640 x 480 JPEG; the image was also classified. Each benchmark CSV contains 200 measurements (100 per phase). Code, JSON/CSV results and camera images are preserved.
