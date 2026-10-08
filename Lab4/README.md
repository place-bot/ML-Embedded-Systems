# Lab 4 Physical adversarial patches

Train a patch for YOLOv2, print it, and compare person detections with a Raspberry Pi camera.
Follow the Tutorial for the full procedure and the Assignment for report requirements.

## Start

Download the [repository](https://github.com/guoyb17/CSE60685-FA26-Lab-4) using **Code > Download ZIP**.
Open the [Colab training notebook](https://colab.research.google.com/drive/1uUrYiNTPYdX4QeIqvOUeqetCpCwGP525?usp=sharing), save a copy in Drive, and select a GPU runtime.
Run the cells in order and upload the downloaded code ZIP when prompted.

The notebook exports `lab4-patch.zip` with the selected `trained` or `reference` pair, print PDFs and training record.
The Pi uses OpenCV for inference; PyTorch is needed only for training.

## Files

- `patch_ops.py`: patch placement, transformations and loss.
- `train_patch.py`: optimization with a frozen detector.
- `make_prints.py`, `export_patch.py`, `import_patch.py`: printing and transfer.
- `live_demo.py`, `camera_stream.py`, `results_io.py`: camera display and saved observations.
- `NOTICE.md`, `licenses/`: sources and licenses.

## Troubleshooting

- **No GPU:** use `reference` in Colab. If Colab itself is unavailable, print the PDFs in `assets/reference/` and start the Pi session with `python live_demo.py --reference`.
- **Download fails:** after export finishes, use Colab's Files panel to download `lab4-patch.zip` from the extracted code folder under `lab4/`.
- **Runtime disconnects:** reconnect and rerun the cells; temporary runtime files may be lost.
- **Camera is busy:** stop other camera programs. Check the cable with power disconnected, then run `rpicam-hello --list-cameras`.
- **Page is unavailable:** check the Pi terminal and SSH connection. If port 8080 is busy, use `ssh -L 8081:127.0.0.1:8081 pi@raspberry.local`, run `python live_demo.py --port 8081`, and open `http://127.0.0.1:8081`.
- **No Clean detection:** improve lighting, include the torso and legs, and wait for several updates.
- **Little patch effect:** keep the paper flat and facing the camera; vary one factor in Explore and record the outcome.
- **Changing pairs:** import the new ZIP and restart the camera session. Import preserves the previous pair in a dated backup folder.

## Local training

With CUDA-enabled PyTorch, install `requirements-colab.txt`, then run `train_patch.py --device cuda`, `make_prints.py` and `export_patch.py`.
CPU training is available with `--device cpu` but is slower.
