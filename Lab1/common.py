"""Shared image processing, model loading, and reporting for Lab 1.
"""

from __future__ import annotations

import os, json, argparse, hashlib, platform, shutil, struct, subprocess
import importlib.metadata
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
MODEL_NAMES = ("mobilenet_v3_small", "squeezenet1_1")
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, data) -> None:
    """Keep earlier measurements: choose a new output name on every run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def check_file(path: Path, expected_hash: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path.name}. Obtain the complete lab folder from the TA."
        )
    if sha256(path) != expected_hash:
        raise ValueError(f"Checksum mismatch for {path.name}. Copy the original file again.")


def preprocess(image_path: Path) -> np.ndarray:
    """Match the ImageNet1K V1 PIL transforms of both Torchvision models.

    RGB -> resize shorter edge to 256 -> center crop 224 -> [0,1] ->
    channel normalization -> contiguous float32 NCHW, batch size 1.
    """
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    width, height = image.size
    if width <= height:
        new_size = (256, int(256 * height / width))
    else:
        new_size = (int(256 * width / height), 256)
    image = image.resize(new_size, resample=Image.Resampling.BILINEAR)
    # Python round matches torchvision.transforms.functional.center_crop.
    left = int(round((image.width - 224) / 2.0))
    top = int(round((image.height - 224) / 2.0))
    image = image.crop((left, top, left + 224, top + 224))
    pixels = np.asarray(image, dtype=np.float32) / np.float32(255.0)
    normalized = (pixels - MEAN) / STD
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None, ...])


def top_predictions(logits: np.ndarray, labels: list[str], count: int = 5) -> list[dict]:
    """Convert logits to softmax scores, then attach the ImageNet class names."""
    scores = np.asarray(logits, dtype=np.float32)
    if scores.shape != (1, len(labels)) or not np.isfinite(scores).all():
        raise ValueError(f"Expected finite logits with shape (1, {len(labels)}).")
    scores = scores[0]
    exp_scores = np.exp(scores - scores.max())  # Stable softmax.
    probabilities = exp_scores / exp_scores.sum()
    indices = np.argsort(-probabilities, kind="stable")[:count]
    return [
        {"class_id": int(i), "label": labels[i], "score": float(probabilities[i])}
        for i in indices
    ]


def session_options(threads: int) -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Use the same waiting policy in every run. Workers sleep instead of spinning.
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return options


def load_model(name: str, threads: int):
    if name not in MODEL_NAMES or threads < 1:
        raise ValueError("Choose a supplied model and a positive thread count.")
    manifest_path = ROOT / "models" / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Model manifest is missing. Obtain the complete lab folder.")
    manifest = read_json(manifest_path)
    entry = manifest["models"][name]
    model_path = ROOT / "models" / entry["file"]
    label_path = ROOT / "models" / manifest["labels"]["file"]
    check_file(model_path, entry["sha256"])
    check_file(label_path, manifest["labels"]["sha256"])
    labels = read_json(label_path)
    if len(labels) != 1000 or not all(isinstance(label, str) for label in labels):
        raise ValueError("The label file must contain 1000 ImageNet class names.")
    session = ort.InferenceSession(
        str(model_path), sess_options=session_options(threads),
        providers=["CPUExecutionProvider"],
    )
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if (len(inputs) != 1 or inputs[0].shape != [1, 3, 224, 224]
            or inputs[0].type != "tensor(float)"
            or len(outputs) != 1 or outputs[0].shape != [1, 1000]):
        raise ValueError("Model input/output does not match this lab's fixed FP32 contract.")
    return session, labels, entry


def classify_image(session, labels: list[str], image_path: Path) -> list[dict]:
    tensor = preprocess(image_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    logits = session.run([output_name], {input_name: tensor})[0]
    return top_predictions(logits, labels)


def read_text_or_none(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip().strip("\x00")
    except OSError:
        return None


def device_state() -> dict:
    """Pi telemetry is optional, so the same program also runs on a laptop."""
    temperature = read_text_or_none("/sys/class/thermal/thermal_zone0/temp")
    frequency = read_text_or_none("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
    result = {
        "temperature_c": float(temperature) / 1000 if temperature else None,
        "cpu0_frequency_khz": int(frequency) if frequency else None,
        "throttled_hex": None,
    }
    command = shutil.which("vcgencmd")
    if command:
        try:
            completed = subprocess.run(
                [command, "get_throttled"], capture_output=True, text=True,
                timeout=3, check=False,
            )
            if completed.returncode == 0 and "=" in completed.stdout:
                result["throttled_hex"] = completed.stdout.strip().split("=", 1)[1]
        except (OSError, subprocess.TimeoutExpired):
            pass
    return result


def environment_info() -> dict:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "python_bits": struct.calcsize("P") * 8,
        "logical_cpus": os.cpu_count(),
        "board_model": read_text_or_none("/proc/device-tree/model"),
        "os_release": read_text_or_none("/etc/os-release"),
        "cpu0_governor": read_text_or_none(
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
        ),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "Pillow", "onnxruntime")
        },
        "available_providers": ort.get_available_providers(),
    }
