"""Measure steady-state batch-one inference and image-to-top-five latency."""

import sys, argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from common import (
    MODEL_NAMES, ROOT, classify_image, device_state, environment_info,
    load_model, positive_int, preprocess, sha256, write_json,
)


def summarize(times_ms: list[float]) -> dict:
    values = np.asarray(times_ms, dtype=np.float64)
    return {
        "count": len(times_ms),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95, method="linear")),
        "mean_ms": float(np.mean(values)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, default=MODEL_NAMES[0])
    parser.add_argument("--image", type=Path, default=ROOT / "samples/chelsea.png")
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--warmup", type=positive_int, default=10)
    parser.add_argument("--runs", type=positive_int, default=100)
    parser.add_argument("--output", type=Path, required=True, help="New results directory.")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output exists: {args.output}. Choose a new directory.")
    if not args.image.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")

    # Load the model and prepare an input BEFORE measuring inference.
    session, labels, entry = load_model(args.model, args.threads)
    tensor = preprocess(args.image)
    feed = {session.get_inputs()[0].name: tensor}
    output_names = [session.get_outputs()[0].name]
    environment = environment_info()
    image_hash = sha256(args.image)
    started = datetime.now(timezone.utc).isoformat()
    before = device_state()

    print(f"{args.model}, {args.threads} thread(s), {args.warmup} warm-ups per phase, "
          f"{args.runs} timed runs per phase", flush=True)
    for _ in range(args.warmup):
        session.run(output_names, feed)

    inference_ms = []
    for _ in range(args.runs):
        # TIMING START: the input tensor is already prepared and in memory.
        start = perf_counter_ns()
        session.run(output_names, feed)
        end = perf_counter_ns()
        # TIMING END: preprocessing, softmax, printing, and saving are excluded.
        inference_ms.append((end - start) / 1000000)
    after_inference = device_state()

    for _ in range(args.warmup):
        classify_image(session, labels, args.image)

    end_to_end_ms = []
    for _ in range(args.runs):
        start = perf_counter_ns()
        predictions = classify_image(session, labels, args.image)
        end = perf_counter_ns()
        end_to_end_ms.append((end - start) / 1000000)
    after_end_to_end = device_state()

    report = {
        "schema_version": 1,
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "model_sha256": entry["sha256"],
        "model_bytes": entry["bytes"],
        "model_mib": entry["bytes"] / (1024 ** 2),
        "parameter_count": entry["parameter_count"],
        "image": args.image.name,
        "image_sha256": image_hash,
        "batch_size": 1,
        "precision": "float32",
        "threads": args.threads,
        "warmup_per_phase": args.warmup,
        "runs_per_phase": args.runs,
        "session": {
            "providers": session.get_providers(),
            "execution_mode": "ORT_SEQUENTIAL",
            "inter_op_threads": 1,
            "intra_op_threads": args.threads,
            "graph_optimization": "ORT_ENABLE_ALL",
            "allow_spinning": False,
        },
        "timing_scope": {
            "clock": "time.perf_counter_ns",
            "inference": "Synchronous session.run with preloaded float32 tensor; includes Python call overhead.",
            "end_to_end": "Open/decode image, EXIF/RGB, resize/crop/normalize, session.run, softmax and top5.",
            "excluded_from_both": "Model/session initialization, warm-ups, printing, result files, telemetry, hashing.",
            "cache_note": "Repeated same image: file bytes may be in OS cache; not a cold-storage benchmark.",
            "phase_order": ["inference", "end_to_end"],
            "p95_method": "NumPy percentile 95, linear interpolation",
        },
        "inference": summarize(inference_ms),
        "end_to_end": summarize(end_to_end_ms),
        "top5": predictions,
        "environment": environment,
        "device_state": {
            "before_warmup": before,
            "after_inference": after_inference,
            "after_end_to_end": after_end_to_end,
        },
    }
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "summary.json", report)
    with (args.output / "raw_timings.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["phase", "iteration", "latency_ms"])
        for phase, timings in (("inference", inference_ms), ("end_to_end", end_to_end_ms)):
            writer.writerows((phase, i, f"{value:.6f}") for i, value in enumerate(timings, 1))
    print(f"Model file: {report['model_mib']:.2f} MiB")
    for phase in ("inference", "end_to_end"):
        stats = report[phase]
        print(f"{phase:>12}: median {stats['median_ms']:.3f} ms, p95 {stats['p95_ms']:.3f} ms")
    print(f"Saved {args.output / 'summary.json'} and raw_timings.csv")
    if any(state["throttled_hex"] not in (None, "0x0") for state in report["device_state"].values()):
        print("Pi reports current or historical throttling/power flags. Record and investigate before comparing runs.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
