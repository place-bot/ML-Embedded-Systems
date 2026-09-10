"""Check retained evidence using the standard library; performs no inference."""
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "Lab1"
REFERENCE_BENCHMARK = "a5968bcb910edc48d4c0d56fff8e7ff74d984cb10e66f4abe1752863496175ba"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    manifest = read_json(ROOT / "results/evidence_sha256.json")
    for relative, expected in manifest.items():
        path = (ROOT / relative).resolve()
        require(path.is_relative_to(ROOT.resolve()), "Unsafe manifest path")
        require(digest(path) == expected, f"Evidence hash mismatch: {relative}")

    original = read_json(ROOT / "provenance/pi_export_sha256.json")
    revisions = read_json(ROOT / "provenance/document_revisions.json")
    require(set(revisions) == {"results/Lab1_Report.pdf", "results/report.md"},
            "Only the two report documents may differ from the original export")
    for relative, expected in original.items():
        actual = digest(ROOT / relative)
        if relative in revisions:
            require(revisions[relative]["original_sha256"] == expected and
                    revisions[relative]["final_sha256"] == actual,
                    f"Document revision mismatch: {relative}")
        else:
            require(actual == expected, f"Original Pi evidence changed: {relative}")

    require(digest(ROOT / "benchmark.py") == REFERENCE_BENCHMARK,
            "Reference benchmark.py was changed")
    image_hash = digest(ROOT / "samples/chelsea.png")
    baseline_environment = None
    intervals = []
    specs = [("mobilenet_t1", "mobilenet_v3_small", 1),
             ("squeezenet_t1", "squeezenet1_1", 1),
             ("mobilenet_t4", "mobilenet_v3_small", 4),
             ("squeezenet_t4", "squeezenet1_1", 4)]
    for name, model, threads in specs:
        folder = ROOT / "results" / name
        data = read_json(folder / "summary.json")
        require(data["model"] == model and data["threads"] == threads,
                f"Model/thread settings differ: {name}")
        require(data["warmup_per_phase"] == 10 and data["runs_per_phase"] == 100,
                f"Warm-up/run settings differ: {name}")
        require(data["image"] == "chelsea.png" and data["image_sha256"] == image_hash,
                f"Input image differs: {name}")
        require(data["model_sha256"] == digest(ROOT / "models" / (model + ".onnx")),
                f"Model bytes differ: {name}")
        require(data["batch_size"] == 1 and data["precision"] == "float32",
                f"Batch/precision differs: {name}")
        require(data["session"]["providers"] == ["CPUExecutionProvider"] and
                data["session"]["intra_op_threads"] == threads and
                data["session"]["inter_op_threads"] == 1,
                f"CPU session settings differ: {name}")
        env = data["environment"]
        require(env["machine"] == "aarch64" and
                env["board_model"].startswith("Raspberry Pi 4 Model B"),
                f"Missing Raspberry Pi 4 evidence: {name}")
        if baseline_environment is None:
            baseline_environment = env
        require(env == baseline_environment, f"Environment differs: {name}")
        require(all(s["throttled_hex"] == "0x0" for s in data["device_state"].values()),
                f"Recorded throttling flag: {name}")
        with (folder / "raw_timings.csv").open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            require(reader.fieldnames == ["phase", "iteration", "latency_ms"],
                    f"Unexpected CSV header: {name}")
            rows = list(reader)
        require(len(rows) == 200, f"Expected 200 timed observations: {name}")
        require({row["phase"] for row in rows} == {"inference", "end_to_end"},
                f"Unexpected phases: {name}")
        for phase in ("inference", "end_to_end"):
            selected = [r for r in rows if r["phase"] == phase]
            require([int(r["iteration"]) for r in selected] == list(range(1, 101)),
                    f"Missing/duplicate iterations: {name}/{phase}")
            values = [float(r["latency_ms"]) for r in selected]
            require(all(math.isfinite(v) and v > 0 for v in values),
                    f"Invalid latency: {name}/{phase}")
            require(data[phase]["count"] == 100 and
                    math.isclose(statistics.median(values), data[phase]["median_ms"],
                                 rel_tol=0, abs_tol=1e-6),
                    f"CSV/JSON median mismatch: {name}/{phase}")
        intervals.append((datetime.fromisoformat(data["started_utc"]),
                          datetime.fromisoformat(data["finished_utc"])))
        print(f"PASS {name}: 200 observations; inference median {data['inference']['median_ms']:.6f} ms")
    intervals.sort()
    require(all(start < end for start, end in intervals), "Invalid run interval")
    require(all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:])),
            "Benchmark run times overlap")
    require("PASS" in (ROOT / "results/camera_check.log").read_text(encoding="utf-8"),
            "Camera PASS record missing")
    print(f"PASS {len(manifest)} file hashes; unchanged benchmark; identical input/environment; sequential Pi runs")


if __name__ == "__main__":
    main()
