"""Exercise: add three timing statements around one inference call.

Complete time_inference using Tutorial Step 4. Model loading, preprocessing,
warm-ups, repetition, and reporting are supplied. Use benchmark.py for the
standardized measurements in the report.
"""

import sys, argparse
from pathlib import Path
from statistics import median
from time import perf_counter_ns

from common import MODEL_NAMES, ROOT, load_model, preprocess


def time_inference(session, output_names, feed):
    start = perf_counter_ns()
    session.run(output_names, feed)
    end = perf_counter_ns()
    return (end - start) / 1_000_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, default=MODEL_NAMES[0])
    parser.add_argument("--image", type=Path, default=ROOT / "samples/chelsea.png")
    args = parser.parse_args()
    session, _, _ = load_model(args.model, threads=1)
    tensor = preprocess(args.image)
    feed = {session.get_inputs()[0].name: tensor}
    output_names = [session.get_outputs()[0].name]

    for _ in range(10):
        session.run(output_names, feed)

    measurements = []
    for _ in range(100):
        elapsed_ms = time_inference(session, output_names, feed)
        if elapsed_ms is None:
            raise ValueError(
                "Complete time_inference at TIMER START / TIMER END using Tutorial Step 4. "
                "The supplied starter intentionally has no timer yet."
            )
        measurements.append(elapsed_ms)

    print(f"Model: {args.model} | CPU threads: 1 | warm-ups: 10 | timed calls: 100")
    print(f"Inference median: {median(measurements):.3f} ms")
    print("Use benchmark.py to collect the two standardized report measurements.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
