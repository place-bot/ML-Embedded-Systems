"""Classify one or more static images using a supplied ONNX model."""

import sys, argparse
from pathlib import Path

from common import (
    MODEL_NAMES, ROOT, classify_image, load_model, positive_int, sha256, write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, default=MODEL_NAMES[0])
    parser.add_argument("--image", type=Path, nargs="+", default=[ROOT / "samples/chelsea.png"])
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--output", type=Path, help="Optional new JSON filename.")
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError(f"Output exists: {args.output}. Choose a new filename.")
    for path in args.image:
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
    session, labels, entry = load_model(args.model, args.threads)
    records = []
    print(f"Model: {args.model} | CPU threads: {args.threads}")
    print("Scores are softmax outputs, not measured accuracy or calibrated confidence.")
    for path in args.image:
        predictions = classify_image(session, labels, path)
        print(f"\nImage: {path.name}")
        for rank, item in enumerate(predictions, 1):
            print(f"  {rank}. {item['label']:<24} {100 * item['score']:6.2f}%")
        records.append({"image": path.name, "sha256": sha256(path), "top5": predictions})
    if args.output:
        write_json(args.output, {
            "model": args.model, "model_sha256": entry["sha256"],
            "threads": args.threads, "images": records,
        })
        print(f"\nSaved {args.output}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
