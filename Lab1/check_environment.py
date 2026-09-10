"""Check CPU runtime, asset checksums, and one inference for each model."""

import sys, json

from common import (
    MODEL_NAMES, ROOT, check_file, classify_image, device_state, environment_info,
    load_model, read_json,
)


def main() -> None:
    info = environment_info()
    print(json.dumps({"environment": info, "device_state": device_state()}, indent=2))
    if info["python_bits"] != 64:
        raise ValueError("This lab requires a 64-bit OS and 64-bit Python.")
    for sample in read_json(ROOT / "samples/manifest.json")["images"]:
        check_file(ROOT / "samples" / sample["file"], sample["sha256"])
    for name in MODEL_NAMES:
        session, labels, _ = load_model(name, 1)
        top = classify_image(session, labels, ROOT / "samples/chelsea.png")[0]
        print(f"PASS {name}: 1000 output classes; top result = {top['label']}")
    if not info["board_model"] or "Raspberry Pi 4" not in info["board_model"]:
        print("Software check passed on this host. Course latency results must be collected on the assigned Pi.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
