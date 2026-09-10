"""CSI camera check: list cameras and capture one JPEG without a GUI."""

import sys, argparse, shutil, subprocess
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="Only list detected cameras.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index from the listing.")
    parser.add_argument("--output", type=Path, default=Path("results/camera_check.jpg"))
    args = parser.parse_args()
    if args.camera < 0:
        parser.error("--camera must be zero or greater")
    hello = shutil.which("rpicam-hello")
    still = shutil.which("rpicam-still")
    if not hello or (not args.list and not still):
        raise FileNotFoundError(
            "rpicam-apps is missing. Install it with: sudo apt install rpicam-apps."
        )

    # The OS camera tools are visible even while the Python venv is active.
    listed = subprocess.run(
        [hello, "--list-cameras"], capture_output=True, text=True, timeout=15,
    )
    listing = listed.stdout + listed.stderr
    print(listing, end="" if listing.endswith("\n") else "\n", flush=True)
    if listed.returncode != 0:
        raise ValueError("Camera listing failed. See the camera-tool output above.")
    if args.list:
        return
    if "No cameras available" in listing:
        raise ValueError("No camera detected. Shut down the Pi and disconnect power before checking the ribbon cable.")
    if args.output.suffix.lower() not in (".jpg", ".jpeg"):
        parser.error("--output must end with .jpg or .jpeg")
    if args.output.exists():
        raise FileExistsError(f"Output exists: {args.output}. Choose a new filename.")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [still, "--camera", str(args.camera), "--nopreview", "--timeout", "2000",
         "--width", "640", "--height", "480", "--encoding", "jpg",
         "--output", str(args.output)],
        check=True, timeout=30,
    )
    with Image.open(args.output) as photo:
        dimensions = photo.size
        if photo.format != "JPEG":
            raise ValueError("The camera output is not a JPEG image.")
        photo.load()
    print(f"PASS: captured a valid {dimensions[0]} x {dimensions[1]} JPEG: {args.output}")
    print("Inspect the photograph for exposure and focus when a display is available.")
    print(f'Optional classification: python classify.py --image "{args.output}"')


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
