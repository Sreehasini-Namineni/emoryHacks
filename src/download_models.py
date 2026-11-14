"""Download the official MediaPipe Hand Landmarker model into ./models."""

from pathlib import Path
from urllib.request import urlretrieve

HAND_MODEL = "hand_landmarker.task"
HAND_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def main() -> None:
    models_dir = Path(__file__).resolve().parents[1] / "models"
    models_dir.mkdir(exist_ok=True)
    dest = models_dir / HAND_MODEL

    if dest.exists() and dest.stat().st_size > 1000:
        print(f"Already present: {dest} ({dest.stat().st_size} bytes)")
    else:
        print(f"Downloading {HAND_MODEL} ...")
        urlretrieve(HAND_URL, dest)
        print(f"Saved {dest} ({dest.stat().st_size} bytes)")

    if dest.stat().st_size < 1000:
        raise RuntimeError(f"SANITY CHECK FAIL: {HAND_MODEL} missing or too small.")

    print("SANITY CHECK PASS: hand_landmarker.task exists and is non-empty.")


if __name__ == "__main__":
    main()
