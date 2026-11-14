"""Step 1: confirm Python, OpenCV, and MediaPipe import correctly."""

import sys


def main() -> None:
    print("Python:", sys.version.split()[0])

    import cv2
    import mediapipe as mp
    import numpy as np

    print("OpenCV:", cv2.__version__)
    print("MediaPipe:", mp.__version__)
    print("NumPy:", np.__version__)
    print("SANITY CHECK PASS: all libraries imported.")


if __name__ == "__main__":
    main()
