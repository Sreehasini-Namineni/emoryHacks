"""Open the Mac webcam through AVFoundation, retrying while macOS shows the permission prompt."""

from __future__ import annotations

import sys
import time

import cv2


def open_webcam() -> cv2.VideoCapture:
    if sys.platform != "darwin":
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            return cap
        raise RuntimeError("SANITY CHECK FAIL: webcam did not open.")

    # Default VideoCapture(0) can fall through to FFMPEG, which cannot use the Mac camera.
    backend = cv2.CAP_AVFOUNDATION

    # Status 0 = NotDetermined. OpenCV requests access, then fails this constructor
    # before you click Allow. Retry so the same process picks up the grant.
    for attempt in range(12):
        for index in (0, 1):
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                continue
            ok, frame = cap.read()
            if ok and frame is not None:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"Opened camera index={index} via AVFoundation at {w}x{h}")
                return cap
            cap.release()
        if attempt == 0:
            print("Waiting for macOS camera permission (click Allow if a prompt is open)...")
        time.sleep(0.5)

    raise RuntimeError(
        "SANITY CHECK FAIL: webcam did not open.\n"
        "macOS granted the prompt to the app that launched Python, not to Python itself.\n"
        "1. System Settings → Privacy & Security → Camera → enable Cursor (or Terminal).\n"
        "2. Run this script again from the same app. First click-Allow often needs a rerun."
    )
