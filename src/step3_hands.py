"""Step 3: detect hands with MediaPipe Hand Landmarker. Press q to quit."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import math

sys.path.insert(0, str(Path(__file__).resolve().parent))
from camera import open_webcam
from controls import Controls
from drawing import draw_hand
from gestures import PeaceRotation, is_thumbsup, is_thumbsdown

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task"

# If the wrong hand gets tracked, swap this to "Left".
CONTROL_HAND = "Left" #takes rigth hand

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


def pick_hand(result, label: str):
    """Return landmarks for the hand with the given handedness label, or None."""
    for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        if handedness[0].category_name == label:
            return landmarks
    return None


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing {MODEL_PATH}. Run: python src/download_models.py"
        )

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = open_webcam()
    tracker = PeaceRotation()
    controls = Controls()
    last_ts = 0

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

            # Strictly increasing, monotonic timestamp (required by VIDEO mode).
            last_ts = max(last_ts + 1, int(time.monotonic() * 1000))
            result = landmarker.detect_for_video(mp_image, last_ts)

            n_hands = len(result.hand_landmarks)
            lm = pick_hand(result, CONTROL_HAND)
            gesture = tracker.update(lm, frame.shape[1] / frame.shape[0])
            if gesture:
                controls.handle(gesture)

            for landmarks in result.hand_landmarks:
                draw_hand(frame, landmarks)

            if lm is not None:
                status = ("PEACE" if tracker.holding 
                          else "THUMBS UP" if is_thumbsup(lm)
                          else "THUMBS DOWN" if is_thumbsdown(lm)
                          else f"HAND DETECTED ({n_hands})")

                color = (0, 255, 0)
            elif n_hands:
                status = f"SHOW {CONTROL_HAND.upper()} HAND"
                color = (0, 165, 255)
            else:
                status = "NO HAND"
                color = (0, 0, 255)

            cv2.putText(frame, status, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(frame, f"Volume {controls.volume}", (16, 76),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, controls.last_action, (16, 116),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
            cv2.putText(frame, f"Rot {math.degrees(tracker.offset):+.0f} deg", (16, 156),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
            cv2.imshow("Step 3 — Hands", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()