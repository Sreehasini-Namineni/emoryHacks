"""Step 3: detect hands with MediaPipe Hand Landmarker. Press q to quit."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from camera import open_webcam
from controls import Controls
from drawing import draw_hand
from gestures import PeaceClockwise, ThreeTurn

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


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
    tracker = PeaceClockwise()
    track_turn = ThreeTurn()
    controls = Controls()

    print("SANITY CHECK: peace sign (index+middle up) rotate clockwise -> terminal prints Volume increased.")

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            timestamp_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            n_hands = len(result.hand_landmarks)
            lm = result.hand_landmarks[0] if n_hands else None
            gesture = tracker.update(lm)
            if gesture:
                controls.handle(gesture)
            track = track_turn.update(lm)
            if track:
                controls.handle(track)

            if n_hands:
                for landmarks in result.hand_landmarks:
                    draw_hand(frame, landmarks)
                status = "PEACE" if tracker.holding else f"HAND DETECTED ({n_hands})"
                color = (0, 255, 0)
            else:
                status = "NO HAND"
                color = (0, 0, 255)

            cv2.putText(frame, status, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(
                frame,
                f"Volume {controls.volume}",
                (16, 76),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Step 3 — Hands", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("SANITY CHECK PASS if the label flipped when you showed / hid a hand.")


if __name__ == "__main__":
    main()
