"""Step 3: detect hands with MediaPipe Hand Landmarker. Press q to quit."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import math
from hud import Hud

sys.path.insert(0, str(Path(__file__).resolve().parent))
from camera import open_webcam
from controls import Controls
from drawing import draw_hand
from gestures import PeaceRotation, is_thumbsup, is_thumbsdown, ThreeRotation
from gestures import ThreeSideways, ThreeTurn, is_three, ThumbsCall, is_fist, PalmToFist
from confidence import ConfidencePanel, all_scores
from settings_ui import SettingsUI  # ADDED

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
    track_turn = ThreeTurn()
    ac_move = ThreeSideways()
    temp_rot = ThreeRotation()
    thumbs_call = ThumbsCall()
    play_pause = PalmToFist()
    last_temp = -1e9  # when temp last changed (skip / AC are blocked right after)
    controls = Controls()
    panel = ConfidencePanel()
    settings = SettingsUI(panel, controls)  # ADDED
    hud = Hud()
    cv2.namedWindow("Step 3 — Hands")  # ADDED
    cv2.setMouseCallback("Step 3 — Hands", settings.on_mouse)  # ADDED
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
            track = track_turn.update(lm)
            ac = ac_move.update(lm)
            temp = temp_rot.update(lm, frame.shape[1] / frame.shape[0])
            now = time.monotonic()
            if temp:
                # Hand is circling for temp: drop any skip / AC it looked like.
                last_temp = now
                track_turn.cancel()
                ac_move.cancel()
                track = ac = None
                controls.handle(temp)
            elif now - last_temp < 1.0:
                track = ac = None  # still circling (or just stopped): no skip / AC
            elif track or ac:
                temp_rot._reset()  # that was a skip / AC move, not a circle
                controls.handle(track or ac)
            call = thumbs_call.update(lm)
            if call:
                controls.handle(call)
            play = play_pause.update(lm)
            if play:
                controls.handle(play)
            # Scores are read after the detectors have updated this frame.
            panel.update(all_scores((tracker, track_turn, ac_move, temp_rot, thumbs_call, play_pause), lm),
                fired=[g for g in (gesture, track, ac, temp, call, play) if g])


            cv2.putText(frame, "Playing" if controls.playing else "Paused", (16, 276),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            progress = min(1.0, max(abs(tracker.offset), abs(temp_rot.offset)) / math.radians(30))
            hud.draw(frame, result.hand_landmarks, lm, controls, progress)
            panel.draw(frame)
            settings.draw(frame)  # ADDED
            cv2.imshow("Step 3 — Hands", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            cv2.setWindowProperty("Step 3 — Hands", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
