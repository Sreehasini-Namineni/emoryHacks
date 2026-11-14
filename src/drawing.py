"""Draw MediaPipe hand landmarks."""

from __future__ import annotations

import cv2
import numpy as np

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def draw_hand(frame: np.ndarray, landmarks, color=(0, 255, 0)) -> None:
    h, w = frame.shape[:2]
    points = []
    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append((x, y))
        cv2.circle(frame, (x, y), 4, color, -1)

    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b], color, 2)
