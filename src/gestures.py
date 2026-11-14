"""Named gesture detectors. Map these names to actions in controls.py."""

from __future__ import annotations

import math
from collections import deque

# Finger length / palm length ratios.
EXTENDED_MIN = 0.80
CURLED_MAX = 0.72
SEPARATION = 1.15
EXTENDED_MIN_HOLD = 0.65
CURLED_MAX_HOLD = 0.90

TRACK_POINT = 0              # landmark whose path we follow (0 = wrist)
WINDOW = 30                  # recent positions used to estimate the circle center
MIN_POINTS = 12              # positions needed before measuring
MIN_RADIUS_PALM = 0.30       # circle radius must exceed this fraction of palm length
STEP = math.radians(30)      # sweep around the circle needed per action
MISS_LIMIT = 8               # frames without the pose before reset


def _d(a, b) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _ratio(lm, tip: int, mcp: int, palm: float) -> float:
    return _d(lm[tip], lm[mcp]) / palm


def _wrap(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def is_peace(lm, holding: bool = False) -> bool:
    """Index+middle extended, ring+pinky curled, all as ratios of palm length."""
    palm = _d(lm[0], lm[9]) + 1e-6
    up = min(_ratio(lm, 8, 5, palm), _ratio(lm, 12, 9, palm))
    down = max(_ratio(lm, 16, 13, palm), _ratio(lm, 20, 17, palm))

    if holding:
        return up > EXTENDED_MIN_HOLD and down < CURLED_MAX_HOLD and up > down
    return up > EXTENDED_MIN and down < CURLED_MAX and up > down * SEPARATION


class PeaceRotation:
    """Peace sign held while the wrist travels in a circle.

    Emits peace_clockwise or peace_counterclockwise every STEP of sweep.
    """

    CW = "peace_clockwise"
    CCW = "peace_counterclockwise"

    def __init__(self) -> None:
        self.holding = False
        self.offset = 0.0                       # radians swept since last action (readout)
        self._pts: deque[tuple[float, float]] = deque(maxlen=WINDOW)
        self._miss = 0

    def _reset(self) -> None:
        self.holding = False
        self.offset = 0.0
        self._pts.clear()

    def update(self, lm, aspect: float = 16 / 9) -> str | None:
        pose = bool(lm) and is_peace(lm, holding=self.holding)
        if not pose:
            self._miss += 1
            if self._miss > MISS_LIMIT:
                self._reset()
            return None

        self._miss = 0
        self.holding = True

        # Scale x by the frame aspect ratio so a circle is a circle, not an ellipse.
        p = lm[TRACK_POINT]
        self._pts.append((p.x * aspect, p.y))
        if len(self._pts) < MIN_POINTS:
            return None

        n = len(self._pts)
        cx = sum(x for x, _ in self._pts) / n
        cy = sum(y for _, y in self._pts) / n
        radius = sum(math.hypot(x - cx, y - cy) for x, y in self._pts) / n

        palm = math.hypot((lm[9].x - lm[0].x) * aspect, lm[9].y - lm[0].y) + 1e-6
        if radius < MIN_RADIUS_PALM * palm:
            self.offset = 0.0                   # not moving in a circle; ignore jitter
            return None

        # Frame is flipped before detection, so this matches the screen.
        # In image coordinates (y down), increasing angle = clockwise on screen.
        (px, py), (x, y) = self._pts[-2], self._pts[-1]
        delta = _wrap(math.atan2(y - cy, x - cx) - math.atan2(py - cy, px - cx))
        self.offset += delta

        if self.offset >= STEP:
            self.offset -= STEP
            return self.CW
        if self.offset <= -STEP:
            self.offset += STEP
            return self.CCW
        return None