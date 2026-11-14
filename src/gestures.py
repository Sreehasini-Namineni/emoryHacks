"""Named gesture detectors. Map these names to actions in controls.py later."""

from __future__ import annotations

import math


def _d(a, b) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _len(lm, tip: int, mcp: int) -> float:
    return _d(lm[tip], lm[mcp])


def is_peace(lm, relaxed: bool = False) -> bool:
    """Index+middle clearly longer than ring+pinky. Thumb is ignored (too noisy)."""
    palm = _d(lm[0], lm[9]) + 1e-6
    index = _len(lm, 8, 5) / palm
    middle = _len(lm, 12, 9) / palm
    ring = _len(lm, 16, 13) / palm
    pinky = _len(lm, 20, 17) / palm
    if relaxed:
        return index > 0.65 and middle > 0.65 and index > ring and middle > ring and ring < 0.9 and pinky < 0.9
    return (
        index > 0.8
        and middle > 0.8
        and ring < 0.72
        and pinky < 0.72
        and index > ring * 1.15
        and middle > ring * 1.15
    )


class PeaceClockwise:
    name = "peace_clockwise"
    _step = math.radians(30)

    def __init__(self) -> None:
        self.holding = False
        self._prev: float | None = None
        self._acc = 0.0
        self._miss = 0

    def update(self, lm) -> str | None:
        pose = bool(lm) and is_peace(lm, relaxed=self.holding)
        if not pose:
            self._miss += 1
            if self._miss > 8:
                self.holding = False
                self._prev = None
                self._acc = 0.0
            return None

        # Palm heading stays stable while the fingertips jitter.
        angle = math.atan2(lm[9].y - lm[0].y, lm[9].x - lm[0].x)

        if not self.holding or self._prev is None:
            self.holding = True
            self._miss = 0
            self._prev = angle
            return None

        self._miss = 0
        delta = angle - self._prev
        if delta > math.pi:
            delta -= 2 * math.pi
        elif delta < -math.pi:
            delta += 2 * math.pi
        self._prev = angle

        # Webcam is mirrored; clockwise on-screen and in-world disagree on sign.
        # Count rotation magnitude so a clockwise turn always registers.
        if abs(delta) < 0.03:
            return None
        self._acc += abs(delta)
        if self._acc >= self._step:
            self._acc -= self._step
            return self.name
        return None
