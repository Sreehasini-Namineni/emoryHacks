"""Named gesture detectors. Map these names to actions in controls.py later."""

from __future__ import annotations

import math
import time


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


def is_three(lm, relaxed: bool = False) -> bool:
    """Index+middle+ring clearly longer than pinky. Thumb is ignored (too noisy)."""
    palm = _d(lm[0], lm[9]) + 1e-6
    index = _len(lm, 8, 5) / palm
    middle = _len(lm, 12, 9) / palm
    ring = _len(lm, 16, 13) / palm
    pinky = _len(lm, 20, 17) / palm
    if relaxed:
        return index > 0.65 and middle > 0.65 and ring > 0.65 and ring > pinky and pinky < 0.9
    return index > 0.8 and middle > 0.8 and ring > 0.8 and pinky < 0.72 and ring > pinky * 1.15


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


def _palm_yaw(lm) -> float:
    """Left/right facing of the palm (turning around the vertical axis), from the palm normal."""
    ax, ay, az = lm[5].x - lm[0].x, lm[5].y - lm[0].y, lm[5].z - lm[0].z
    bx, by, bz = lm[17].x - lm[0].x, lm[17].y - lm[0].y, lm[17].z - lm[0].z
    nx = ay * bz - az * by
    nz = ax * by - ay * bx
    return math.atan2(nx, nz)


class ThreeTurn:
    """Three fingers up with the palm turned 15 deg left or right."""

    right = "three_turn_right"
    left = "three_turn_left"
    _step = math.radians(15)
    _window_s = 0.5
    _cooldown_s = 1.0

    def __init__(self) -> None:
        self._hist: list[tuple[float, float]] = []
        self._last_fire = 0.0

    def update(self, lm) -> str | None:
        if not lm or not is_three(lm, relaxed=bool(self._hist)):
            self._hist.clear()
            return None

        now = time.time()
        self._hist.append((now, _palm_yaw(lm)))
        self._hist = [h for h in self._hist if now - h[0] <= self._window_s]
        if now - self._last_fire < self._cooldown_s or len(self._hist) < 3:
            return None

        turn = (self._hist[-1][1] - self._hist[0][1] + math.pi) % (2 * math.pi) - math.pi
        if abs(turn) >= self._step:
            self._last_fire = now
            self._hist.clear()
            return self.right if turn > 0 else self.left
        return None
