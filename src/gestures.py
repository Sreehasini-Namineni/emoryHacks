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


def _is_sideways(lm) -> bool:
    """Fingers pointing left/right: wrist -> middle knuckle is more horizontal than vertical."""
    return abs(lm[9].x - lm[0].x) > abs(lm[9].y - lm[0].y)


def _is_upright(lm) -> bool:
    """Fingers pointing up: middle knuckle above the wrist and more vertical than horizontal."""
    return (lm[0].y - lm[9].y) > abs(lm[9].x - lm[0].x)


class ThreeTurn:
    """Three fingers up, palm turned 15 deg left or right and then back to where it started."""

    right = "three_turn_right"
    left = "three_turn_left"
    _step = math.radians(15)
    _home = math.radians(5)   # this close to the start counts as back
    _window_s = 0.5           # time to make the turn
    _return_s = 1.0           # time to turn back after it

    def __init__(self) -> None:
        self._hist: list[tuple[float, float]] = []
        self._pending: tuple[str, float, float] | None = None  # (gesture, start yaw, turn time)

    def update(self, lm) -> str | None:
        # Hand must point up (sideways three fingers is the AC gesture).
        if not lm or not _is_upright(lm) or not is_three(lm, relaxed=bool(self._hist) or self._pending is not None):
            self._hist.clear()
            self._pending = None
            return None

        now = time.time()
        yaw = _palm_yaw(lm)

        if self._pending is not None:
            gesture, start, t = self._pending
            if abs(_wrap(yaw - start)) <= self._home:
                self._pending = None
                self._hist.clear()
                return gesture
            if now - t > self._return_s:
                self._pending = None
                self._hist.clear()
            return None

        self._hist.append((now, yaw))
        self._hist = [h for h in self._hist if now - h[0] <= self._window_s]
        if len(self._hist) < 3:
            return None

        start = self._hist[0][1]
        turn = _wrap(yaw - start)
        if abs(turn) >= self._step:
            self._pending = (self.right if turn > 0 else self.left, start, now)
        return None


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class ThreeSideways:
    """Three fingers pointing sideways, moved up or down. Hand must return to where it started before the next step."""

    up = "three_sideways_up"
    down = "three_sideways_down"
    _min_dy = 0.10   # vertical travel, fraction of frame height
    _home_dy = 0.04  # this close to the start counts as back in the middle
    _window_s = 0.5

    def __init__(self) -> None:
        self._hist: list[tuple[float, float]] = []
        self._home: float | None = None  # set after a step; cleared once the hand is back

    def update(self, lm) -> str | None:
        if not lm or not _is_sideways(lm) or not is_three(lm, relaxed=bool(self._hist) or self._home is not None):
            self._hist.clear()
            self._home = None
            return None

        now = time.time()
        y = lm[9].y

        if self._home is not None:
            if abs(y - self._home) <= self._home_dy:
                self._home = None
                self._hist.clear()
            return None

        self._hist.append((now, y))
        self._hist = [h for h in self._hist if now - h[0] <= self._window_s]
        if len(self._hist) < 3:
            return None

        start = self._hist[0][1]
        dy = y - start
        if abs(dy) >= self._min_dy:
            self._home = start
            self._hist.clear()
            # Image y points down, so negative dy is moving up.
            return self.up if dy < 0 else self.down
        return None
