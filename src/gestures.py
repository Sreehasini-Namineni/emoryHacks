"""Named gesture detectors. Map these names to actions in controls.py."""

from __future__ import annotations

import math
import time
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

def _len(lm, tip: int, mcp: int) -> float:
    return _d(lm[tip], lm[mcp])

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

def is_thumbsup(lm):
    palm = _d(lm[0], lm[9]) + 1e-6

    index = _len(lm, 8, 5) / palm
    middle = _len(lm, 12, 9) / palm
    ring = _len(lm, 16, 13) / palm
    pinky = _len(lm, 20, 17) / palm

    thumb = _len(lm, 4, 2) / palm

    thumb_up = lm[4].y < lm[2].y - (0.15 * palm)

    return (
        thumb > 0.7
        and thumb_up
        and index < 0.75
        and middle < 0.75
        and ring < 0.75
        and pinky < 0.75
    )

def is_thumbsdown(lm):
    palm = _d(lm[0], lm[9]) + 1e-6

    index = _len(lm, 8, 5) / palm
    middle = _len(lm, 12, 9) / palm
    ring = _len(lm, 16, 13) / palm
    pinky = _len(lm, 20, 17) / palm

    thumb = _len(lm, 4, 2) / palm

    thumb_down = lm[4].y > lm[2].y + (0.15 * palm)

    return (
        thumb > 0.7
        and thumb_down
        and index < 0.75
        and middle < 0.75
        and ring < 0.75
        and pinky < 0.75
    )

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


def _is_sideways(lm) -> bool:
    """Fingers pointing left/right: wrist -> middle knuckle is more horizontal than vertical."""
    return abs(lm[9].x - lm[0].x) > abs(lm[9].y - lm[0].y)


def _is_upright(lm) -> bool:
    """Fingers pointing up: middle knuckle above the wrist and more vertical than horizontal."""
    return (lm[0].y - lm[9].y) > abs(lm[9].x - lm[0].x)


class ThreeTurn:
    """Three fingers up, hand moved left or right and then back to where it started."""

    right = "three_turn_right"
    left = "three_turn_left"
    _min_dx = 0.06     # sideways travel, fraction of frame width
    _home_dx = 0.03    # this close to the start counts as back
    _window_s = 0.8    # time to make the move
    _return_s = 1.5    # time to come back after it

    def __init__(self) -> None:
        self._hist: list[tuple[float, float]] = []
        self._pending: tuple[str, float, float] | None = None  # (gesture, start x, move time)

    def update(self, lm) -> str | None:
        # Hand must point up (sideways three fingers is the AC gesture).
        if not lm or not _is_upright(lm) or not is_three(lm, relaxed=bool(self._hist) or self._pending is not None):
            self._hist.clear()
            self._pending = None
            return None

        now = time.time()
        x = lm[9].x

        if self._pending is not None:
            gesture, start, t = self._pending
            if abs(x - start) <= self._home_dx:
                self._pending = None
                self._hist.clear()
                return gesture
            if now - t > self._return_s:
                self._pending = None
                self._hist.clear()
            return None

        self._hist.append((now, x))
        self._hist = [h for h in self._hist if now - h[0] <= self._window_s]
        if len(self._hist) < 3:
            return None

        start = self._hist[0][1]
        dx = x - start
        if abs(dx) >= self._min_dx:
            # Frame is mirrored, so +x is the user's right.
            self._pending = (self.right if dx > 0 else self.left, start, now)
        return None

class ThreeSideways:
    """Three fingers pointing sideways, moved up or down and then back to where it started."""

    up = "three_sideways_up"
    down = "three_sideways_down"
    _min_dy = 0.04     # vertical travel, fraction of frame height
    _home_dy = 0.03    # this close to the start counts as back
    _window_s = 0.8    # time to make the move
    _return_s = 1.5    # time to come back after it
    _cooldown_s = 0.5  # minimum time between AC changes

    def __init__(self) -> None:
        self._hist: list[tuple[float, float]] = []
        self._pending: tuple[str, float, float] | None = None  # (gesture, start y, move time)
        self._last_fire = 0.0

    def update(self, lm) -> str | None:
        if not lm or not _is_sideways(lm) or not is_three(lm, relaxed=bool(self._hist) or self._pending is not None):
            self._hist.clear()
            self._pending = None
            return None

        now = time.time()
        y = lm[9].y

        if self._pending is not None:
            gesture, start, t = self._pending
            if abs(y - start) <= self._home_dy:
                self._pending = None
                self._hist.clear()
                if now - self._last_fire >= self._cooldown_s:
                    self._last_fire = now
                    return gesture
                return None
            if now - t > self._return_s:
                self._pending = None
                self._hist.clear()
            return None

        self._hist.append((now, y))
        self._hist = [h for h in self._hist if now - h[0] <= self._window_s]
        if len(self._hist) < 3:
            return None

        start = self._hist[0][1]
        dy = y - start
        if abs(dy) >= self._min_dy:
            # Image y points down, so negative dy is moving up.
            self._pending = (self.up if dy < 0 else self.down, start, now)
        return None
