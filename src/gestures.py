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
STEP = math.radians(360)      # sweep around the circle needed per action
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


# ---------------------------------------------------------------------------
# Confidence scoring (added). Read-only: nothing here changes detector state or
# what update() returns. Score 1.0 = at the trigger point, 0.0 = nowhere close.
# These are heuristic match scores, not probabilities.
# ---------------------------------------------------------------------------

MOTION_FLOOR = 0.25          # a held pose with no motion yet shows this fraction of its pose score

# Same numbers as the literals inside is_three().
THREE_EXT, THREE_CURL, THREE_SEP = 0.8, 0.72, 1.15
THREE_EXT_HOLD, THREE_CURL_HOLD = 0.65, 0.9


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def _at_least(value: float, thr: float) -> float:
    """1.0 once value >= thr, falling to 0 as value -> 0."""
    return _clamp(value / thr)


def _at_most(value: float, thr: float) -> float:
    """1.0 while value <= thr, falling to 0 at 2 * thr."""
    return _clamp(2 - value / thr)


def _finger_ratios(lm):
    palm = _d(lm[0], lm[9]) + 1e-6
    return (
        _ratio(lm, 8, 5, palm),
        _ratio(lm, 12, 9, palm),
        _ratio(lm, 16, 13, palm),
        _ratio(lm, 20, 17, palm),
    )


def peace_pose_score(lm, holding: bool) -> float:
    """Weakest condition wins, so a low score points at what is holding it back."""
    index, middle, ring, pinky = _finger_ratios(lm)
    up, down = min(index, middle), max(ring, pinky)
    if holding:
        return min(
            _at_least(up, EXTENDED_MIN_HOLD),
            _at_most(down, CURLED_MAX_HOLD),
            _at_least(up / (down + 1e-6), 1.0),
        )
    return min(
        _at_least(up, EXTENDED_MIN),
        _at_most(down, CURLED_MAX),
        _at_least(up / (down * SEPARATION + 1e-6), 1.0),
    )


def three_pose_score(lm, relaxed: bool) -> float:
   index, middle, ring, _ = _finger_ratios(lm)
   ext = THREE_EXT_HOLD if relaxed else THREE_EXT
   bent = 0.88 if relaxed else 0.82            # same pinky limits as is_three()
   pinky_straight = _straightness(lm, (17, 18, 19, 20))
   return min(
       _at_least(min(index, middle, ring), ext),
       _clamp((1.0 - pinky_straight) / (1.0 - bent)),   # 1.0 once bent enough, 0 when straight
   )



def upright_score(lm) -> float:
    return _clamp((lm[0].y - lm[9].y) / (abs(lm[9].x - lm[0].x) + 1e-6))


def sideways_score(lm) -> float:
    return _clamp(abs(lm[9].x - lm[0].x) / (abs(lm[9].y - lm[0].y) + 1e-6))


def _combine(pose: float, progress: float) -> float:
    return pose * (MOTION_FLOOR + (1 - MOTION_FLOOR) * progress)


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
    
    def _is_pose(self, lm) -> bool:
        return is_peace(lm, holding=self.holding)
    
    def update(self, lm, aspect: float = 16 / 9) -> str | None:
        pose = bool(lm) and self._is_pose(lm)
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

    def score(self, lm) -> dict[str, float]:
        """Call after update(). Pose match x sweep progress in each direction."""
        if not lm:
            return {self.CW: 0.0, self.CCW: 0.0}
        pose = peace_pose_score(lm, self.holding)
        return {
            self.CW: _combine(pose, _clamp(self.offset / STEP)),
            self.CCW: _combine(pose, _clamp(-self.offset / STEP)),
        }

def is_three(lm, relaxed: bool = False) -> bool:
   """Index+middle+ring extended, pinky bent. Thumb is ignored (too noisy).


   The pinky is judged by how bent it is, not its length: a straight pinky is naturally
   short, so a length check made an open palm look like three fingers.
   """
   palm = _d(lm[0], lm[9]) + 1e-6
   index = _len(lm, 8, 5) / palm
   middle = _len(lm, 12, 9) / palm
   ring = _len(lm, 16, 13) / palm
   pinky_straight = _straightness(lm, (17, 18, 19, 20))   # ~1.0 straight, lower when bent
   if relaxed:
       return index > 0.65 and middle > 0.65 and ring > 0.65 and pinky_straight < 0.88
   return index > 0.8 and middle > 0.8 and ring > 0.8 and pinky_straight < 0.82




def _is_sideways(lm) -> bool:
    """Fingers pointing left/right: wrist -> middle knuckle is more horizontal than vertical."""
    return abs(lm[9].x - lm[0].x) > abs(lm[9].y - lm[0].y)


def _is_upright(lm) -> bool:
    """Fingers pointing up: middle knuckle above the wrist and more vertical than horizontal."""
    return (lm[0].y - lm[9].y) > abs(lm[9].x - lm[0].x)


class ThreeRotation(PeaceRotation):
   """Same circle tracking as PeaceRotation, but with three fingers up."""


   CW = "three_clockwise"
   CCW = "three_counterclockwise"
   _min_round = 0.25  # path must be at least this wide as it is long (a skip is a flat line)


   def _is_pose(self, lm) -> bool:
       return is_three(lm, relaxed=self.holding)


   def update(self, lm, aspect: float = 16 / 9) -> str | None:
       result = super().update(lm, aspect)

       # Only count real circles, not a left/right or up/down line.
       if self._pts:
           xs = [x for x, _ in self._pts]
           ys = [y for _, y in self._pts]
           w, h = max(xs) - min(xs), max(ys) - min(ys)
           if min(w, h) < self._min_round * max(w, h, 1e-6):
               self.offset = 0.0
               return None
       return result


   def score(self, lm) -> dict[str, float]:
       """Call after update(). Same as PeaceRotation.score, but scores the three-finger pose."""
       if not lm:
           return {self.CW: 0.0, self.CCW: 0.0}
       pose = three_pose_score(lm, self.holding)
       return {
           self.CW: _combine(pose, _clamp(self.offset / STEP)),
           self.CCW: _combine(pose, _clamp(-self.offset / STEP)),
       }




class ThreeTurn:
   """Three fingers up, hand moved left or right and then back to where it started."""


   right = "three_turn_right"
   left = "three_turn_left"
   _min_dx = 0.06     # sideways travel, fraction of frame width
   _home_dx = 0.03    # this close to the start counts as back
   _max_dy = 0.03     # more vertical drift than this is a circle, not a skip
   _window_s = 0.8    # time to make the move
   _return_s = 1.5    # time to come back after it


   def __init__(self) -> None:
       self._hist: list[tuple[float, float, float]] = []
       self._pending: tuple[str, float, float, float] | None = None  # (gesture, start x, start y, move time)


   def cancel(self) -> None:
       """Drop any half-finished move (e.g. the hand is drawing a temp circle)."""
       self._hist.clear()
       self._pending = None


   def update(self, lm) -> str | None:
       # Hand must point up (sideways three fingers is the AC gesture).
       if not lm or not _is_upright(lm) or not is_three(lm, relaxed=bool(self._hist) or self._pending is not None):
           self._hist.clear()
           self._pending = None
           return None


       now = time.time()
       x, y = lm[9].x, lm[9].y


       if self._pending is not None:
           gesture, start, start_y, t = self._pending
           if abs(y - start_y) > self._max_dy or now - t > self._return_s:
               self._pending = None
               self._hist.clear()
           elif abs(x - start) <= self._home_dx:
               self._pending = None
               self._hist.clear()
               return gesture
           return None


       self._hist.append((now, x, y))
       self._hist = [h for h in self._hist if now - h[0] <= self._window_s]
       if len(self._hist) < 3:
           return None


       _, start, start_y = self._hist[0]
       dx = x - start
       if abs(dx) >= self._min_dx and abs(y - start_y) <= self._max_dy:
           # Frame is mirrored, so +x is the user's right.
           self._pending = (self.right if dx > 0 else self.left, start, start_y, now)
       return None


   def score(self, lm) -> dict[str, float]:
       """Call after update(). First half = travel out (0-50%), second half = return (50-100%)."""
       if not lm:
           return {self.right: 0.0, self.left: 0.0}
       relaxed = bool(self._hist) or self._pending is not None
       pose = min(three_pose_score(lm, relaxed), upright_score(lm))
       x = lm[9].x
       right = left = 0.0
       if self._pending is not None:
           gesture, start, _, _ = self._pending
           back = 1 - _clamp((abs(x - start) - self._home_dx) / (self._min_dx - self._home_dx))
           if gesture == self.right:
               right = 0.5 + 0.5 * back
           else:
               left = 0.5 + 0.5 * back
       elif self._hist:
           dx = x - self._hist[0][1]
           right = 0.5 * _clamp(dx / self._min_dx)
           left = 0.5 * _clamp(-dx / self._min_dx)
       return {self.right: _combine(pose, right), self.left: _combine(pose, left)}


class ThreeSideways:
   """Three fingers pointing sideways, moved up or down and then back to where it started."""


   up = "three_sideways_up"
   down = "three_sideways_down"
   _min_dy = 0.04     # vertical travel, fraction of frame height
   _home_dy = 0.03    # this close to the start counts as back
   _max_dx = 0.03     # more sideways drift than this is a circle, not an AC change
   _window_s = 0.8    # time to make the move
   _return_s = 1.5    # time to come back after it
   _cooldown_s = 0.5  # minimum time between AC changes


   def __init__(self) -> None:
       self._hist: list[tuple[float, float, float]] = []
       self._pending: tuple[str, float, float, float] | None = None  # (gesture, start y, start x, move time)
       self._last_fire = 0.0


   def cancel(self) -> None:
       """Drop any half-finished move (e.g. the hand is drawing a temp circle)."""
       self._hist.clear()
       self._pending = None


   def update(self, lm) -> str | None:
       if not lm or not _is_sideways(lm) or not is_three(lm, relaxed=bool(self._hist) or self._pending is not None):
           self._hist.clear()
           self._pending = None
           return None


       now = time.time()
       x, y = lm[9].x, lm[9].y


       if self._pending is not None:
           gesture, start, start_x, t = self._pending
           if abs(x - start_x) > self._max_dx or now - t > self._return_s:
               self._pending = None
               self._hist.clear()
           elif abs(y - start) <= self._home_dy:
               self._pending = None
               self._hist.clear()
               if now - self._last_fire >= self._cooldown_s:
                   self._last_fire = now
                   return gesture
           return None


       self._hist.append((now, y, x))
       self._hist = [h for h in self._hist if now - h[0] <= self._window_s]
       if len(self._hist) < 3:
           return None


       _, start, start_x = self._hist[0]
       dy = y - start
       if abs(dy) >= self._min_dy and abs(x - start_x) <= self._max_dx:
           # Image y points down, so negative dy is moving up.
           self._pending = (self.up if dy < 0 else self.down, start, start_x, now)
       return None


   def score(self, lm) -> dict[str, float]:
       """Call after update(). First half = travel out (0-50%), second half = return (50-100%)."""
       if not lm:
           return {self.up: 0.0, self.down: 0.0}
       relaxed = bool(self._hist) or self._pending is not None
       pose = min(three_pose_score(lm, relaxed), sideways_score(lm))
       y = lm[9].y
       up = down = 0.0
       if self._pending is not None:
           gesture, start, _, _ = self._pending
           back = 1 - _clamp((abs(y - start) - self._home_dy) / (self._min_dy - self._home_dy))
           if gesture == self.up:
               up = 0.5 + 0.5 * back
           else:
               down = 0.5 + 0.5 * back
       elif self._hist:
           dy = y - self._hist[0][1]           # image y points down, so negative = up
           up = 0.5 * _clamp(-dy / self._min_dy)
           down = 0.5 * _clamp(dy / self._min_dy)
       return {self.up: _combine(pose, up), self.down: _combine(pose, down)}

class ThumbsCall:
   """Thumbs up = accept call, thumbs down = decline call.

   Pose must be held briefly, fires once, and must be released before it can fire again.
   """

   ACCEPT = "thumbs_up"
   DECLINE = "thumbs_down"
   FIST = "closed_fist"
   _hold_s = 0.4      # how long the pose must be held

   def __init__(self) -> None:
       self._kind: str | None = None
       self._start = 0.0
       self._fired = False
       self._miss = 0

   def _current(self, lm) -> str | None:
       if not lm:
           return None
       if is_thumbsup(lm):
           return self.ACCEPT
       if is_thumbsdown(lm):
           return self.DECLINE
       if is_fist(lm):
           return self.FIST
       return None

   def update(self, lm) -> str | None:
       kind = self._current(lm)
       now = time.monotonic()
       if kind is None:
           self._miss += 1
           if self._miss > MISS_LIMIT:      # pose gone: re-arm
               self._kind = None
               self._fired = False
           return None
       self._miss = 0
       if kind != self._kind:               # new pose (or switched up <-> down)
           self._kind = kind
           self._start = now
           self._fired = False
           return None
       if not self._fired and now - self._start >= self._hold_s:
           self._fired = True
           return kind
       return None

   def score(self, lm) -> dict[str, float]:
       """Call after update(). Fills as you hold the pose, full once fired."""
       out = {self.ACCEPT: 0.0, self.DECLINE: 0.0, self.FIST: 0.0}
       kind = self._current(lm)
       if kind is not None and kind == self._kind:
           out[kind] = 1.0 if self._fired else _clamp((time.monotonic() - self._start) / self._hold_s)
       return out


# ---------------------------------------------------------------------------
# Pause / play: open palm, then close into a fist.
# ---------------------------------------------------------------------------

# Each finger as (knuckle, middle joint, top joint, tip).
FINGER_JOINTS = ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
OPEN_STRAIGHT = 0.85    # every finger at least this straight -> open palm
FIST_STRAIGHT = 0.82    # every finger at most this straight -> fist (tight ~0.45, loose ~0.75)
FRAME_ASPECT = 16 / 9   # webcam width / height; x is scaled by this so lengths match in every direction


def _straightness(lm, joints) -> float:
    """Knuckle-to-tip distance / length along the finger: ~1.0 straight, lower when curled.
    Works the same for short fingers (pinky) and for any hand direction."""
    def d(a, b):
        return math.sqrt(((a.x - b.x) * FRAME_ASPECT) ** 2 + (a.y - b.y) ** 2 + ((a.z - b.z) * FRAME_ASPECT) ** 2)
    mcp, pip, dip, tip = (lm[j] for j in joints)
    path = d(mcp, pip) + d(pip, dip) + d(dip, tip)
    return d(mcp, tip) / (path + 1e-6)


def is_open_palm(lm) -> bool:
    """All four fingers straight (thumb ignored)."""
    return min(_straightness(lm, j) for j in FINGER_JOINTS) > OPEN_STRAIGHT


def is_fist(lm) -> bool:
    """All four fingers curled. A thumbs up / down is left to ThumbsCall (accept / decline call)."""
    return (max(_straightness(lm, j) for j in FINGER_JOINTS) < FIST_STRAIGHT
            and not is_thumbsup(lm) and not is_thumbsdown(lm))


class PalmToFist:
    """Open palm, then close into a fist -> toggle play/pause."""

    name = "palm_to_fist"
    _arm_s = 1.0       # fist must come within this long after the open palm
    _hold_frames = 3   # fist frames in a row needed (ignores one-frame flickers)

    def __init__(self) -> None:
        self._palm_t: float | None = None   # last time an open palm was seen
        self._fist_frames = 0
        self._fired = False                 # True after toggling, until the palm opens again

    def _armed(self, now: float) -> bool:
        return self._palm_t is not None and now - self._palm_t <= self._arm_s

    def update(self, lm) -> str | None:
        now = time.time()
        if not lm:
            self._fist_frames = 0
            return None

        if is_open_palm(lm):
            self._palm_t = now
            self._fist_frames = 0
            self._fired = False
            return None

        if is_fist(lm):
            self._fist_frames += 1
            if self._armed(now) and not self._fired and self._fist_frames >= self._hold_frames:
                self._fired = True
                self._palm_t = None
                return self.name
            return None

        self._fist_frames = 0
        return None

    def score(self, lm) -> dict[str, float]:
        """Call after update(). Open palm = halfway, closing to a fist fills the rest."""
        if not lm or self._fired:
            return {self.name: 0.0}
        straight = [_straightness(lm, j) for j in FINGER_JOINTS]
        if self._armed(time.time()):
            fist = _at_most(max(straight), FIST_STRAIGHT)
            progress = 0.5 + 0.5 * fist * min(1.0, self._fist_frames / self._hold_frames)
        else:
            progress = 0.5 * _at_least(min(straight), OPEN_STRAIGHT)
        return {self.name: _clamp(progress)}
