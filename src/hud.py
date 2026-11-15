"""Sci-fi HUD drawn on top of the camera frame. OpenCV for shapes, Pillow for text."""

from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ASSETS = Path(__file__).resolve().parent / "assets"
FONT_PATH = ASSETS / "Rajdhani-SemiBold.ttf"

# BGR colors
CYAN = (255, 220, 90)
CYAN_DIM = (170, 130, 40)
WHITE = (255, 255, 255)
GREEN = (90, 230, 120)
AMBER = (0, 165, 255)
PANEL_TINT = (45, 35, 25)
STEP = math.radians(30)          # same as gestures.STEP

CONNECTIONS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
               (9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]


# ---------- helpers ----------
_fonts: dict[int, ImageFont.FreeTypeFont] = {}

def _font(size: int):
    if size not in _fonts:
        _fonts[size] = ImageFont.truetype(str(FONT_PATH), size)
    return _fonts[size]


def text(frame, s: str, xy, size: int = 28, color=WHITE) -> None:
    """Anti-aliased TTF text. Only the small region around the text goes through Pillow."""
    x, y = xy
    l, t, r, b = _font(size).getbbox(s)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(frame.shape[1], x + r + 4), min(frame.shape[0], y + b + 4)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    img = Image.fromarray(np.ascontiguousarray(roi[..., ::-1]))
    ImageDraw.Draw(img).text((x - x0, y - y0), s, font=_font(size), fill=(color[2], color[1], color[0]))
    roi[:] = np.asarray(img)[..., ::-1]


_icons: dict = {}

def icon(frame, name: str, x: int, y: int, size: int = 44, color=CYAN) -> None:
    """Tinted PNG icon with alpha. Silently skipped if the file is missing."""
    key = (name, size)
    if key not in _icons:
        im = cv2.imread(str(ASSETS / f"{name}.png"), cv2.IMREAD_UNCHANGED)
        _icons[key] = None if im is None or im.shape[2] < 4 else cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
    im = _icons[key]
    if im is None:
        return
    roi = frame[y:y + size, x:x + size]
    a = im[..., 3:4].astype(np.float32) / 255.0
    roi[:] = (roi * (1 - a) + np.array(color, np.float32) * a).astype(np.uint8)


def chamfer(x: int, y: int, w: int, h: int, c: int = 18) -> np.ndarray:
    """Rectangle with top-left and bottom-right corners cut off."""
    return np.array([(x + c, y), (x + w, y), (x + w, y + h - c), (x + w - c, y + h), (x, y + h), (x, y + c)], np.int32)


def panel(frame, pts: np.ndarray, alpha: float = 0.55, blur: bool = True, edge=CYAN) -> None:
    """Translucent (optionally frosted) polygon with a cyan border. alpha=1.0 gives a solid panel."""
    x, y, w, h = cv2.boundingRect(pts)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    mask = np.zeros(roi.shape[:2], np.uint8)
    cv2.fillPoly(mask, [pts - np.array([x0, y0])], 255)
    base = cv2.GaussianBlur(roi, (0, 0), 10) if blur else roi
    mixed = cv2.addWeighted(base, 1 - alpha, np.full_like(roi, PANEL_TINT), alpha, 0)
    roi[mask > 0] = mixed[mask > 0]
    cv2.polylines(frame, [pts], True, edge, 2, cv2.LINE_AA)


def ease_out(t: float) -> float:
    return 1 - (1 - min(max(t, 0.0), 1.0)) ** 3


def fade(color, k: float):
    return tuple(int(c * k) for c in color)


class Glow:
    """Draw bright lines on .layer, then end() adds a blurred halo plus the crisp lines to the frame."""

    def begin(self, frame) -> None:
        self.layer = np.zeros_like(frame)

    def end(self, frame, strength: float = 2.2) -> None:
        small = cv2.resize(self.layer, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 3)
        halo = cv2.resize(small, (frame.shape[1], frame.shape[0]))
        cv2.add(frame, cv2.convertScaleAbs(halo, alpha=strength), frame)
        cv2.add(frame, self.layer, frame)


# ---------- the HUD ----------
class Hud:
    TOAST_S = 2.0

    def __init__(self) -> None:
        self.glow = Glow()
        self._toast_seen = -1e9            # last controls.last_action_time we reacted to
        self._toast: tuple[str, float] | None = None
        self._fired_at = -1e9              # for the expanding ring

    def draw(self, frame, hands, lm, controls, progress: float = 0.0) -> None:
        """hands: all detected hands. lm: the control hand (or None). progress: 0..1 circle sweep."""
        now = time.monotonic()
        h, w = frame.shape[:2]

        if controls.last_action_time != self._toast_seen:      # a control just fired
            self._toast_seen = controls.last_action_time
            self._toast = (controls.last_action, now)
            self._fired_at = now

        # 1) glowing hand + rings (everything on one glow layer)
        self.glow.begin(frame)
        for hand in hands:
            pts = [(int(p.x * w), int(p.y * h)) for p in hand]
            for a, b in CONNECTIONS:
                cv2.line(self.glow.layer, pts[a], pts[b], CYAN, 2, cv2.LINE_AA)
            for p in pts:
                cv2.circle(self.glow.layer, p, 4, WHITE, -1, cv2.LINE_AA)
        if lm is not None:
            self._rings(w, h, lm, now, progress)
        self.glow.end(frame)

        # 2) panels on top
        self._brand(frame)
        self._status(frame, w, lm is not None, now)
        self._stats(frame, controls)
        self._toast_card(frame, w, now)

    def _rings(self, w, h, lm, now, progress) -> None:
        cx = int((lm[0].x + lm[9].x) / 2 * w)
        cy = int((lm[0].y + lm[9].y) / 2 * h)
        L = self.glow.layer
        for i, r in enumerate((70, 95, 120)):
            start = (now * 40 * (1 if i % 2 == 0 else -1) + i * 70) % 360
            for off in (0, 180):
                cv2.ellipse(L, (cx, cy), (r, r), 0, start + off, start + off + 100, CYAN_DIM, 2, cv2.LINE_AA)
        if progress > 0.02:                                    # fills as a circle gesture is swept
            cv2.ellipse(L, (cx, cy), (140, 140), -90, 0, 360 * min(progress, 1.0), WHITE, 3, cv2.LINE_AA)
        age = now - self._fired_at
        if age < 0.5:                                          # pulse when something fires
            cv2.circle(L, (cx, cy), int(70 + ease_out(age / 0.5) * 110), fade(CYAN, 1 - age / 0.5), 3, cv2.LINE_AA)

    def _brand(self, frame) -> None:
        panel(frame, chamfer(20, 16, 260, 64, 20))
        icon(frame, "car", 36, 26, 44)
        text(frame, "DRIVE.AI", (96, 28), 34)

    def _status(self, frame, w, tracking: bool, now) -> None:
        pulse = 0.6 + 0.4 * math.sin(now * 4)
        color = GREEN if tracking else AMBER
        cv2.circle(frame, (w - 190, 36), 8, fade(color, pulse), -1, cv2.LINE_AA)
        text(frame, "System Active" if tracking else "Show hand", (w - 170, 20), 28, color)

    def _stats(self, frame, controls) -> None:
        rows = [
            ("volume", "Volume", f"{controls.volume}%"),
            ("temp", "Temp", f"{controls.temp}°F"),
            ("music", "Music", "Playing" if getattr(controls, "playing", True) else "Paused"),
            ("ac", "AC", f"Level {controls.ac}"),
        ]
        for i, (ic, label, value) in enumerate(rows):
            x, y = 20, 110 + i * 92
            panel(frame, chamfer(x, y, 250, 80, 14), alpha=0.5)
            icon(frame, ic, x + 16, y + 18, 44)
            text(frame, label, (x + 82, y + 8), 22, CYAN)
            text(frame, value, (x + 82, y + 32), 34)

    def _toast_card(self, frame, w, now) -> None:
        if self._toast is None:
            return
        title, t0 = self._toast
        age = now - t0
        if age > self.TOAST_S:
            self._toast = None
            return
        slide = int((1 - ease_out(age / 0.25)) * 80)          # slides in from the right
        a = min(1.0, (self.TOAST_S - age) / 0.3)              # fades out over the last 0.3s
        x, y, bw, bh = w - 340 + slide, 90, 320, 96
        before = frame.copy()
        panel(frame, chamfer(x, y, bw, bh, 18), alpha=0.6)
        icon(frame, "fast_forward", x + 22, y + 26, 44)
        text(frame, title, (x + 86, y + 12), 36)
        text(frame, "Gesture Recognized!", (x + 86, y + 54), 22, CYAN)
        if a < 1.0:
            cv2.addWeighted(frame, a, before, 1 - a, 0, frame)