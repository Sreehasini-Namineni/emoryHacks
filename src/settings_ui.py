"""Gear (settings) and wrench (dev tools) buttons at the bottom right, each opening a popup.

Drawn with OpenCV and driven by mouse clicks. The dev tools popup holds the confidence bars.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

AA = cv2.LINE_AA
BTN_R = 22                 # button radius
BTN_GAP = 10               # space between the two buttons
MARGIN = 16
POP_W = 300
# key -> (title, popup height)
POPUPS = {"settings": ("Settings", 64), "dev": ("Dev Tools", 180)}


class SettingsUI:
    def __init__(self, panel) -> None:
        self.panel = panel                 # ConfidencePanel; its bars are drawn inside the dev popup
        self.panel.visible = False         # the standalone bottom-left bars are replaced by the dev popup
        self.active: str | None = None     # None, "settings" or "dev"
        self._size = (0, 0)                # last frame (w, h), used for click hit-tests

    # ---- geometry ----
    def _gear_center(self, w: int, h: int):
        return (w - MARGIN - BTN_R, h - MARGIN - BTN_R)

    def _wrench_center(self, w: int, h: int):
        gx, gy = self._gear_center(w, h)
        return (gx - 2 * BTN_R - BTN_GAP, gy)

    def _popup_rect(self, w: int, h: int, key: str):
        _, cy = self._gear_center(w, h)
        x2 = w - MARGIN
        y2 = cy - BTN_R - 10
        return (x2 - POP_W, y2 - POPUPS[key][1], x2, y2)

    # ---- mouse ----
    def on_mouse(self, event, x, y, flags, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        w, h = self._size
        if not w:
            return
        for key, center in (("settings", self._gear_center(w, h)), ("dev", self._wrench_center(w, h))):
            if math.hypot(x - center[0], y - center[1]) <= BTN_R:
                self.active = None if self.active == key else key
                return
        if self.active is None:
            return
        x1, y1, x2, y2 = self._popup_rect(w, h, self.active)
        if math.hypot(x - (x2 - 22), y - (y1 + 22)) <= 14:      # close X
            self.active = None

    # ---- icons ----
    def _draw_gear(self, frame, center, color, hole_color) -> None:
        cx, cy = center
        r = BTN_R * 0.75
        for i in range(8):
            a = i * math.pi / 4
            p1 = (int(cx + math.cos(a) * r * 0.55), int(cy + math.sin(a) * r * 0.55))
            p2 = (int(cx + math.cos(a) * r * 0.95), int(cy + math.sin(a) * r * 0.95))
            cv2.line(frame, p1, p2, color, 5, AA)
        cv2.circle(frame, center, int(r * 0.65), color, -1, AA)
        cv2.circle(frame, center, int(r * 0.25), hole_color, -1, AA)

    def _draw_wrench(self, frame, center, color, bg) -> None:
        cx, cy = center
        s = BTN_R / 22.0

        def pt(dx, dy):
            return (int(round(cx + dx * s)), int(round(cy + dy * s)))

        cv2.line(frame, pt(-11, 11), pt(3, -3), color, max(3, int(5 * s)), AA)       # handle
        cv2.circle(frame, pt(-11, 11), int(3.5 * s), color, -1, AA)                  # handle end
        cv2.circle(frame, pt(6, -6), int(8 * s), color, -1, AA)                      # head
        cv2.line(frame, pt(6, -6), pt(12, -12), bg, max(3, int(6 * s)), AA)          # open jaw
        cv2.circle(frame, pt(9, -9), int(3 * s), bg, -1, AA)                         # jaw notch

    def _draw_button(self, frame, center, key: str, icon) -> None:
        bg = (40, 40, 40)
        cv2.circle(frame, center, BTN_R, bg, -1, AA)
        cv2.circle(frame, center, BTN_R, (120, 120, 120), 1, AA)
        icon(frame, center, (255, 255, 255) if self.active == key else (200, 200, 200), bg)

    # ---- drawing ----
    def draw(self, frame) -> None:
        h, w = frame.shape[:2]
        self._size = (w, h)

        self._draw_button(frame, self._gear_center(w, h), "settings", self._draw_gear)
        self._draw_button(frame, self._wrench_center(w, h), "dev", self._draw_wrench)

        if self.active is None:
            return

        title = POPUPS[self.active][0]
        x1, y1, x2, y2 = self._popup_rect(w, h, self.active)
        roi = frame[y1:y2, x1:x2]
        cv2.addWeighted(roi, 0.15, np.full_like(roi, 30), 0.85, 0, roi)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 120, 120), 1)

        cv2.putText(frame, title, (x1 + 16, y1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, AA)
        cx, cy = x2 - 22, y1 + 22                                 # close X
        cv2.line(frame, (cx - 6, cy - 6), (cx + 6, cy + 6), (200, 200, 200), 2, AA)
        cv2.line(frame, (cx - 6, cy + 6), (cx + 6, cy - 6), (200, 200, 200), 2, AA)

        if self.active == "dev":
            self.panel.draw_at(frame, x1 + 16, y1 + 70)