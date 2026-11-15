"""Gear button (bottom right) + settings popup, drawn with OpenCV and driven by mouse clicks."""

from __future__ import annotations

import math

import cv2
import numpy as np

AA = cv2.LINE_AA
BTN_R = 22                 # gear button radius
MARGIN = 16
POP_W, POP_H = 300, 150


class SettingsUI:
    def __init__(self, panel) -> None:
        self.panel = panel                 # ConfidencePanel; its .visible is the setting
        self.open = False
        self._size = (0, 0)                # last frame (w, h), used for click hit-tests

    # ---- geometry ----
    def _btn_center(self, w: int, h: int):
        return (w - MARGIN - BTN_R, h - MARGIN - BTN_R)

    def _popup_rect(self, w: int, h: int):
        _, cy = self._btn_center(w, h)
        x2 = w - MARGIN
        y2 = cy - BTN_R - 10
        return (x2 - POP_W, y2 - POP_H, x2, y2)

    # ---- mouse ----
    def on_mouse(self, event, x, y, flags, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        w, h = self._size
        if not w:
            return
        bx, by = self._btn_center(w, h)
        if math.hypot(x - bx, y - by) <= BTN_R:
            self.open = not self.open
            return
        if not self.open:
            return
        x1, y1, x2, y2 = self._popup_rect(w, h)
        if not (x1 <= x <= x2 and y1 <= y <= y2):
            return
        if math.hypot(x - (x2 - 22), y - (y1 + 22)) <= 14:      # close X
            self.open = False
            return
        if x1 + 16 <= x <= x2 - 16 and y1 + 62 <= y <= y1 + 98:  # confidence-bars row
            self.panel.visible = not self.panel.visible

    # ---- drawing ----
    def _draw_gear(self, frame, center, r: float, color, hole_color) -> None:
        cx, cy = center
        for i in range(8):
            a = i * math.pi / 4
            p1 = (int(cx + math.cos(a) * r * 0.55), int(cy + math.sin(a) * r * 0.55))
            p2 = (int(cx + math.cos(a) * r * 0.95), int(cy + math.sin(a) * r * 0.95))
            cv2.line(frame, p1, p2, color, 5, AA)
        cv2.circle(frame, center, int(r * 0.65), color, -1, AA)
        cv2.circle(frame, center, int(r * 0.25), hole_color, -1, AA)

    def _draw_switch(self, frame, x: int, y: int, on: bool) -> None:
        w, h = 44, 22
        r = h // 2
        col = (0, 200, 0) if on else (90, 90, 90)
        cv2.circle(frame, (x + r, y), r, col, -1, AA)
        cv2.circle(frame, (x + w - r, y), r, col, -1, AA)
        cv2.rectangle(frame, (x + r, y - r), (x + w - r, y + r), col, -1)
        knob_x = x + w - r if on else x + r
        cv2.circle(frame, (knob_x, y), r - 3, (255, 255, 255), -1, AA)

    def draw(self, frame) -> None:
        h, w = frame.shape[:2]
        self._size = (w, h)

        # Gear button.
        c = self._btn_center(w, h)
        bg = (40, 40, 40)
        cv2.circle(frame, c, BTN_R, bg, -1, AA)
        cv2.circle(frame, c, BTN_R, (120, 120, 120), 1, AA)
        self._draw_gear(frame, c, BTN_R * 0.75, (255, 255, 255) if self.open else (200, 200, 200), bg)

        if not self.open:
            return

        # Popup (semi-transparent panel).
        x1, y1, x2, y2 = self._popup_rect(w, h)
        roi = frame[y1:y2, x1:x2]
        cv2.addWeighted(roi, 0.15, np.full_like(roi, 30), 0.85, 0, roi)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 120, 120), 1)

        cv2.putText(frame, "Settings", (x1 + 16, y1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, AA)
        cx, cy = x2 - 22, y1 + 22                                 # close X
        cv2.line(frame, (cx - 6, cy - 6), (cx + 6, cy + 6), (200, 200, 200), 2, AA)
        cv2.line(frame, (cx - 6, cy + 6), (cx + 6, cy - 6), (200, 200, 200), 2, AA)

        cv2.putText(frame, "Confidence bars", (x1 + 16, y1 + 86),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, AA)
        self._draw_switch(frame, x2 - 16 - 44, y1 + 80, self.panel.visible)
