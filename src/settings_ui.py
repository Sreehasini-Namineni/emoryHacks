"""Gear (settings) and wrench (dev tools) buttons at the bottom right, each opening a popup.

Drawn with OpenCV and driven by mouse clicks. The dev tools popup holds the confidence bars.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from controls import GESTURES

AA = cv2.LINE_AA
BTN_R = 22                 # button radius
BTN_GAP = 10               # space between the two buttons
MARGIN = 16
POP_W = 380
FONT = cv2.FONT_HERSHEY_SIMPLEX
ROW_TOP = 50               # y offset of the first control row from the popup top
ROW_H = 30
ITEM_H = 24                # height of one entry in an open dropdown list
DD_X = 160                 # dropdown box starts this far from the popup's left edge
DD_PAD = 16                # dropdown box ends this far from the popup's right edge
# key -> (title, popup height). Settings height = ROW_TOP + ROW_H * (number of controls) + 10
POPUPS = {"settings": ("Settings", 360), "dev": ("Dev Tools", 180)}


class SettingsUI:
    def __init__(self, panel, controls) -> None:
        self.panel = panel                 # ConfidencePanel; its bars are drawn inside the dev popup
        self.panel.visible = False         # the standalone bottom-left bars are replaced by the dev popup
        self.controls = controls           # Controls; its bindings are edited in the settings popup
        self.active: str | None = None     # None, "settings" or "dev"
        self.open_dd: str | None = None    # control id whose dropdown list is open
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

    def _row_box(self, w: int, h: int, i: int):
        """Dropdown box of row i: (x1, y1, x2, y2)."""
        x1, y1, x2, y2 = self._popup_rect(w, h, "settings")
        ry = y1 + ROW_TOP + i * ROW_H
        return (x1 + DD_X, ry, x2 - DD_PAD, ry + ROW_H - 6)

    def _list_rect(self, w: int, h: int, i: int, n: int):
        """Open list under row i's box, pushed up if it would run off the frame."""
        bx1, by1, bx2, by2 = self._row_box(w, h, i)
        top = by2 + 2
        list_h = n * ITEM_H
        if top + list_h > h - 4:
            top = max(4, h - 4 - list_h)
        return (bx1, top, bx2, top + list_h)
    
    # ---- mouse ----
    def on_mouse(self, event, x, y, flags, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        w, h = self._size
        if not w:
            return
        # An open dropdown list gets the click first (it can overlap other widgets).
        if self.open_dd is not None:
            self._click_list(x, y, w, h)
            return
        for key, center in (("settings", self._gear_center(w, h)), ("dev", self._wrench_center(w, h))):
            if math.hypot(x - center[0], y - center[1]) <= BTN_R:
                self.active = None if self.active == key else key
                self.open_dd = None
                return
        if self.active is None:
            return
        x1, y1, x2, y2 = self._popup_rect(w, h, self.active)
        if math.hypot(x - (x2 - 22), y - (y1 + 22)) <= 14:      # close X
            self.active = None
            return
        if self.active == "settings":
            self._click_rows(x, y, w, h)

    def _click_rows(self, x, y, w, h) -> None:
        for i, cid in enumerate(self.controls.actions):
            bx1, by1, bx2, by2 = self._row_box(w, h, i)
            if bx1 <= x <= bx2 and by1 <= y <= by2:
                self.open_dd = cid
                return

    def _click_list(self, x, y, w, h) -> None:
        cid = self.open_dd
        self.open_dd = None                                     # any click closes the list
        opts = self.controls.options_for(cid)
        i = list(self.controls.actions).index(cid)
        lx1, ly1, lx2, ly2 = self._list_rect(w, h, i, len(opts))
        if lx1 <= x <= lx2 and ly1 <= y < ly2:
            gesture, _ = opts[(y - ly1) // ITEM_H]
            self.controls.set_binding(cid, gesture)

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
    def _draw_rows(self, frame, w: int, h: int) -> None:
        x1, y1, x2, y2 = self._popup_rect(w, h, "settings")
        for i, (cid, (label, _)) in enumerate(self.controls.actions.items()):
            bx1, by1, bx2, by2 = self._row_box(w, h, i)
            ty = (by1 + by2) // 2 + 5
            cv2.putText(frame, label, (x1 + 16, ty), FONT, 0.45, (255, 255, 255), 1, AA)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (60, 60, 60), -1)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (120, 120, 120), 1)
            g = self.controls.bindings[cid]
            cv2.putText(frame, GESTURES[g] if g else "Off", (bx1 + 8, ty), FONT, 0.45,
                        (255, 255, 255) if g else (160, 160, 160), 1, AA)
            ax, ay = bx2 - 14, (by1 + by2) // 2                 # dropdown arrow
            tri = np.array([(ax - 5, ay - 3), (ax + 5, ay - 3), (ax, ay + 4)], np.int32)
            cv2.fillPoly(frame, [tri], (200, 200, 200), AA)

    def _draw_list(self, frame, w: int, h: int) -> None:
        cid = self.open_dd
        opts = self.controls.options_for(cid)
        i = list(self.controls.actions).index(cid)
        lx1, ly1, lx2, ly2 = self._list_rect(w, h, i, len(opts))
        cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (45, 45, 45), -1)
        cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (160, 160, 160), 1)
        current = self.controls.bindings[cid]
        for k, (g, label) in enumerate(opts):
            cv2.putText(frame, label, (lx1 + 8, ly1 + k * ITEM_H + 17), FONT, 0.45,
                        (0, 255, 255) if g == current else (255, 255, 255), 1, AA)
    
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

        if self.active == "settings":
            self._draw_rows(frame, w, h)
            if self.open_dd is not None:
                self._draw_list(frame, w, h)