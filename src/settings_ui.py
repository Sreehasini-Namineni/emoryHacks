"""Gear (settings) and wrench (dev tools) buttons at the bottom right, each opening a big centered popup.

Drawn with OpenCV and driven by mouse clicks. The dev tools popup holds the confidence bars.
All sizes below are for a 1080p frame and scale with the frame height.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import cv2
import numpy as np

from controls import GESTURES
from hud import CYAN, CYAN_DIM, WHITE, chamfer, panel, text_mid

AA = cv2.LINE_AA
MARGIN = 72                # distance of the buttons from the screen edge
BTN_R = 30                 # button radius
BTN_GAP = 32               # space between the two buttons
POP_W = 660
ROW_TOP = 84               # y offset of the first control row from the popup top
ROW_H = 48
ITEM_H = 38                # height of one entry in an open dropdown list
DD_X = 260                 # dropdown box starts this far from the popup's left edge
DD_PAD = 30                # dropdown box ends this far from the popup's right edge
DEV_H = 360                # dev tools popup height
TITLES = {"settings": "Settings", "dev": "Dev Tools"}


class SettingsUI:
    def __init__(self, panel, controls) -> None:
        self.panel = panel                 # ConfidencePanel; its bars are drawn inside the dev popup
        self.panel.visible = False         # the standalone bottom-left bars are replaced by the dev popup
        self.controls = controls           # Controls; its bindings are edited in the settings popup
        self.active: str | None = None     # None, "settings" or "dev"
        self.open_dd: str | None = None    # control id whose dropdown list is open
        self._size = (0, 0)                # last frame (w, h), used for click hit-tests
        self._r = BTN_R

    # ---- geometry (everything scales with frame height) ----
    def _d(self, h: int) -> SimpleNamespace:
        k = min(1.4, max(0.6, h / 1080))
        s = lambda v: int(round(v * k))
        n = len(self.controls.actions)
        return SimpleNamespace(
            k=k, s=s, margin=s(MARGIN), r=s(BTN_R), gap=s(BTN_GAP), pw=s(POP_W),
            row_top=s(ROW_TOP), row_h=s(ROW_H), box_h=s(ROW_H) - s(8), item_h=s(ITEM_H),
            dd_x=s(DD_X), dd_pad=s(DD_PAD),
            ph={"settings": s(ROW_TOP) + s(ROW_H) * n + s(28), "dev": s(DEV_H)},
        )

    def _gear_center(self, w: int, h: int):
        d = self._d(h)
        return (w - d.margin - d.r, h - d.margin - d.r)

    def _wrench_center(self, w: int, h: int):
        d = self._d(h)
        gx, gy = self._gear_center(w, h)
        return (gx - 2 * d.r - d.gap, gy)

    def _popup_rect(self, w: int, h: int, key: str):
        """Centered on the screen."""
        d = self._d(h)
        ph = d.ph[key]
        x1 = max(4, (w - d.pw) // 2)
        y1 = max(4, (h - ph) // 2)
        return (x1, y1, x1 + d.pw, y1 + ph)

    def _close_center(self, w: int, h: int, key: str):
        d = self._d(h)
        x1, y1, x2, y2 = self._popup_rect(w, h, key)
        return (x2 - d.s(36), y1 + d.s(40))

    def _row_box(self, w: int, h: int, i: int):
        """Dropdown box of row i: (x1, y1, x2, y2)."""
        d = self._d(h)
        x1, y1, x2, y2 = self._popup_rect(w, h, "settings")
        ry = y1 + d.row_top + i * d.row_h
        return (x1 + d.dd_x, ry, x2 - d.dd_pad, ry + d.box_h)

    def _list_rect(self, w: int, h: int, i: int, n: int):
        """Open list under row i's box, pushed up if it would run off the frame."""
        d = self._d(h)
        bx1, by1, bx2, by2 = self._row_box(w, h, i)
        top = by2 + 2
        list_h = n * d.item_h
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
        d = self._d(h)
        # An open dropdown list gets the click first (it can overlap other widgets).
        if self.open_dd is not None:
            self._click_list(x, y, w, h)
            return
        for key, center in (("settings", self._gear_center(w, h)), ("dev", self._wrench_center(w, h))):
            if math.hypot(x - center[0], y - center[1]) <= d.r:
                self.active = None if self.active == key else key
                self.open_dd = None
                return
        if self.active is None:
            return
        cx, cy = self._close_center(w, h, self.active)
        if math.hypot(x - cx, y - cy) <= d.s(20):                # close X
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
        d = self._d(h)
        cid = self.open_dd
        self.open_dd = None                                     # any click closes the list
        opts = self.controls.options_for(cid)
        i = list(self.controls.actions).index(cid)
        lx1, ly1, lx2, ly2 = self._list_rect(w, h, i, len(opts))
        if lx1 <= x <= lx2 and ly1 <= y < ly2:
            gesture, _ = opts[(y - ly1) // d.item_h]
            self.controls.set_binding(cid, gesture)

    # ---- icons ----
    def _draw_gear(self, frame, center, color, hole_color) -> None:
        cx, cy = center
        r = self._r * 0.75
        for i in range(8):
            a = i * math.pi / 4
            p1 = (int(cx + math.cos(a) * r * 0.55), int(cy + math.sin(a) * r * 0.55))
            p2 = (int(cx + math.cos(a) * r * 0.95), int(cy + math.sin(a) * r * 0.95))
            cv2.line(frame, p1, p2, color, max(4, int(r * 0.3)), AA)
        cv2.circle(frame, center, int(r * 0.65), color, -1, AA)
        cv2.circle(frame, center, int(r * 0.25), hole_color, -1, AA)

    def _draw_wrench(self, frame, center, color, bg) -> None:
        cx, cy = center
        s = self._r / 22.0

        def pt(dx, dy):
            return (int(round(cx + dx * s)), int(round(cy + dy * s)))

        cv2.line(frame, pt(-11, 11), pt(3, -3), color, max(3, int(5 * s)), AA)       # handle
        cv2.circle(frame, pt(-11, 11), int(3.5 * s), color, -1, AA)                  # handle end
        cv2.circle(frame, pt(6, -6), int(8 * s), color, -1, AA)                      # head
        cv2.line(frame, pt(6, -6), pt(12, -12), bg, max(3, int(6 * s)), AA)          # open jaw
        cv2.circle(frame, pt(9, -9), int(3 * s), bg, -1, AA)                         # jaw notch

    def _draw_button(self, frame, center, key: str, icon) -> None:
        bg = (40, 30, 20)
        on = self.active == key
        cv2.circle(frame, center, self._r + 6, CYAN if on else CYAN_DIM, 1, AA)      # soft outer ring
        cv2.circle(frame, center, self._r, bg, -1, AA)
        cv2.circle(frame, center, self._r, CYAN if on else CYAN_DIM, 2, AA)
        icon(frame, center, CYAN if on else (235, 215, 160), bg)

    # ---- drawing ----
    def _draw_rows(self, frame, w: int, h: int) -> None:
        d = self._d(h)
        x1, y1, x2, y2 = self._popup_rect(w, h, "settings")
        for i, (cid, (label, _)) in enumerate(self.controls.actions.items()):
            bx1, by1, bx2, by2 = self._row_box(w, h, i)
            cy = (by1 + by2) // 2
            text_mid(frame, label, x1 + d.s(32), cy, d.s(25), WHITE)
            is_open = self.open_dd == cid
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (64, 48, 30), -1)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), CYAN if is_open else CYAN_DIM, 2 if is_open else 1, AA)
            g = self.controls.bindings[cid]
            text_mid(frame, GESTURES[g] if g else "Off", bx1 + d.s(14), cy, d.s(25),
                     WHITE if g else (150, 140, 120))
            ax = bx2 - d.s(24)                                  # dropdown arrow
            t = d.s(7)
            tri = np.array([(ax - t, cy - t // 2), (ax + t, cy - t // 2), (ax, cy + t // 2 + 2)], np.int32)
            cv2.fillPoly(frame, [tri], CYAN, AA)

    def _draw_list(self, frame, w: int, h: int) -> None:
        d = self._d(h)
        cid = self.open_dd
        opts = self.controls.options_for(cid)
        i = list(self.controls.actions).index(cid)
        lx1, ly1, lx2, ly2 = self._list_rect(w, h, i, len(opts))
        box = np.array([(lx1, ly1), (lx2, ly1), (lx2, ly2), (lx1, ly2)], np.int32)
        panel(frame, box, alpha=0.96, blur=False, edge=CYAN)
        current = self.controls.bindings[cid]
        for k, (g, label) in enumerate(opts):
            top = ly1 + k * d.item_h
            if g == current:
                cv2.rectangle(frame, (lx1 + 2, top + 1), (lx2 - 2, top + d.item_h - 1), (95, 70, 20), -1)
            text_mid(frame, label, lx1 + d.s(14), top + d.item_h // 2, d.s(23), CYAN if g == current else WHITE)

    def _draw_bars(self, frame, x1, y1, x2, y2, d) -> None:
        """Render the confidence bars off-screen, enlarge them, and center them in the dev popup.

        (Done from the outside so confidence.py doesn't have to change. If you later give
        ConfidencePanel its own scale/size, replace this with a direct draw_at call.)
        """
        area = frame[y1 + d.s(70):y2 - d.s(10), x1 + d.s(10):x2 - d.s(10)]
        if area.size == 0:
            return
        bg = tuple(int(v) for v in np.median(area.reshape(-1, 3), axis=0))     # popup's own color
        canvas = np.full((300, 520, 3), bg, np.uint8)
        self.panel.draw_at(canvas, 20, 20)
        diff = np.abs(canvas.astype(np.int16) - np.array(bg, np.int16)).max(axis=2)
        ys, xs = np.nonzero(diff > 6)
        if len(xs) == 0:
            return
        pad = 10
        bx0, bx1 = max(0, xs.min() - pad), min(canvas.shape[1], xs.max() + 1 + pad)
        by0, by1 = max(0, ys.min() - pad), min(canvas.shape[0], ys.max() + 1 + pad)
        crop = canvas[by0:by1, bx0:bx1]
        avail_w, avail_h = (x2 - x1) - d.s(60), (y2 - y1) - d.s(70) - d.s(24)
        sc = min(d.pw * 0.66 / max(1, crop.shape[1]), avail_h / max(1, crop.shape[0]), 2.6)
        sc = max(1.0, sc)
        up = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
        uh, uw = up.shape[:2]
        cx = (x1 + x2) // 2
        cy = (y1 + d.s(70) + y2) // 2                              # centered below the title line
        px, py = cx - uw // 2, cy - uh // 2
        fp = int(pad * sc)                                         # feather the pasted edge
        wgt = np.zeros((uh, uw), np.float32)
        wgt[fp:uh - fp, fp:uw - fp] = 1.0
        wgt = cv2.GaussianBlur(wgt, (0, 0), max(1.0, fp / 2.5))[..., None]
        H, W = frame.shape[:2]
        x0, y0, xe, ye = max(0, px), max(0, py), min(W, px + uw), min(H, py + uh)
        if xe <= x0 or ye <= y0:
            return
        roi = frame[y0:ye, x0:xe]
        sub, wsub = up[y0 - py:ye - py, x0 - px:xe - px], wgt[y0 - py:ye - py, x0 - px:xe - px]
        roi[:] = (roi * (1 - wsub) + sub * wsub).astype(np.uint8)

    def draw(self, frame) -> None:
        h, w = frame.shape[:2]
        self._size = (w, h)
        d = self._d(h)
        self._r = d.r

        if self.active is not None:                               # dim the camera behind the popup
            frame[:] = cv2.convertScaleAbs(frame, alpha=0.6)

        self._draw_button(frame, self._gear_center(w, h), "settings", self._draw_gear)
        self._draw_button(frame, self._wrench_center(w, h), "dev", self._draw_wrench)

        if self.active is None:
            return

        x1, y1, x2, y2 = self._popup_rect(w, h, self.active)
        panel(frame, chamfer(x1, y1, x2 - x1, y2 - y1, d.s(26)), alpha=0.86)

        text_mid(frame, TITLES[self.active].upper(), x1 + d.s(32), y1 + d.s(40), d.s(36), CYAN)
        cv2.line(frame, (x1 + d.s(32), y1 + d.s(68)), (x2 - d.s(32), y1 + d.s(68)), CYAN_DIM, 1, AA)
        cx, cy = self._close_center(w, h, self.active)            # close X
        e = d.s(9)
        cv2.line(frame, (cx - e, cy - e), (cx + e, cy + e), CYAN, 3, AA)
        cv2.line(frame, (cx - e, cy + e), (cx + e, cy - e), CYAN, 3, AA)

        if self.active == "dev":
            self._draw_bars(frame, x1, y1, x2, y2, d)

        if self.active == "settings":
            self._draw_rows(frame, w, h)
            if self.open_dd is not None:
                self._draw_list(frame, w, h)