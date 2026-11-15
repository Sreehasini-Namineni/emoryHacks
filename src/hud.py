"""Sci-fi HUD drawn on top of the camera frame. OpenCV for shapes, Pillow for text.

Icons: if assets/<name>.png exists it is used (tinted); otherwise a built-in vector icon is drawn,
so no image files are needed.
"""

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
CYAN = (250, 244, 186)           # #baf4fa  (icons, card borders, accents)
CYAN_DIM = (140, 134, 100)       # CYAN at ~55%
SKEL = (255, 220, 90)            # hand skeleton blue
WHITE = (255, 255, 255)
GREEN = (90, 230, 120)
AMBER = (0, 165, 255)
PANEL_TINT = (38, 26, 12)        # deep navy, like the mockup
STEP = math.radians(30)          # same as gestures.STEP

CONNECTIONS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
               (9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
AA = cv2.LINE_AA


# ---------- helpers ----------
_fonts: dict[int, ImageFont.FreeTypeFont] = {}

FONT_CANDIDATES = [
    FONT_PATH,
    Path("/System/Library/Fonts/Helvetica.ttc"),                  # macOS
    Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),    # macOS
    Path("/Library/Fonts/Arial Unicode.ttf"),                     # macOS
    Path("C:/Windows/Fonts/segoeuib.ttf"),                        # Windows
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"), # Linux
]

def _font(size: int):
    if size not in _fonts:
        for p in FONT_CANDIDATES:
            try:
                _fonts[size] = ImageFont.truetype(str(p), size)
                break
            except OSError:
                continue
        else:
            _fonts[size] = ImageFont.load_default(size)           # last resort (Pillow 10.1+)
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


def text_mid(frame, s: str, x: int, cy: int, size: int = 28, color=WHITE) -> None:
    """Like text(), but x is the left edge and cy is the vertical CENTER of the capital letters."""
    l, t, r, b = _font(size).getbbox("H")
    text(frame, s, (x, int(cy - (t + b) / 2)), size, color)


def text_width(s: str, size: int) -> int:
    l, t, r, b = _font(size).getbbox(s)
    return r - l


# ---------- vector icons (no image files needed) ----------
# Each icon is drawn on a 100x100 grid into a supersampled mask, then shrunk for smooth edges.
def _vector_mask(name: str, size: int) -> np.ndarray:
    if name == "rewind":                                           # fast_forward, mirrored
        return _vector_mask("fast_forward", size)[:, ::-1].copy()
    SS = 4
    S = size * SS
    u = S / 100.0
    m = np.zeros((S, S), np.uint8)

    def P(x, y):
        return (int(round(x * u)), int(round(y * u)))

    def poly(pts, v=255):
        cv2.fillPoly(m, [np.array([P(*p) for p in pts], np.int32)], v, AA)

    def line(a, b, w, v=255):
        cv2.line(m, P(*a), P(*b), v, max(1, int(round(w * u))), AA)
        r = max(1, int(round(w * u / 2)))                          # round caps
        cv2.circle(m, P(*a), r, v, -1, AA)
        cv2.circle(m, P(*b), r, v, -1, AA)

    def disc(c, r, v=255):
        cv2.circle(m, P(*c), int(round(r * u)), v, -1, AA)

    if name == "car":                                              # front view, like the mockup
        poly([(8, 54), (15, 46), (85, 46), (92, 54), (92, 74), (8, 74)])      # body
        poly([(23, 47), (31, 22), (69, 22), (77, 47)])                        # cabin
        poly([(32, 43), (37, 28), (63, 28), (68, 43)], 0)                     # windshield
        disc((24, 59), 6, 0)                                                  # headlights
        disc((76, 59), 6, 0)
        line((38, 65), (62, 65), 4, 0)                                        # grille
        line((22, 74), (22, 86), 14)                                          # wheels
        line((78, 74), (78, 86), 14)
    elif name == "volume":
        poly([(8, 38), (26, 38), (48, 18), (48, 82), (26, 62), (8, 62)])
        for r in (20, 34):
            cv2.ellipse(m, P(50, 50), (int(r * u), int(r * u)), 0, -48, 48, 255, int(6 * u), AA)
    elif name == "temp":
        line((46, 22), (46, 62), 16)                               # stem
        disc((46, 72), 17)                                         # bulb
        line((46, 30), (46, 64), 6, 0)                             # hollow
        disc((46, 72), 9, 0)
        line((46, 52), (46, 72), 4)                                # mercury
        disc((46, 72), 5)
        for y in (26, 40, 54):                                     # ticks
            line((70, y), (84, y), 5)
    elif name == "music":
        cv2.ellipse(m, P(30, 76), (int(14 * u), int(10 * u)), 0, 0, 360, 255, -1, AA)
        cv2.ellipse(m, P(72, 68), (int(14 * u), int(10 * u)), 0, 0, 360, 255, -1, AA)
        line((41, 74), (41, 24), 6)
        line((83, 66), (83, 16), 6)
        poly([(41, 24), (83, 14), (83, 30), (41, 40)])
    elif name == "ac":                                             # snowflake
        c = (50, 50)
        for k in range(6):
            a = math.radians(k * 60 - 90)
            tip = (c[0] + 42 * math.cos(a), c[1] + 42 * math.sin(a))
            line(c, tip, 6)
            b = (c[0] + 27 * math.cos(a), c[1] + 27 * math.sin(a))
            for da in (-60, 60):
                aa = a + math.radians(da)
                line(b, (b[0] + 14 * math.cos(aa), b[1] + 14 * math.sin(aa)), 5)
    elif name == "fast_forward":
        poly([(8, 20), (8, 80), (50, 50)])
        poly([(50, 20), (50, 80), (92, 50)])
    elif name == "check":
        line((16, 54), (40, 76), 11)
        line((40, 76), (84, 26), 11)
    elif name == "cross":
        line((22, 22), (78, 78), 11)
        line((78, 22), (22, 78), 11)
    return cv2.resize(m, (size, size), interpolation=cv2.INTER_AREA)


_icons: dict = {}
_halos: dict = {}

def icon(frame, name: str, x: int, y: int, size: int = 44, color=CYAN, glow: bool = True) -> None:
    """Tinted icon with a soft glow. Uses assets/<name>.png if present, else the built-in vector icon."""
    key = (name, size)
    if key not in _icons:
        p = ASSETS / f"{name}.png"
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED) if p.exists() else None
        if im is not None and im.ndim == 3 and im.shape[2] >= 4:
            alpha = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)[..., 3]
        else:
            alpha = _vector_mask(name, size)
        _icons[key] = alpha.astype(np.float32)[..., None] / 255.0
    a = _icons[key]
    H, W = frame.shape[:2]
    col = np.array(color, np.float32)

    if glow:
        pad = max(4, size // 3)
        if key not in _halos:
            big = np.zeros((size + 2 * pad, size + 2 * pad), np.float32)
            big[pad:pad + size, pad:pad + size] = a[..., 0]
            _halos[key] = cv2.GaussianBlur(big, (0, 0), size * 0.09)[..., None]
        hl = _halos[key]
        hx, hy = x - pad, y - pad
        x0, y0, x1, y1 = max(0, hx), max(0, hy), min(W, hx + hl.shape[1]), min(H, hy + hl.shape[0])
        if x1 > x0 and y1 > y0:
            roi = frame[y0:y1, x0:x1]
            sub = hl[y0 - hy:y1 - hy, x0 - hx:x1 - hx]
            roi[:] = np.clip(roi.astype(np.float32) + col * sub * 0.9, 0, 255).astype(np.uint8)

    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + size), min(H, y + size)
    if x1 <= x0 or y1 <= y0:
        return
    sub = a[y0 - y:y1 - y, x0 - x:x1 - x]
    roi = frame[y0:y1, x0:x1]
    roi[:] = (roi * (1 - sub) + col * sub).astype(np.uint8)


def chamfer(x: int, y: int, w: int, h: int, c: int = 18) -> np.ndarray:
    """Rectangle with top-left and bottom-right corners cut off."""
    return np.array([(x + c, y), (x + w, y), (x + w, y + h - c), (x + w - c, y + h), (x, y + h), (x, y + c)], np.int32)


def rrect(x: int, y: int, w: int, h: int, r: int) -> np.ndarray:
    """Rounded rectangle as a polygon."""
    pts = []
    for cx, cy, a0 in ((x + w - r, y + r, -90), (x + w - r, y + h - r, 0), (x + r, y + h - r, 90), (x + r, y + r, 180)):
        pts.extend(cv2.ellipse2Poly((cx, cy), (r, r), 0, a0, a0 + 90, 10))
    return np.array(pts, np.int32)


def panel(frame, pts: np.ndarray, alpha: float = 0.7, blur: bool = True, edge=CYAN, glow: bool = True,
          thick: int = 2) -> None:
    """Translucent (optionally frosted) polygon with a (glowing) border. alpha=1.0 gives a solid panel."""
    x, y, w, h = cv2.boundingRect(pts)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    mask = np.zeros(roi.shape[:2], np.uint8)
    cv2.fillPoly(mask, [(pts - np.array([x0, y0])).astype(np.int32)], 255)
    if blur:
        sm = cv2.resize(roi, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        base = cv2.resize(cv2.GaussianBlur(sm, (0, 0), 2.5), (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_LINEAR)
    else:
        base = roi
    mixed = cv2.addWeighted(base, 1 - alpha, np.full_like(roi, PANEL_TINT), alpha, 0)
    roi[mask > 0] = mixed[mask > 0]
    if glow:                                                   # soft halo around the border
        pad = 16
        gx0, gy0 = max(0, x - pad), max(0, y - pad)
        gx1, gy1 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
        g = np.zeros((gy1 - gy0, gx1 - gx0, 3), np.uint8)
        cv2.polylines(g, [(pts - np.array([gx0, gy0])).astype(np.int32)], True, edge, 3, AA)
        g = cv2.GaussianBlur(g, (0, 0), 6)
        groi = frame[gy0:gy1, gx0:gx1]
        cv2.add(groi, cv2.convertScaleAbs(g, alpha=0.9), groi)
    cv2.polylines(frame, [pts], True, edge, thick, AA)


def ease_out(t: float) -> float:
    return 1 - (1 - min(max(t, 0.0), 1.0)) ** 3


def fade(color, k: float):
    return tuple(int(c * k) for c in color)


class Glow:
    """Draw bright lines on .layer, then end() adds a blurred halo plus the crisp lines to the frame."""

    def begin(self, frame) -> None:
        self.layer = np.zeros_like(frame)

    def end(self, frame, strength: float = 1.6, inv3=None) -> None:
        """inv3: optional uint8 (h, w, 3) map, 255 = show, 0 = hide (used to tuck the layer behind the hand)."""
        small = cv2.resize(self.layer, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 3)
        halo = cv2.convertScaleAbs(cv2.resize(small, (frame.shape[1], frame.shape[0])), alpha=strength)
        layer = self.layer
        if inv3 is not None:
            halo = cv2.multiply(halo, inv3, scale=1 / 255)
            layer = cv2.multiply(layer, inv3, scale=1 / 255)
        cv2.add(frame, halo, frame)
        cv2.add(frame, layer, frame)


def _toast_icon(title: str) -> str:
    t = title.lower()
    if "call" in t:
        return "check" if "accept" in t else "cross"
    if "track" in t:
        return "rewind" if "<<" in t else "fast_forward"
    if "volume" in t:
        return "volume"
    if "temp" in t:
        return "temp"
    if t.startswith("ac"):
        return "ac"
    if t.startswith("play") or t.startswith("pause"):
        return "music"
    return "fast_forward"


# ---------- the HUD ----------
MARGIN = 40                  # distance of the cards from the screen edge (at 1080p; scales with frame height)
FRAME_INSET = 14             # distance of the screen border lines from the edge


class Hud:
    TOAST_S = 2.0
    AC_S = 2.6               # how long the AC card stays after the level changes

    def __init__(self) -> None:
        self.glow = Glow()                 # hand skeleton
        self.ring_glow = Glow()            # rings behind the hand
        self.k = 1.0                       # UI scale, set each frame from the frame height
        self._toast_seen = -1e9            # last controls.last_action_time we reacted to
        self._toast: tuple[str, float] | None = None
        self._fired_at = -1e9              # for the expanding ring
        self._ac_prev: int | None = None
        self._ac_changed = -1e9
        self._ring_c: np.ndarray | None = None   # smoothed ring center
        self._ring_L = 0.0                       # smoothed hand length (px)
        self._static_key = None
        self._mult3 = None                 # vignette + rounded-corner darkening (uint8 multiplier)
        self._overlay = None               # screen border lines + glow (added to the frame)

    def S(self, v: float) -> int:
        return int(round(v * self.k))

    # ---------- screen border + vignette (built once per frame size) ----------
    @staticmethod
    def _rrect_lines(L, x0, y0, x1, y1, r, color, t) -> None:
        cv2.line(L, (x0 + r, y0), (x1 - r, y0), color, t, AA)
        cv2.line(L, (x0 + r, y1), (x1 - r, y1), color, t, AA)
        cv2.line(L, (x0, y0 + r), (x0, y1 - r), color, t, AA)
        cv2.line(L, (x1, y0 + r), (x1, y1 - r), color, t, AA)
        cv2.ellipse(L, (x0 + r, y0 + r), (r, r), 0, 180, 270, color, t, AA)
        cv2.ellipse(L, (x1 - r, y0 + r), (r, r), 0, 270, 360, color, t, AA)
        cv2.ellipse(L, (x1 - r, y1 - r), (r, r), 0, 0, 90, color, t, AA)
        cv2.ellipse(L, (x0 + r, y1 - r), (r, r), 0, 90, 180, color, t, AA)

    @staticmethod
    def _gline(L, p0, p1, c0, c1, t, n: int = 48) -> None:
        """Line whose color fades from c0 to c1."""
        for i in range(n):
            a, b = i / n, (i + 1) / n
            col = tuple(int(c0[j] + (c1[j] - c0[j]) * (a + b) / 2) for j in range(3))
            pa = (int(p0[0] + (p1[0] - p0[0]) * a), int(p0[1] + (p1[1] - p0[1]) * a))
            pb = (int(p0[0] + (p1[0] - p0[0]) * b), int(p0[1] + (p1[1] - p0[1]) * b))
            cv2.line(L, pa, pb, col, t, AA)

    def _build_static(self, w: int, h: int) -> None:
        f, m = self.S(FRAME_INSET), self.S(MARGIN)
        L = np.zeros((h, w, 3), np.uint8)
        faint, mid, strong = fade(CYAN, 0.28), fade(CYAN, 0.48), fade(CYAN, 0.78)
        dark = (0, 0, 0)

        self._rrect_lines(L, f, f, w - f, h - f, self.S(40), faint, 1)          # outer rounded border

        bw = self.S(380)                                                        # top line leaving the brand tab
        self._gline(L, (m + bw, m), (int(w * 0.58), m), mid, dark, 2)

        c = self.S(70)                                                          # bottom corner brackets
        for sx in (1, -1):
            cx = f if sx == 1 else w - f
            cv2.line(L, (cx, h - f - c), (cx + sx * c, h - f), strong, 2, AA)    # slanted corner
            self._gline(L, (cx + sx * c, h - f), (w // 2, h - f), strong, dark, 2)
            o = self.S(10)                                                      # inner parallel line
            cv2.line(L, (cx + sx * o, h - f - c - o), (cx + sx * (c + o), h - f - o), mid, 1, AA)
            self._gline(L, (cx + sx * (c + o), h - f - o), (w // 2 + sx * -int(w * 0.18), h - f - o), mid, dark, 1)
        cv2.line(L, (f, h - f - c), (f, int(h * 0.62)), mid, 2, AA)              # left edge tick

        small = cv2.resize(L, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 2)
        halo = cv2.resize(small, (w, h))
        self._overlay = cv2.add(cv2.convertScaleAbs(halo, alpha=1.1), L)

        # vignette (faint) times a dark rounded-corner mask
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.sqrt(((xs - w / 2) / (w / 2)) ** 2 + ((ys - h / 2) / (h / 2)) ** 2) / math.sqrt(2)
        vig = 1.0 - 0.38 * np.clip((r - 0.40) / 0.60, 0, 1) ** 2
        inside = np.zeros((h, w), np.uint8)
        e = self.S(6)
        cv2.fillPoly(inside, [rrect(e, e, w - 2 * e, h - 2 * e, self.S(44))], 255)
        inside = cv2.GaussianBlur(inside, (0, 0), max(1.0, self.k * 1.5)).astype(np.float32) / 255
        mult = vig * (0.12 + 0.88 * inside)
        self._mult3 = cv2.merge([(mult * 255).astype(np.uint8)] * 3)
        self._static_key = (w, h)

    def draw(self, frame, hands, lm, controls, progress: float = 0.0) -> None:
        """hands: all detected hands. lm: the control hand (or None). progress: 0..1 circle sweep."""
        now = time.monotonic()
        h, w = frame.shape[:2]
        self.k = min(1.4, max(0.6, h / 1080))
        if self._static_key != (w, h):
            self._build_static(w, h)

        if controls.last_action_time != self._toast_seen:      # a control just fired
            self._toast_seen = controls.last_action_time
            self._toast = (controls.last_action, now)
            self._fired_at = now

        # 0) vignette + screen border
        cv2.multiply(frame, self._mult3, frame, scale=1 / 255)
        cv2.add(frame, self._overlay, frame)

        # 1) rings, tucked behind the hand
        if lm is not None:
            self.ring_glow.begin(frame)
            self._rings(w, h, lm, now, progress)
            inv = 255 - self._hand_mask(w, h, hands)
            self.ring_glow.end(frame, 1.0, cv2.merge([inv, inv, inv]))
        else:
            self._ring_c = None

        # 2) glowing hand skeleton on top
        self.glow.begin(frame)
        for hand in hands:
            pts = [(int(p.x * w), int(p.y * h)) for p in hand]
            for a, b in CONNECTIONS:
                cv2.line(self.glow.layer, pts[a], pts[b], SKEL, 2, AA)
            for p in pts:
                cv2.circle(self.glow.layer, p, max(3, self.S(4)), WHITE, -1, AA)
        self.glow.end(frame, 1.3)

        # 3) panels on top
        self._brand(frame)
        self._status(frame, w, lm is not None, now)
        self._wave(frame, w, h, now, getattr(controls, "playing", True))
        self._stats(frame, controls, now)
        self._toast_card(frame, w, now)

    # ---------- rings ----------
    def _hand_mask(self, w: int, h: int, hands) -> np.ndarray:
        """Rough hand silhouette (palm + thick finger bones), feathered. Used to hide rings behind the hand."""
        hw, hh = w // 2, h // 2
        m = np.zeros((hh, hw), np.uint8)
        for hand in hands:
            pts = np.array([(p.x * hw, p.y * hh) for p in hand], np.float32)
            scale = float(np.linalg.norm(pts[0] - pts[9]))
            tf = max(4, int(scale * 0.26))
            ip = pts.astype(np.int32)
            palm = ip[[0, 1, 5, 9, 13, 17]]
            cv2.fillPoly(m, [palm], 255)
            cv2.polylines(m, [palm], True, 255, tf)
            for a, b in CONNECTIONS:
                t = int(tf * 1.2) if (a, b) in ((1, 2), (2, 3), (3, 4)) else tf
                cv2.line(m, tuple(ip[a]), tuple(ip[b]), 255, t, AA)
            for i in range(1, 21):
                cv2.circle(m, tuple(ip[i]), tf // 2 + 1, 255, -1, AA)
        m = cv2.GaussianBlur(m, (0, 0), max(1.5, 3 * self.k))
        return cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)

    def _rings(self, w, h, lm, now, progress) -> None:
        wrist = np.array([lm[0].x * w, lm[0].y * h])
        tip = np.array([lm[12].x * w, lm[12].y * h])
        L = float(np.linalg.norm(tip - wrist))
        center = wrist + 0.6 * (tip - wrist)                    # sits over the fingers / upper palm
        if self._ring_c is None:
            self._ring_c, self._ring_L = center, L
        else:                                                   # smooth so the rings don't jitter
            self._ring_c = self._ring_c + (center - self._ring_c) * 0.3
            self._ring_L += (L - self._ring_L) * 0.3
        cx, cy = int(self._ring_c[0]), int(self._ring_c[1])
        L = max(self._ring_L, self.S(220))

        layer = self.ring_glow.layer
        fracs = (0.20, 0.27, 0.35, 0.44, 0.54)
        radii = [max(self.S(60), int(L * f)) for f in fracs]
        for i, r in enumerate(radii):
            cv2.circle(layer, (cx, cy), r, fade(CYAN, 0.18), 1, AA)                    # faint full ring
            start = (now * 28 * (1 if i % 2 == 0 else -1) + i * 63) % 360
            span = 70 + 18 * (i % 3)
            t = 3 if i in (1, 4) else 2
            for off in (0, 180):
                cv2.ellipse(layer, (cx, cy), (r, r), 0, start + off, start + off + span, fade(CYAN, 0.42), t, AA)
        r = radii[3]                                                                   # tick marks on one ring
        rot = now * 12
        for j in range(72):
            a = math.radians(j * 5 + rot)
            ln = 9 if j % 6 == 0 else 4
            p0 = (int(cx + (r + 8) * math.cos(a)), int(cy + (r + 8) * math.sin(a)))
            p1 = (int(cx + (r + 8 + ln) * math.cos(a)), int(cy + (r + 8 + ln) * math.sin(a)))
            cv2.line(layer, p0, p1, fade(CYAN, 0.3), 1, AA)
        if progress > 0.02:                                    # fills as a circle gesture is swept
            R = radii[-1] + self.S(18)
            cv2.ellipse(layer, (cx, cy), (R, R), -90, 0, 360 * min(progress, 1.0), fade(WHITE, 0.65), 3, AA)
        age = now - self._fired_at
        if age < 0.6:                                          # pulse when something fires
            pr = int(radii[0] + ease_out(age / 0.6) * (radii[-1] - radii[0] + self.S(60)))
            cv2.circle(layer, (cx, cy), pr, fade(CYAN, 0.8 * (1 - age / 0.6)), 3, AA)

    def _faded(self, frame, rect, a: float, draw) -> None:
        """Run draw(), then blend only the area around rect back toward the old pixels by (1 - a)."""
        x, y, w, h = rect
        pad = self.S(20)
        H, W = frame.shape[:2]
        x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)
        if x1 <= x0 or y1 <= y0:
            return
        before = frame[y0:y1, x0:x1].copy() if a < 1.0 else None
        draw()
        if before is not None:
            roi = frame[y0:y1, x0:x1]
            cv2.addWeighted(roi, a, before, 1 - a, 0, roi)

    # ---------- panels ----------
    def _brand(self, frame) -> None:
        m, bw, bh = self.S(MARGIN), self.S(380), self.S(84)
        c, d = self.S(24), self.S(56)
        tab = np.array([(m + c, m), (m + bw, m), (m + bw - d, m + bh), (m, m + bh), (m, m + c)], np.int32)
        panel(frame, tab, alpha=0.45)
        isz = self.S(62)
        icon(frame, "car", m + self.S(26), m + (bh - isz) // 2, isz)
        text_mid(frame, "DRIVE.AI", m + self.S(108), m + bh // 2, self.S(36))

    def _status(self, frame, w, tracking: bool, now) -> None:
        pulse = 0.6 + 0.4 * math.sin(now * 4)
        color = GREEN if tracking else AMBER
        label = "System Active" if tracking else "Show hand"
        size = self.S(30)
        m = self.S(MARGIN)
        x = w - m - text_width(label, size)
        cy = m + self.S(30)
        cv2.circle(frame, (x - self.S(24), cy), self.S(9), fade(color, pulse), -1, AA)
        text_mid(frame, label, x, cy, size, color)

    def _wave(self, frame, w, h, now, playing: bool) -> None:
        """Little equalizer at the bottom left (moves while music is playing)."""
        f = self.S(FRAME_INSET)
        x0 = f + self.S(96)
        base = h - f - self.S(12)
        amp = 1.0 if playing else 0.12
        for i in range(30):
            v = (0.5 + 0.5 * math.sin(now * 5 + i * 0.9)) * (0.5 + 0.5 * math.sin(now * 1.7 + i * 1.9))
            top = base - int(self.S(4) + self.S(28) * v * amp)
            x = x0 + i * self.S(7)
            cv2.line(frame, (x, base), (x, top), fade(CYAN, 0.75), 2, AA)

    def _card(self, frame, x, y, cw, ch, ic, label, value) -> None:
        panel(frame, rrect(x, y, cw, ch, self.S(18)), alpha=0.6, edge=fade(CYAN, 0.45), glow=False, thick=1)
        isz = self.S(66)
        icon(frame, ic, x + self.S(26), y + (ch - isz) // 2, isz)
        tx = x + self.S(116)
        text_mid(frame, label, tx, y + int(ch * 0.31), self.S(24), CYAN)
        text_mid(frame, value, tx, y + int(ch * 0.69), self.S(42))

    def _stats(self, frame, controls, now) -> None:
        """Volume, Temp and Music cards are always on. The AC card pops in under them when AC changes."""
        m = self.S(MARGIN)
        cw, ch, gap = self.S(310), self.S(108), self.S(12)
        top = m + self.S(84) + self.S(24)
        rows = [
            ("volume", "Volume", f"{controls.volume}%"),
            ("temp", "Temp", f"{controls.temp}°F"),
            ("music", "Music", "Playing" if getattr(controls, "playing", True) else "Paused"),
        ]
        for i, (ic, label, value) in enumerate(rows):
            self._card(frame, m, top + i * (ch + gap), cw, ch, ic, label, value)

        if self._ac_prev is not None and controls.ac != self._ac_prev:
            self._ac_changed = now
        self._ac_prev = controls.ac
        age = now - self._ac_changed
        if age < self.AC_S:
            a_in = ease_out(age / 0.25)
            a = min(a_in, min(1.0, (self.AC_S - age) / 0.4))
            x = m - int((1 - a_in) * self.S(50))
            y = top + 3 * (ch + gap)
            self._faded(frame, (x, y, cw, ch), a,
                        lambda: self._card(frame, x, y, cw, ch, "ac", "AC", f"Level {controls.ac}"))

    def _toast_card(self, frame, w, now) -> None:
        if self._toast is None:
            return
        title, t0 = self._toast
        age = now - t0
        if age > self.TOAST_S:
            self._toast = None
            return
        slide = int((1 - ease_out(age / 0.25)) * self.S(90))  # slides in from the right
        a = min(1.0, (self.TOAST_S - age) / 0.3)              # fades out over the last 0.3s
        m = self.S(MARGIN)
        bw, bh = self.S(420), self.S(122)
        x, y = w - m - bw + slide, m + self.S(84)

        def draw():
            panel(frame, chamfer(x, y, bw, bh, self.S(24)))
            isz = self.S(70)
            icon(frame, _toast_icon(title), x + self.S(30), y + (bh - isz) // 2, isz)
            tsize = self.S(42)
            room = bw - self.S(132) - self.S(24)
            while tsize > self.S(22) and text_width(title, tsize) > room:   # shrink long titles to fit
                tsize -= 2
            text_mid(frame, title, x + self.S(132), y + self.S(44), tsize)
            text_mid(frame, "Gesture Recognized!", x + self.S(132), y + self.S(88), self.S(24), CYAN)

        self._faded(frame, (x, y, bw, bh), a, draw)
