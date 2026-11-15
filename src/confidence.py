"""Top-N confidence bars drawn from each detector's .score(lm)."""

from __future__ import annotations

import time

import cv2


def all_scores(detectors, lm) -> dict[str, float]:
    """Merge score(lm) from every detector. A new gesture only needs a score() method."""
    scores: dict[str, float] = {}
    for d in detectors:
        scores.update(d.score(lm))
    return scores


class ConfidencePanel:
    """Smoothed top-N bars. The right edge of each bar is the trigger point."""

    def __init__(self, top_n=3, alpha=0.35, show_floor=0.15, flash_s=0.4, stickiness=0.05):
        self.top_n, self.alpha, self.show_floor = top_n, alpha, show_floor
        self.flash_s, self.stickiness = flash_s, stickiness
        self.visible = True                  # standalone bottom-left bars (draw); off when the dev tools window owns them
        self._smooth: dict[str, float] = {}
        self._flash: dict[str, float] = {}
        self._shown: list[str] = []

    def _value(self, name: str, now: float) -> float:
        if now - self._flash.get(name, -1e9) < self.flash_s:
            return 1.0               # just fired: hold the bar full briefly
        return self._smooth.get(name, 0.0)

    def update(self, scores: dict[str, float], fired=()) -> None:
        now = time.monotonic()
        for name, raw in scores.items():
            prev = self._smooth.get(name, 0.0)
            self._smooth[name] = prev + self.alpha * (raw - prev)
        for name in fired:
            self._flash[name] = now
        # Entries already on screen get a small bonus so near-ties don't flicker.
        ranked = sorted(
            self._smooth,
            key=lambda n: self._value(n, now) + (self.stickiness if n in self._shown else 0),
            reverse=True,
        )
        self._shown = [n for n in ranked if self._value(n, now) >= self.show_floor][: self.top_n]

    def _draw_rows(self, frame, x0: int, y0: int, bar_w: int = 220) -> None:
        now = time.monotonic()
        bar_h, gap = 14, 34
        for i, name in enumerate(self._shown):
            v = self._value(name, now)
            y = y0 + i * gap
            by = y + 6
            color = (0, 255, 0) if v >= 1.0 else (0, 255, 255) if v >= 0.5 else (0, 165, 255)
            cv2.putText(frame, name.replace("_", " "), (x0, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x0, by), (x0 + bar_w, by + bar_h), (60, 60, 60), -1)
            cv2.rectangle(frame, (x0, by), (x0 + int(bar_w * v), by + bar_h), color, -1)
            cv2.line(frame, (x0 + bar_w, by - 3), (x0 + bar_w, by + bar_h + 3), (255, 255, 255), 2)

    def draw(self, frame) -> None:
        """Standalone bars in the bottom-left corner."""
        if not self.visible:
            return
        h = frame.shape[0]
        self._draw_rows(frame, 16, h - 16 - 34 * self.top_n)

    def draw_at(self, frame, x0: int, y0: int, bar_w: int = 220) -> None:
        """Bars at a given position (used by the dev tools popup). Ignores .visible."""
        if not self._shown:
            cv2.putText(frame, "no gesture in progress", (x0, y0 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)
            return
        self._draw_rows(frame, x0, y0, bar_w)