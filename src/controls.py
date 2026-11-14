"""Hardcoded gesture → action map. Swap this dict later for user-chosen bindings."""

from __future__ import annotations


class Controls:
    def __init__(self) -> None:
        self.volume = 50
        self.last_action = ""
        self.mapping = {
            "peace_clockwise": self.volume_up,
            "peace_counterclockwise": self.volume_down,
        }

    def handle(self, gesture: str) -> None:
        action = self.mapping.get(gesture)
        if action:
            action()

    def volume_up(self) -> None:
        self.volume = min(100, self.volume + 5)
        self.last_action = "Volume UP"

    def volume_down(self) -> None:
        self.volume = max(0, self.volume - 5)
        self.last_action = "Volume DOWN"