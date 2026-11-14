"""Hardcoded gesture → action map. Swap this dict later for user-chosen bindings."""

from __future__ import annotations


class Controls:
    def __init__(self) -> None:
        self.volume = 50
        self.mapping = {
            "peace_clockwise": self.volume_up,
        }

    def handle(self, gesture: str) -> None:
        action = self.mapping.get(gesture)
        if action:
            action()

    def volume_up(self) -> None:
        self.volume = min(100, self.volume + 5)
        print(f"Volume increased -> {self.volume}")
