"""Hardcoded gesture → action map. Swap this dict later for user-chosen bindings."""

from __future__ import annotations


class Controls:
    def __init__(self) -> None:
        self.volume = 50
        self.mapping = {
            "peace_clockwise": self.volume_up,
            "three_turn_right": self.next_track,
            "three_turn_left": self.previous_track,
        }

    def handle(self, gesture: str) -> None:
        action = self.mapping.get(gesture)
        if action:
            action()

    def volume_up(self) -> None:
        self.volume = min(100, self.volume + 5)
        print(f"Volume increased -> {self.volume}")

    def next_track(self) -> None:
        print("Next track >>")

    def previous_track(self) -> None:
        print("<< Previous track")
