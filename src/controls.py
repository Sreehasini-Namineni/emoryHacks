"""Hardcoded gesture → action map. Swap this dict later for user-chosen bindings."""

from __future__ import annotations


class Controls:
    def __init__(self) -> None:
        self.volume = 50
        self.ac = 0
        self.last_action = ""
        self.mapping = {
            "peace_clockwise": self.volume_up,
            "three_turn_right": self.next_track,
            "three_turn_left": self.previous_track,
            "three_sideways_up": self.ac_up,
            "three_sideways_down": self.ac_down,
        }

    def handle(self, gesture: str) -> None:
        action = self.mapping.get(gesture)
        if action:
            action()

    def volume_up(self) -> None:
        self.volume = min(100, self.volume + 5)
        self.last_action = "Volume UP"
    def next_track(self) -> None:
        print("Next track >>")

    def previous_track(self) -> None:
        print("<< Previous track")

    def ac_up(self) -> None:
        self.ac = min(5, self.ac + 1)
        print(f"AC increased -> {self.ac}")

    def ac_down(self) -> None:
        self.ac = max(0, self.ac - 1)
        print(f"AC decreased -> {self.ac}")
 
    def volume_down(self) -> None:
        self.volume = max(0, self.volume - 5)
        self.last_action = "Volume DOWN"
