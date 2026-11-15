"""Hardcoded gesture → action map. Swap this dict later for user-chosen bindings."""

from __future__ import annotations
import time

class Controls:
    def __init__(self) -> None:
        self.volume = 50
        self.ac = 0
        self.temp = 72
        self.playing = True
        self.last_action = ""
        self.last_action_time = -1e9
        self.mapping = {
            "peace_clockwise": self.volume_up,
            "peace_counterclockwise": self.volume_down,
            "three_turn_right": self.next_track,
            "three_turn_left": self.previous_track,
            "three_sideways_up": self.ac_up,
            "three_sideways_down": self.ac_down,
            "three_clockwise": self.temp_up,
            "three_counterclockwise": self.temp_down,
            "thumbs_up": self.accept_call,
            "thumbs_down": self.decline_call,
            "palm_to_fist": self.toggle_play,
        }

    def handle(self, gesture: str) -> None:
        action = self.mapping.get(gesture)
        if action:
            action()

    def _say(self, text: str) -> None:
        """Show text on screen (step3 draws it for 2s) and echo to the console."""
        self.last_action = text
        self.last_action_time = time.monotonic()
        print(text)

    def volume_up(self) -> None:
        self.volume = min(100, self.volume + 5)
        self._say("Volume UP")

    def volume_down(self) -> None:
        self.volume = max(0, self.volume - 5)
        self._say("Volume DOWN")

    def next_track(self) -> None:
        self._say("Next track >>")

    def previous_track(self) -> None:
        self._say("<< Previous track")

    def ac_up(self) -> None:
        self.ac = min(5, self.ac + 1)
        self._say(f"AC increased -> {self.ac}")

    def ac_down(self) -> None:
        self.ac = max(0, self.ac - 1)
        self._say(f"AC decreased -> {self.ac}")

    def temp_up(self) -> None:
        self.temp = min(84, self.temp + 1)
        self._say(f"Temp increased -> {self.temp}")

    def temp_down(self) -> None:
        self.temp = max(60, self.temp - 1)
        self._say(f"Temp decreased -> {self.temp}")

    def accept_call(self) -> None:
        self._say("Call ACCEPTED")

    def decline_call(self) -> None:
        self._say("Call DECLINED")

    def toggle_play(self) -> None:
        self.playing = not self.playing
        self._say("Play >" if self.playing else "Pause ||")
