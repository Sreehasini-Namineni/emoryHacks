"""Hardcoded gesture → action map. Swap this dict later for user-chosen bindings."""

from __future__ import annotations
import time

# gesture id -> label shown in the dropdowns (this order is the dropdown order)
GESTURES = {
    "peace_clockwise": "Peace circle CW",
    "peace_counterclockwise": "Peace circle CCW",
    "three_turn_right": "Three: swipe right",
    "three_turn_left": "Three: swipe left",
    "three_sideways_up": "Three side: up",
    "three_sideways_down": "Three side: down",
    "three_clockwise": "Three circle CW",
    "three_counterclockwise": "Three circle CCW",
    "thumbs_up": "Thumbs up",
    "thumbs_down": "Thumbs down",
    "closed_fist": "Closed fist",
}

class Controls:
    def __init__(self) -> None:
        self.volume = 50
        self.ac = 0
        self.temp = 72
        self.playing = True
        self.last_action = ""
        self.last_action_time = -1e9
        # control id -> (label shown in settings, method to call)
        self.actions = {
            "volume_up": ("Volume up", self.volume_up),
            "volume_down": ("Volume down", self.volume_down),
            "next_track": ("Next track", self.next_track),
            "previous_track": ("Previous track", self.previous_track),
            "ac_up": ("AC up", self.ac_up),
            "ac_down": ("AC down", self.ac_down),
            "temp_up": ("Temp up", self.temp_up),
            "temp_down": ("Temp down", self.temp_down),
            "accept_call": ("Accept call", self.accept_call),
            "decline_call": ("Decline call", self.decline_call),
        }
        # control id -> gesture id (None = off). Edited from the settings popup.
        self.bindings = {
            "volume_up": "peace_clockwise",
            "volume_down": "peace_counterclockwise",
            "next_track": "three_turn_right",
            "previous_track": "three_turn_left",
            "ac_up": "three_sideways_up",
            "ac_down": "three_sideways_down",
            "temp_up": "three_clockwise",
            "temp_down": "three_counterclockwise",
            "accept_call": "thumbs_up",
            "decline_call": "thumbs_down",
        }

    def handle(self, gesture: str) -> None:
        for control, bound in self.bindings.items():
            if bound == gesture:
                self.actions[control][1]()
                return

    def options_for(self, control: str) -> list[tuple[str | None, str]]:
        """Dropdown choices for one control: Off, its own gesture, and every gesture no other control uses."""
        taken = {g for c, g in self.bindings.items() if g and c != control}
        return [(None, "Off")] + [(g, label) for g, label in GESTURES.items() if g not in taken]

    def set_binding(self, control: str, gesture: str | None) -> None:
        if gesture is not None and gesture in {g for c, g in self.bindings.items() if c != control}:
            return                                   # already used by another control
        self.bindings[control] = gesture

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
