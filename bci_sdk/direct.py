"""
Author: Anson
Created on: 13/10/2026
Purpose: dispatch a direct-button model command without LEFT/RIGHT navigation.

DirectButtonSelector is separate from SelectionTree and ScanSelector so the
existing two-class MI behaviour cannot change accidentally. Its input command
is already a button name (for example ``"paper"``), so one accepted decision
fires exactly that button.
"""


def button_name(button):
    """Public UI name used by recording, training and live dispatch."""
    for attr in ("name", "label"):
        value = getattr(button, attr, None)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ValueError("every direct-MI button needs a non-empty .name or .label")


class DirectButtonSelector:
    """Map a predicted button name directly to the matching UI object."""

    def __init__(self, buttons, on_select=None, on_change=None, auto_fire=True):
        self.buttons = list(buttons)
        if len(self.buttons) < 2:
            raise ValueError("direct button selection needs at least 2 buttons")
        self.on_select = on_select
        self.on_change = on_change
        self.auto_fire = bool(auto_fire)
        self._by_name = {}
        for button in self.buttons:
            name = button_name(button)
            if name in self._by_name:
                raise ValueError(f"direct button names must be unique: {name!r}")
            self._by_name[name] = button
        self.stats = {"decisions": 0, "selections": 0, "resets": 0}

    @property
    def names(self):
        return list(self._by_name)

    @property
    def candidates(self):
        return list(range(len(self.buttons)))

    def candidate_buttons(self):
        # All buttons remain selectable at all times; there is no tree state.
        return list(self.buttons)

    def reset(self):
        self.stats["resets"] += 1
        if self.on_change:
            self.on_change(self.candidate_buttons())
        return self.candidates

    def decide(self, command):
        """Fire the button whose public name exactly matches ``command``."""
        self.stats["decisions"] += 1
        if command not in self._by_name:
            raise ValueError(
                f"model predicted button {command!r}, but this UI has "
                f"{sorted(self._by_name)!r}")
        button = self._by_name[command]
        self.stats["selections"] += 1

        if self.auto_fire:
            for attr in ("trigger", "select", "on_click"):
                callback = getattr(button, attr, None)
                if callable(callback):
                    callback()
                    break
            else:
                if callable(button):
                    button()
        if self.on_select:
            self.on_select(button)
        return button

    def __len__(self):
        return len(self.buttons)

    def __repr__(self):
        return f"<DirectButtonSelector {self.names!r}>"
