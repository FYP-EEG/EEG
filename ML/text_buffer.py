"""
Author: Brian
Created on: 16/8/2026
Purpose: Backend text buffer for the BCI Speller (Gap 1, item 2).

Deliberately pygame-free so it can be unit-tested headlessly and reused by any
front-end the SDK later exposes. Accumulates individually selected letters and
phrases into coherent sentences, and owns all editing/formatting rules.
"""

from collections import deque


class TextBuffer:
    """Accumulates BCI selections into a sentence."""

    #: control tokens a tile may emit as its `value`
    SPACE = "__SPACE__"
    BACKSPACE = "__BACKSPACE__"
    CLEAR = "__CLEAR__"
    SPEAK = "__SPEAK__"
    UNDO = "__UNDO__"

    def __init__(self, max_history=64, on_change=None, on_speak=None):
        """
        :param on_change: callback(text) fired whenever the buffer content changes
        :param on_speak:  callback(text) fired when the SPEAK action is selected
        """
        self._chars = []
        self._history = deque(maxlen=max_history)   # snapshots for UNDO
        self._log = []                              # every accepted selection
        self.on_change = on_change
        self.on_speak = on_speak

    # -------------------------------------------------------------- properties
    @property
    def text(self):
        return "".join(self._chars)

    @property
    def selection_log(self):
        return list(self._log)

    def __len__(self):
        return len(self._chars)

    def __str__(self):
        return self.text

    # ----------------------------------------------------------------- helpers
    def _snapshot(self):
        self._history.append(list(self._chars))

    def _changed(self):
        if self.on_change:
            self.on_change(self.text)

    # ------------------------------------------------------------------- input
    def accept(self, value):
        """
        Feed one confirmed selection. `value` is a tile's `.value`: either literal
        text ("A", "Yes, please. ") or one of the control tokens above.
        Returns the resulting text.
        """
        if value is None:
            return self.text

        self._log.append(value)

        if value == TextBuffer.BACKSPACE:
            self._snapshot()
            if self._chars:
                self._chars.pop()
        elif value == TextBuffer.SPACE:
            self._snapshot()
            self._append_text(" ")
        elif value == TextBuffer.CLEAR:
            self._snapshot()
            self._chars = []
        elif value == TextBuffer.UNDO:
            self.undo()
            return self.text
        elif value == TextBuffer.SPEAK:
            if self.on_speak:
                self.on_speak(self.text)
            return self.text
        else:
            self._snapshot()
            self._append_text(value)

        self._changed()
        return self.text

    def _append_text(self, s):
        """Append with light sentence hygiene: no double spaces, capitalise starts."""
        if s == " " and (not self._chars or self._chars[-1] == " "):
            return                                    # collapse repeated spaces
        if len(s) == 1 and s.isalpha():
            at_start = (not self._chars) or "".join(self._chars).rstrip().endswith((".", "?", "!"))
            s = s.upper() if at_start else s.lower()
        self._chars.extend(s)

    # ------------------------------------------------------------------ edits
    def undo(self):
        if self._history:
            self._chars = self._history.pop()
            self._changed()
        return self.text

    def clear(self):
        self._snapshot()
        self._chars = []
        self._changed()
        return self.text

    # ---------------------------------------------------------------- display
    def wrapped(self, chars_per_line=38, max_lines=4, cursor="_"):
        """Word-wrapped view for the output panel; returns the last `max_lines`."""
        text = self.text + cursor
        lines, line = [], ""
        for word in text.split(" "):
            candidate = word if not line else line + " " + word
            while len(candidate) > chars_per_line:
                lines.append(candidate[:chars_per_line])
                candidate = candidate[chars_per_line:]
            line = candidate
        lines.append(line)
        return lines[-max_lines:] if len(lines) > max_lines else lines
