"""
Event-driven brain-controlled UI widgets (instructions.md Gap 3, item 2).

    button = bci_sdk.BrainButton("YES", rect=(50, 50, 200, 80))

    @button.on_brain_select
    def chosen(btn):
        print("user selected", btn.label)

Two selection paradigms are supported, because the right one depends on how many
commands your classifier can reliably produce:

  DIRECT   -- each button owns its own flicker frequency; gazing at it selects it.
              Fast, but needs one reliable frequency per button (~4-6 max on a
              60 Hz monitor; see suggest_frequencies()).

  SCANNING -- a highlight sweeps the buttons; one command advances, another selects.
              Only needs TWO commands, so it scales to a full keyboard. This is what
              the speller uses.

`Scanner` implements the second and drives the same callbacks, so application code
does not change when you switch paradigms.
"""

import time

import pygame

from TileButton import TileButton


class BrainButton(TileButton):
    """A TileButton with brain-selection callbacks and dwell confirmation."""

    def __init__(self, label, rect, value=None, id=None, kind="char",
                 freq=None, dwell=1.0, on_select=None, **kw):
        """
        :param dwell: seconds the button must stay armed before committing. Gives
                      the user a visible chance to look away and cancel.
        """
        super().__init__(id=id if id is not None else abs(hash(label)) % 100000,
                         label=label, value=value if value is not None else label,
                         rect=pygame.Rect(rect), kind=kind, freq=freq, **kw)
        self.dwell = dwell
        self._select_cbs = []
        self._enter_cbs = []
        self._leave_cbs = []
        self._armed_since = None
        self.enabled = True
        if on_select:
            self.on_brain_select(on_select)

    # ----------------------------------------------------------- callbacks
    def on_brain_select(self, fn):
        """Register callback(button) fired when this button is selected. Usable
        as a decorator; returns fn so stacking works."""
        self._select_cbs.append(fn)
        return fn

    def on_focus(self, fn):
        """callback(button) when the highlight/gaze arrives."""
        self._enter_cbs.append(fn)
        return fn

    def on_blur(self, fn):
        """callback(button) when the highlight/gaze leaves before committing."""
        self._leave_cbs.append(fn)
        return fn

    # -------------------------------------------------------------- arming
    def arm(self):
        """Begin the dwell countdown."""
        if not self.enabled:
            return
        if self._armed_since is None:
            self._armed_since = time.time()
            self.state = TileButton.ARMED
            self._fire(self._enter_cbs)

    def disarm(self):
        if self._armed_since is not None:
            self._armed_since = None
            self.state = TileButton.IDLE
            self._fire(self._leave_cbs)

    @property
    def is_armed(self):
        return self._armed_since is not None

    @property
    def progress(self):
        """0..1 dwell completion, for the on-tile progress bar."""
        if self._armed_since is None:
            return 0.0
        return min(1.0, (time.time() - self._armed_since) / max(1e-6, self.dwell))

    def update(self):
        """Advance the dwell timer; fires selection when it completes."""
        if self._armed_since is not None and self.progress >= 1.0:
            self._armed_since = None
            self.select()
            return True
        return False

    def select(self):
        """Fire the selection callbacks immediately (also used for mouse clicks)."""
        if not self.enabled:
            return None
        self.state = TileButton.SELECTED
        self._fire(self._select_cbs)
        return self.value

    def _fire(self, cbs):
        for cb in cbs:
            try:
                cb(self)
            except Exception:
                pass

    def draw(self, surface, progress=None):
        super().draw(surface, progress if progress is not None else self.progress)


class ButtonGroup:
    """A collection of BrainButtons with direct frequency-to-button mapping.

    Use when you have one reliable flicker frequency per button.
    """

    def __init__(self, buttons=None, freqs=None):
        self.buttons = list(buttons or [])
        if freqs:
            self.assign_frequencies(freqs)

    def add(self, button):
        self.buttons.append(button)
        return button

    def assign_frequencies(self, freqs):
        """Give each button its own flicker frequency, round-robin."""
        for i, b in enumerate(self.buttons):
            b.freq = freqs[i % len(freqs)]
        return self

    def handle_command(self, command):
        """Route a Command to the button whose index matches. Returns it or None."""
        idx = getattr(command, "index", -1)
        if 0 <= idx < len(self.buttons):
            b = self.buttons[idx]
            b.arm()
            return b
        return None

    def update(self):
        return [b for b in self.buttons if b.update()]

    def draw(self, surface):
        for b in self.buttons:
            b.draw(surface)

    def apply_flicker(self, stim):
        """Drive flicker state from a StimulusEngine (frame-accurate)."""
        for b in self.buttons:
            if b.freq is None:
                continue
            for i, plan in enumerate(stim.plans):
                if abs(plan.actual - b.freq) < 1e-6:
                    b.flicker_on = stim.is_on(i)
                    break

    def at(self, pos):
        for b in self.buttons:
            if b.check_within(pos):
                return b
        return None

    def __len__(self):
        return len(self.buttons)

    def __iter__(self):
        return iter(self.buttons)


class Scanner:
    """Two-command row/column scanning over a grid of BrainButtons.

    Lets a 2-class classifier address an arbitrary number of buttons.

        scanner = Scanner(rows_of_buttons)
        scanner.next()      # advance highlight  (command 0)
        scanner.select()    # descend / commit   (command 1)
    """

    ROW = "ROW"
    COL = "COL"

    def __init__(self, rows, wrap_escape=True):
        """:param rows: list of lists of BrainButton"""
        self.rows = [list(r) for r in rows]
        self.wrap_escape = wrap_escape
        self.stage = Scanner.ROW
        self.row_idx = 0
        self.col_idx = 0
        self._stage_cbs = []

    def on_stage_change(self, fn):
        self._stage_cbs.append(fn)
        return fn

    @property
    def current_row(self):
        return self.rows[self.row_idx % len(self.rows)]

    @property
    def current(self):
        """The button currently under the highlight, or None on the escape cell."""
        if self.stage == Scanner.ROW:
            return None
        row = self.current_row
        return row[self.col_idx] if self.col_idx < len(row) else None

    def next(self):
        """Advance the highlight (command 0)."""
        for b in self._all():
            b.disarm()
        if self.stage == Scanner.ROW:
            self.row_idx = (self.row_idx + 1) % len(self.rows)
        else:
            limit = len(self.current_row) + (1 if self.wrap_escape else 0)
            self.col_idx = (self.col_idx + 1) % limit
        return self

    def select(self):
        """Descend into the row, or commit the highlighted button (command 1)."""
        if self.stage == Scanner.ROW:
            self.stage = Scanner.COL
            self.col_idx = 0
            self._notify()
            return None
        btn = self.current
        if btn is None:                       # escape cell -> back to row scanning
            self.stage = Scanner.ROW
            self._notify()
            return None
        self.stage = Scanner.ROW
        self._notify()
        return btn.select()

    def handle_command(self, command):
        """Map a Command onto next()/select() by its index."""
        idx = getattr(command, "index", -1)
        if idx == 0:
            self.next()
        elif idx == 1:
            self.select()
        return self

    def update_highlight(self):
        """Refresh each button's visual state to match the scan position."""
        for r, row in enumerate(self.rows):
            for c, b in enumerate(row):
                if b.state == TileButton.SELECTED:
                    continue
                if self.stage == Scanner.ROW:
                    b.state = TileButton.SCAN if r == self.row_idx else TileButton.IDLE
                else:
                    if r == self.row_idx and c == self.col_idx:
                        b.state = (TileButton.ARMED if b.is_armed
                                   else TileButton.SCAN)
                    else:
                        b.state = TileButton.IDLE

    def update(self):
        return [b for b in self._all() if b.update()]

    def draw(self, surface):
        for b in self._all():
            b.draw(surface)

    def _all(self):
        return [b for row in self.rows for b in row]

    def _notify(self):
        for cb in self._stage_cbs:
            try:
                cb(self.stage)
            except Exception:
                pass
