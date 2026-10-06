"""
Author: Anson
Created on: 6/10/2026
Purpose: address N buttons with two motor-imagery classes.

Both classes are selectors. The candidate set is halved at every decision, so
each decision carries one bit, in the manner of a binary search. This is the
scheme measured in tests/exp_selection_scheme.py:

    5 buttons   scan: 3.0 decisions, 29.0% first-attempt, 46.9 s per selection
                tree: 2.4 decisions, 31.9% first-attempt, 34.1 s per selection
    (at a per-decision accuracy of 0.614)

The tree wins because reliability compounds as P^n, so removing a decision is
worth more than improving the classifier. For the same reason the useful design
lever is keeping the button count small: at 0.614 per decision, 2 buttons
resolve in 7.4 s, 3 in 24 s and 5 in 59 s.
"""

LEFT = "LEFT"
RIGHT = "RIGHT"


class SelectionTree:
    """Balanced binary tree over a flat list of buttons.

        tree = SelectionTree(buttons)
        tree.decide(LEFT)      # keep the first half
        tree.decide(RIGHT)     # keep the second half of that
        # ... when one candidate remains it is selected and fired

    The tree never needs an explicit 'confirm' command, which is what makes it
    cheaper than scanning: in scanning one of the two classes is spent moving a
    highlight and carries no information about *which* button is wanted.
    """

    @staticmethod
    def _screen_x(b):
        """Best-effort horizontal position of a button, or None."""
        r = getattr(b, "rect", None)
        if r is not None:
            for attr in ("centerx", "x", "left"):
                v = getattr(r, attr, None)
                if isinstance(v, (int, float)):
                    return float(v)
        for attr in ("centerx", "x"):
            v = getattr(b, attr, None)
            if isinstance(v, (int, float)):
                return float(v)
        pos = getattr(b, "pos", None)
        if isinstance(pos, (tuple, list)) and pos:
            return float(pos[0])
        return None

    def __init__(self, buttons, on_select=None, on_change=None,
                 auto_fire=True, spatial=True):
        """
        :param buttons: flat sequence; anything with .trigger(), .select() or
                        __call__ can be fired, or plain values if you only want
                        the on_select callback.
        :param on_select: called with the chosen button when a leaf is reached
        :param on_change: called with the current candidate list after a decision
        :param auto_fire: also invoke the button's own trigger/select callback
        :param spatial: split by SCREEN POSITION rather than list order.

            This matters more than it looks. The tree halves a candidate set,
            and the user is told "imagine your LEFT hand to keep the left
            half". If the halves are taken from list order while the buttons
            are drawn in some other order, imagining LEFT can select a button
            on the RIGHT of the screen - the instruction and the display
            disagree, and the user has no way to tell which one to trust.

            With spatial=True the candidates are ordered by on-screen x, so
            LEFT always means the left-hand group of what is currently
            highlighted, whatever order the list happens to be in. Falls back
            to list order for buttons with no geometry.
        """
        self.buttons = list(buttons)
        if len(self.buttons) < 2:
            raise ValueError("a selection tree needs at least 2 buttons")
        self.on_select = on_select
        self.on_change = on_change
        self.auto_fire = auto_fire
        self._path = []                 # decisions taken since the last reset
        xs = [self._screen_x(b) for b in self.buttons]
        if spatial and all(x is not None for x in xs):
            self._order = sorted(range(len(self.buttons)), key=lambda i: xs[i])
            self.spatial = True
        else:
            self._order = list(range(len(self.buttons)))
            self.spatial = False
        self.candidates = list(self._order)
        self.stats = {"decisions": 0, "selections": 0, "resets": 0}

    # ------------------------------------------------------------- geometry
    @staticmethod
    def _split(indices):
        """Halve a candidate list. The larger half goes left when odd, which
        keeps the maximum depth at ceil(log2(n))."""
        mid = (len(indices) + 1) // 2
        return indices[:mid], indices[mid:]

    def depth_of(self, index):
        """How many decisions are needed to reach this button."""
        cand, d = list(self._order), 0
        while len(cand) > 1:
            lo, hi = self._split(cand)
            cand = lo if index in lo else hi
            d += 1
        return d

    def depths(self):
        return [self.depth_of(i) for i in range(len(self.buttons))]

    def mean_depth(self):
        ds = self.depths()
        return sum(ds) / len(ds)

    def max_depth(self):
        return max(self.depths())

    # -------------------------------------------------------------- control
    @property
    def path(self):
        return list(self._path)

    @property
    def current(self):
        """The chosen button once resolved, else None."""
        if len(self.candidates) == 1:
            return self.buttons[self.candidates[0]]
        return None

    def reset(self):
        self._path.clear()
        self.candidates = list(self._order)
        self.stats["resets"] += 1
        self._notify()
        return self.candidates

    def undo(self):
        """Step back one decision. Gives the user a way out of a misread."""
        if not self._path:
            return self.candidates
        path = self._path[:-1]
        self._path.clear()
        self.candidates = list(self._order)
        for cmd in path:
            self._descend(cmd)
        self._notify()
        return self.candidates

    def _descend(self, command):
        lo, hi = self._split(self.candidates)
        self.candidates = lo if command == LEFT else hi
        self._path.append(command)

    def decide(self, command):
        """Apply one classifier decision.

        :param command: LEFT or RIGHT
        :return: the selected button if this decision resolved the tree, else None
        """
        if command not in (LEFT, RIGHT):
            raise ValueError(f"expected {LEFT!r} or {RIGHT!r}, got {command!r}")
        if len(self.candidates) <= 1:
            self.reset()

        self._descend(command)
        self.stats["decisions"] += 1
        self._notify()

        if len(self.candidates) == 1:
            return self._fire(self.buttons[self.candidates[0]])
        return None

    # ------------------------------------------------------------- dispatch
    def _fire(self, button):
        self.stats["selections"] += 1
        if self.auto_fire:
            for attr in ("trigger", "select", "on_click"):
                fn = getattr(button, attr, None)
                if callable(fn):
                    fn()
                    break
            else:
                if callable(button):
                    button()
        if self.on_select:
            self.on_select(button)
        # ready for the next selection
        self._path.clear()
        self.candidates = list(self._order)
        self._notify()
        return button

    def _notify(self):
        if self.on_change:
            self.on_change([self.buttons[i] for i in self.candidates])

    # ----------------------------------------------------------------- misc
    def candidate_buttons(self):
        return [self.buttons[i] for i in self.candidates]

    def __len__(self):
        return len(self.buttons)

    def __repr__(self):
        return (f"<SelectionTree {len(self.buttons)} buttons, "
                f"{'spatial' if self.spatial else 'list'} order, "
                f"max {self.max_depth()} decisions, "
                f"mean {self.mean_depth():.1f}, "
                f"{len(self.candidates)} candidate(s) now>")
