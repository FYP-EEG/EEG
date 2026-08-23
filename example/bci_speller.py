"""
Author: Brian
Created on: 16/8/2026
Purpose: BCI Speller -- EEG-to-Text virtual keyboard (instructions.md Gap 1).

Implements the three Gap 1 items:
  1. Virtual keyboard grid  -> 6x7 grid of TileButtons: A-Z, 0-9, phrase tiles, actions.
  2. Text buffer logic      -> ML.text_buffer.TextBuffer (headless, unit-tested).
  3. Output panel           -> live word-wrapped display of the accumulated sentence.

SELECTION PARADIGM
------------------
The classifier currently exposes only TWO reliable commands (SSVEP 10 Hz / 12 Hz),
which cannot address 42 tiles directly. This app therefore uses classic
row-column scanning, the standard 2-command speller paradigm:

    NEXT    (15 Hz / LEFT arrow) : advance the highlight
    SELECT  (20 Hz / RIGHT arrow): commit the highlighted row, then the tile

Targets are 15/20 Hz, not 10/12: 10 Hz sat inside the occipital alpha band and
caused the prediction bias measured at 73.10%. Both are exact 60 Hz divisors.

Stage ROW: highlight sweeps rows.  SELECT enters that row.
Stage COL: highlight sweeps tiles in the row. SELECT commits the tile.
An "escape" cell at the end of each row returns to row-scanning, so a mis-select
is always recoverable without hardware input.

INPUT SOURCES
-------------
--source keyboard : arrow keys / mouse click  (default -- no hardware needed)
--source sim      : synthetic EEG through the real StreamBridge + BCIEngine
--source replay   : a Step 3 .npz recording replayed through the live path
--source engine   : live OpenBCI Cyton via BrainFlow

sim/replay/engine all share ONE code path (EEG.stream_bridge.StreamBridge), so the
only difference is which backend supplies samples.

--profile <json>  : load per-subject thresholds from ML/calibration.py (Step 3)

Run:
    python example/bci_speller.py
    python example/bci_speller.py --source sim --dwell 1.2
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pygame
from pygame.locals import *

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pygame_lib"))

from TileButton import TileButton                     # noqa: E402
from stimulus import StimulusEngine, open_display, suggest_frequencies  # noqa: E402
from ML.text_buffer import TextBuffer                 # noqa: E402

# ----------------------------------------------------------------- keyboard map
PHRASES = [
    ("YES", "Yes. "),
    ("NO", "No. "),
    ("HELP", "I need help. "),
    ("PAIN", "I am in pain. "),
    ("WATER", "I would like some water. "),
    ("THANKS", "Thank you. "),
    ("HELLO", "Hello. "),
    ("TIRED", "I am tired. "),
    ("TOILET", "I need the toilet. "),
    ("WAIT", "Please wait. "),
]

ACTIONS = [
    ("SPACE", TextBuffer.SPACE),
    ("DEL", TextBuffer.BACKSPACE),
    ("UNDO", TextBuffer.UNDO),
    (".", "."),
    ("?", "?"),
    ("CLEAR", TextBuffer.CLEAR),
    ("SPEAK", TextBuffer.SPEAK),
]

BG = (22, 24, 32)
PANEL_BG = (30, 33, 44)
ACCENT = (58, 110, 165)


def build_layout():
    """Return a list of rows, each row a list of (label, value, kind)."""
    letters = [(c, c, "char") for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    digits = [(d, d, "char") for d in "0123456789"]
    phrases = [(l, v, "phrase") for l, v in PHRASES]
    actions = [(l, v, "action") for l, v in ACTIONS]

    cells = letters + digits + phrases + actions
    per_row = 7
    return [cells[i:i + per_row] for i in range(0, len(cells), per_row)]


class Speller:
    def __init__(self, source="keyboard", dwell=1.0, scan_period=1.1,
                 size=(1024, 700), profile=None, replay_npz=None,
                 serial_port="COM4", board_id=0, flicker=False,
                 target_freqs=(15.0, 20.0), refresh=None, vsync=True):
        pygame.init()
        pygame.font.init()
        # Step 5: V-synced, blit-compatible display.
        # NOT pygame.OPENGL (instructions.md A3) -- that breaks every blit.
        self.flicker = flicker
        self.win, self.display_info = open_display(
            size, refresh_hint=refresh, resizable=not flicker, vsync=vsync)
        pygame.display.set_caption("BCI Speller - EEG to Text")
        self.clock = pygame.time.Clock()

        self.refresh = self.display_info["refresh"]
        self.stim = StimulusEngine(target_freqs=target_freqs, refresh_hz=self.refresh)
        # The classifier must track what the monitor ACTUALLY renders, not what was
        # requested -- on a 144 Hz panel 15.0 Hz becomes 14.40 Hz, and matching the
        # reference signal to the request would silently cost accuracy.
        self.actual_freqs = tuple(p.actual for p in self.stim.plans)

        self.source = source
        self.dwell = dwell                 # seconds a tile must stay armed to commit
        self.scan_period = scan_period     # auto-advance period in engine/sim mode

        self.buffer = TextBuffer(on_speak=self._on_speak)
        self.spoken = ""
        self.status = "Ready"
        self.engine_state = "-"

        self.rows = build_layout()
        self.tiles = []                    # list[list[TileButton]]
        self.stage = "ROW"
        self.row_idx = 0
        self.col_idx = 0
        self.armed_since = None
        self.last_scan = time.time()

        self.engine = None
        self.bridge = None
        self.sim = None                  # synthetic backend handle, when present
        self.profile = profile
        self.replay_npz = replay_npz
        self.serial_port = serial_port
        self.board_id = board_id
        if source in ("engine", "sim", "replay"):
            self._init_engine(source)

        self._build_tiles()

    # ------------------------------------------------------------------ engine
    def _init_engine(self, source):
        """Wire StreamBridge -> BCIEngine. One path for sim, replay and hardware."""
        try:
            from ML.bci_engine import BCIEngine
            from EEG.stream_bridge import (StreamBridge, SyntheticBackend,
                                           ReplayBackend, BrainFlowBackend)

            if self.profile:
                self.engine = BCIEngine.from_profile(
                    self.profile, mode="SSVEP", target_freqs=self.actual_freqs)
                self.status = f"profile loaded ({Path(self.profile).name})"
            else:
                self.engine = BCIEngine(mode="SSVEP", debounce_window=3,
                                        confidence_threshold=0.15,
                                        montage="cyton8_ssvep",
                                        target_freqs=self.actual_freqs)
                self.status = f"BCIEngine ready ({source}, default threshold)"

            if source == "sim":
                backend = SyntheticBackend(seed=1)
                self.sim = backend
            elif source == "replay":
                import glob as _g
                pat = self.replay_npz or str(ROOT / "dataset" / "calib_*.npz")
                hits = sorted(_g.glob(pat))
                if not hits:
                    raise FileNotFoundError(f"no recording matched {pat}")
                backend = ReplayBackend(hits[-1], loop=True)
                self.status = f"replaying {Path(hits[-1]).name}"
            else:
                backend = BrainFlowBackend(board_id=self.board_id,
                                           serial_port=self.serial_port)

            self.bridge = StreamBridge(backend=backend, window=750, hop=250,
                                       max_pending=2)
            self.bridge.start()
        except Exception as exc:
            self.engine = None
            self.bridge = None
            self.status = f"Engine unavailable: {exc} - using keyboard"
            self.source = "keyboard"

    # ------------------------------------------------------------------- build
    def _build_tiles(self):
        w, h = self.win.get_size()
        panel_h = int(h * 0.26)
        pad = max(6, int(w * 0.008))
        grid_top = panel_h + pad
        grid_h = h - grid_top - pad
        n_rows = len(self.rows)
        row_h = grid_h / n_rows

        self.tiles = []
        tid = 0
        for r, row in enumerate(self.rows):
            n_cols = len(row)
            col_w = (w - 2 * pad) / n_cols
            tile_row = []
            for c, (label, value, kind) in enumerate(row):
                rect = pygame.Rect(
                    int(pad + c * col_w) + 2,
                    int(grid_top + r * row_h) + 2,
                    int(col_w) - 4,
                    int(row_h) - 4,
                )
                tile_row.append(TileButton(tid, label, value, rect, kind=kind))
                tid += 1
            self.tiles.append(tile_row)

    # -------------------------------------------------------------- navigation
    def _current_row(self):
        return self.tiles[self.row_idx % len(self.tiles)]

    def cmd_next(self):
        """NEXT command: advance the scan highlight."""
        self.armed_since = None
        if self.stage == "ROW":
            self.row_idx = (self.row_idx + 1) % len(self.tiles)
        else:
            row = self._current_row()
            self.col_idx += 1
            if self.col_idx > len(row):        # one past the end == escape cell
                self.col_idx = 0
        self.last_scan = time.time()

    def cmd_select(self):
        """SELECT command: descend a stage, or commit the tile."""
        if self.stage == "ROW":
            self.stage = "COL"
            self.col_idx = 0
            self.status = f"Row {self.row_idx + 1} - scanning tiles"
        else:
            row = self._current_row()
            if self.col_idx >= len(row):       # escape cell
                self.stage = "ROW"
                self.status = "Back to row scan"
            else:
                self._commit(row[self.col_idx])
        self.armed_since = None
        self.last_scan = time.time()

    def _commit(self, tile):
        tile.state = TileButton.SELECTED
        self.buffer.accept(tile.value)
        self.status = f"Selected: {tile.label}"
        self.stage = "ROW"
        pygame.time.set_timer(pygame.USEREVENT + 1, 180, loops=1)

    def _on_speak(self, text):
        self.spoken = text
        self.status = f'SPEAK -> "{text.strip()}"'

    # ------------------------------------------------------------------ states
    def _apply_flicker(self):
        """Drive tile flicker from the frame counter (never from wall-clock time)."""
        if not self.flicker:
            return
        on0, on1 = self.stim.states()[0], self.stim.states()[1]
        for r, row in enumerate(self.tiles):
            for c, tile in enumerate(row):
                if self.stage == "ROW":
                    # rows alternate the two frequencies so NEXT/SELECT map to gaze
                    tile.flicker_on = on0 if (r % 2 == 0) else on1
                    tile.freq = self.actual_freqs[0] if (r % 2 == 0) else self.actual_freqs[1]
                else:
                    if r == self.row_idx:
                        tile.flicker_on = on0 if (c % 2 == 0) else on1
                        tile.freq = self.actual_freqs[0] if (c % 2 == 0) else self.actual_freqs[1]
                    else:
                        tile.flicker_on = True

    def _refresh_states(self):
        for r, row in enumerate(self.tiles):
            for c, tile in enumerate(row):
                if tile.state == TileButton.SELECTED:
                    continue
                if self.stage == "ROW":
                    tile.state = TileButton.SCAN if r == self.row_idx else TileButton.IDLE
                else:
                    if r == self.row_idx and c == self.col_idx:
                        tile.state = (TileButton.ARMED if self.armed_since
                                      else TileButton.SCAN)
                    else:
                        tile.state = TileButton.IDLE

    def _dwell_progress(self):
        if not self.armed_since:
            return 0.0
        return (time.time() - self.armed_since) / self.dwell

    # ------------------------------------------------------------------ render
    def _draw_panel(self):
        w, h = self.win.get_size()
        panel_h = int(h * 0.26)
        panel = pygame.Rect(8, 8, w - 16, panel_h - 12)
        pygame.draw.rect(self.win, PANEL_BG, panel, border_radius=12)
        pygame.draw.rect(self.win, ACCENT, panel, width=2, border_radius=12)

        title_f = pygame.font.SysFont("dejavusans,arial", 15, bold=True)
        text_f = pygame.font.SysFont("dejavusansmono,couriernew,monospace",
                                     max(16, int(panel_h * 0.19)), bold=True)
        small_f = pygame.font.SysFont("dejavusans,arial", 13)

        self.win.blit(title_f.render("OUTPUT", True, (140, 170, 210)), (22, 18))

        chars_per_line = max(20, int((w - 60) / (text_f.size("M")[0] or 12)))
        y = 40
        for line in self.buffer.wrapped(chars_per_line=chars_per_line, max_lines=3):
            self.win.blit(text_f.render(line, True, (240, 244, 250)), (22, y))
            y += text_f.get_height() + 2

        if self.flicker:
            r = self.stim.timing_report()
            fps = f"{r['measured_fps']:.0f}fps" if r else "--"
            jit = f"jit {r['jitter_pct']:.0f}%" if r else ""
            self.flicker_info = (f"{self.actual_freqs[0]:.1f}/"
                                 f"{self.actual_freqs[1]:.1f}Hz {fps} {jit}")
        else:
            self.flicker_info = "flicker off"
        mode = {"ROW": "SCANNING ROWS", "COL": "SCANNING TILES"}[self.stage]
        info = (f"{mode}  |  src={self.source}  |  chars={len(self.buffer)}  "
                f"|  {self.flicker_info}  |  {self.status}")
        self.win.blit(small_f.render(info, True, (150, 158, 176)),
                      (22, panel.bottom - 40))
        keys = ("LEFT/15Hz = NEXT     RIGHT/20Hz = SELECT     mouse = direct     "
                f"ESC = quit     engine: {self.engine_state}")
        self.win.blit(small_f.render(keys, True, (110, 118, 136)),
                      (22, panel.bottom - 22))

    def draw(self):
        self.win.fill(BG)
        self._draw_panel()
        prog = self._dwell_progress()
        for r, row in enumerate(self.tiles):
            for c, tile in enumerate(row):
                p = prog if (self.stage == "COL" and r == self.row_idx
                             and c == self.col_idx) else 0.0
                tile.draw(self.win, progress=p)
            if self.stage == "COL" and r == self.row_idx and self.col_idx >= len(row):
                last = row[-1]
                pygame.draw.rect(self.win, (196, 132, 32),
                                 last.rect.inflate(6, 6), width=3, border_radius=10)
        pygame.display.flip()
        self.stim.tick()

    # -------------------------------------------------------------- engine tick
    def _engine_tick(self):
        """Poll the bridge once per UI frame. Non-blocking: usually returns None."""
        if not self.engine or not self.bridge:
            return
        window = self.bridge.poll()
        if window is None:
            return                      # no new window this frame -- normal
        try:
            decision, triggered = self.engine.process_frame(window)
        except Exception as exc:
            self.status = f"engine error: {exc}"
            return
        # Non-triggering states are informative, not errors: this is the whole
        # point of Gap 4 -- the engine is allowed to say "no command".
        if not triggered:
            self.engine_state = decision
            if decision == "BLINK":
                self.status = "blink -> classifier locked out"
            elif decision == "LOCKOUT":
                self.status = "artifact lockout"
            elif decision == "NO_ACTION":
                self.status = "idle - no command"
            elif decision == "BUILDING":
                self.status = "confirming..."
            return

        self.engine_state = decision
        if decision.endswith("_0"):        # 15 Hz -> NEXT
            self.cmd_next()
        elif decision.endswith("_1"):      # 20 Hz -> SELECT
            if self.armed_since is None:
                self.armed_since = time.time()

    # -------------------------------------------------------------------- loop
    def run(self):
        running = True
        while running:
            for e in pygame.event.get():
                if e.type == QUIT or (e.type == KEYDOWN and e.key == K_ESCAPE):
                    running = False
                elif e.type == VIDEORESIZE:
                    self.win = pygame.display.set_mode(e.size, pygame.RESIZABLE)
                    self._build_tiles()
                elif e.type == pygame.USEREVENT + 1:
                    for row in self.tiles:
                        for t in row:
                            if t.state == TileButton.SELECTED:
                                t.state = TileButton.IDLE
                elif e.type == KEYDOWN:
                    if e.key in (K_LEFT, K_a):
                        self.cmd_next()
                    elif e.key in (K_RIGHT, K_d, K_SPACE, K_RETURN):
                        self.cmd_select()
                    elif e.key == K_BACKSPACE:
                        self.buffer.accept(TextBuffer.BACKSPACE)
                elif e.type == MOUSEBUTTONDOWN and e.button == 1:
                    pos = pygame.mouse.get_pos()
                    for r, row in enumerate(self.tiles):
                        for c, tile in enumerate(row):
                            if tile.check_within(pos):
                                self.row_idx, self.col_idx = r, c
                                self._commit(tile)

            if self.source in ("engine", "sim", "replay"):
                self._engine_tick()
                if (self.stage == "COL" and self.armed_since is None
                        and time.time() - self.last_scan > self.scan_period):
                    self.cmd_next()

            if self.armed_since and self._dwell_progress() >= 1.0:
                self.armed_since = None
                self.cmd_select()

            self._refresh_states()
            self._apply_flicker()
            self.draw()
            # With vsync the flip itself paces us; tick() only caps a runaway loop.
            self.clock.tick(int(self.refresh) + 5)

        if self.bridge:
            self.bridge.stop()
        if self.flicker:
            self.stim.print_report()
        pygame.quit()
        return self.buffer.text


def main():
    ap = argparse.ArgumentParser(description="BCI Speller (EEG-to-Text)")
    ap.add_argument("--source", choices=["keyboard", "sim", "replay", "engine"],
                    default="keyboard")
    ap.add_argument("--profile", default=None,
                    help="per-subject profile json from ML/calibration.py")
    ap.add_argument("--replay-npz", default=None)
    ap.add_argument("--serial-port", default="COM4")
    ap.add_argument("--board-id", type=int, default=0)
    ap.add_argument("--flicker", action="store_true",
                    help="enable V-synced SSVEP flicker (Step 5)")
    ap.add_argument("--no-vsync", action="store_true")
    ap.add_argument("--refresh", type=float, default=None,
                    help="override detected monitor refresh in Hz")
    ap.add_argument("--target-freqs", type=float, nargs=2, default=[15.0, 20.0])
    ap.add_argument("--dwell", type=float, default=1.0)
    ap.add_argument("--scan-period", type=float, default=1.1)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=700)
    args = ap.parse_args()

    app = Speller(source=args.source, dwell=args.dwell,
                  scan_period=args.scan_period, size=(args.width, args.height),
                  profile=args.profile, replay_npz=args.replay_npz,
                  serial_port=args.serial_port, board_id=args.board_id,
                  flicker=args.flicker, target_freqs=tuple(args.target_freqs),
                  refresh=args.refresh, vsync=not args.no_vsync)
    final = app.run()
    print(f"\nFinal text: {final!r}")


if __name__ == "__main__":
    main()
