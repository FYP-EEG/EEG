"""
Author: Anson Li
Created on: 26/6/2026
Purpose: script for easier data recording
Location: project_dir/EEG/data_record.py

Edited on: 20/9/2026
Changes: added --file mode. Runs a script from example/, cues the user with a
         flashing button, and records EEG for the whole session.

TWO MODES
---------
1. Character mode (original, unchanged) - flashes typed/random characters:
       py EEG/data_record.py

2. Cued app mode (new) - runs a real app and cues button targets:
       py EEG/data_record.py --file rockpaperscissors
       py EEG/data_record.py --file rockpaperscissors --trials 20 --cue 5
       py EEG/data_record.py --file realistic_ui --dry-run     (no EEG needed)
       py EEG/data_record.py --list

WHY CUED RECORDING
------------------
To train a classifier you need to know WHICH target the user was attending to,
and WHEN. Watching someone use an app freely gives you EEG with no labels.

So the recorder drives the session: it picks a target, makes that button ripple,
and marks the EEG stream. Everything between the start and end marker is labelled
attention-on-that-button. The user's click is recorded too, which tells you
whether they actually complied with the cue - trials where they clicked the wrong
button should be dropped during analysis.

HOW IT DRIVES AN UNMODIFIED EXAMPLE SCRIPT
------------------------------------------
Example scripts are top-level code with their own `while run:` loop, so importing
one just runs it. Instead of rewriting every example, this hooks
`pygame.display.flip()` - the one call every frame makes - and executes the script
inside a namespace we own. From the hook we can reach the script's Button objects,
drive the cue, and draw an overlay on top of whatever it rendered.

Zero changes required in the example scripts.
"""

import argparse
import csv
import datetime as dt
import random
import string
import sys
from pathlib import Path

import pygame
#import constants for easier access to key events
from pygame.locals import *

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = ROOT / "example"
sys.path.insert(0, str(ROOT / "pygame_lib"))


def _load_bci():
    """Import data_receive lazily.

    Kept out of module scope so --dry-run works on a machine with no brainflow
    installed, and so `--list` never touches the hardware stack.
    """
    import data_receive as BCI
    return BCI


# =====================================================================
#  CUED APP MODE
# =====================================================================
class CuedRecorder:
    """Drives cue → wait → rest trials over an unmodified example script."""

    CUE = "CUE"
    REST = "REST"
    DONE = "DONE"

    def __init__(self, bci, trials=10, cue_s=5.0, rest_s=2.0, script="app"):
        self.bci = bci                    # None in --dry-run
        self.n_trials = trials
        self.cue_ms = int(cue_s * 1000)
        self.rest_ms = int(rest_s * 1000)
        self.script = script

        self.buttons = []
        self.order = []
        self.trial = 0
        self.phase = self.REST
        self.phase_end = 0
        self.target = None
        self.started = False

        self.clicked = None               # last button clicked, for the overlay
        self.clicked_at = 0
        self.trial_click = None           # click recorded for the current trial
        self.log = []

        self.font_big = None
        self.font = None
        self.font_sm = None

    # ---------------------------------------------------------- discovery
    def find_buttons(self, namespace):
        """Pull Button instances out of the running script's globals."""
        from Button import Button
        found, seen = [], set()
        for value in namespace.values():
            if isinstance(value, Button) and id(value) not in seen:
                seen.add(id(value))
                found.append(value)
        # stable left-to-right order so trial labels mean something
        found.sort(key=lambda b: (b.rect.centerx, b.rect.centery))
        return found

    def _wrap_clicks(self):
        """Record clicks without touching the example script.

        The script calls button.click(); we wrap that method so the click is
        logged and the icon overlay is triggered, then the original runs as
        normal (including any func= the developer supplied).
        """
        for btn in self.buttons:
            if getattr(btn, "_dr_wrapped", False):
                continue
            original = btn.click

            def make(b, orig):
                def wrapped(*a, **k):
                    self.clicked = b
                    self.clicked_at = pygame.time.get_ticks()
                    if self.phase == self.CUE and self.trial_click is None:
                        self.trial_click = b
                    return orig(*a, **k)
                return wrapped

            btn.click = make(btn, original)
            btn._dr_wrapped = True

    # ------------------------------------------------------------ markers
    def _marker(self, name, phase):
        if self.bci is None:
            return
        try:
            self.bci.put_marker(name, phase=phase)
        except Exception as exc:
            print(f"  [marker failed: {exc}]")

    # ------------------------------------------------------- trial engine
    def _build_order(self):
        """Balanced, shuffled target sequence - every button appears equally."""
        order = []
        while len(order) < self.n_trials:
            batch = list(self.buttons)
            random.shuffle(batch)
            order.extend(batch)
        return order[:self.n_trials]

    def _start_cue(self, now):
        self.trial += 1
        self.target = self.order[self.trial - 1]
        self.trial_click = None
        self.phase = self.CUE
        self.phase_end = now + self.cue_ms
        self.cue_start = now

        for b in self.buttons:
            b.stop_flash()
        # loop=True so the ripple runs for the whole cue, however long it is
        self.target.flash(loop=True, waves=2, pulse_hz=0.5)

        self._marker(self.target.name, "start")
        print(f"  trial {self.trial}/{self.n_trials}: LOOK AT '{self.target.name}'")

    def _end_cue(self, now):
        self._marker(self.target.name, "end")
        self.target.stop_flash()

        hit = self.trial_click
        ok = hit is self.target
        verdict = "correct" if ok else (f"clicked {hit.name}" if hit else "no click")
        print(f"           -> {verdict}")

        self.log.append({
            "trial": self.trial,
            "target": self.target.name,
            "target_id": self.target.id,
            "cue_start_ms": self.cue_start,
            "cue_end_ms": now,
            "clicked": hit.name if hit else "",
            "correct": int(ok),
        })

        self.phase = self.REST
        self.phase_end = now + self.rest_ms

    def update(self, now):
        if self.phase == self.DONE:
            return

        if not self.started:
            self.started = True
            self.order = self._build_order()
            self.phase = self.REST
            self.phase_end = now + self.rest_ms
            return

        if self.phase == self.CUE:
            # self-heal: the example script may restart its own non-looping
            # flash on click, which would end our cue ripple early
            if not self.target.is_flashing:
                self.target.flash(loop=True, waves=2, pulse_hz=0.5)
            if now >= self.phase_end:
                self._end_cue(now)

        elif self.phase == self.REST and now >= self.phase_end:
            if self.trial >= self.n_trials:
                self.phase = self.DONE
                print("\n  all trials complete - close the window to finish")
            else:
                self._start_cue(now)

    # --------------------------------------------------------- rendering
    def _fonts(self):
        if self.font is None:
            self.font_big = pygame.font.SysFont("dejavusans,arial", 30, bold=True)
            self.font = pygame.font.SysFont("dejavusans,arial", 19, bold=True)
            self.font_sm = pygame.font.SysFont("dejavusans,arial", 14)

    def draw(self, win):
        """Overlay drawn on top of whatever the example rendered this frame."""
        self._fonts()
        w, h = win.get_size()

        # --- clicked icon at screen centre -----------------------------
        # Shown for 1.2 s after any click, so the user gets confirmation that
        # the selection registered.
        if self.clicked is not None:
            age = pygame.time.get_ticks() - self.clicked_at
            if age < 1200:
                side = int(min(w, h) * 0.22)
                icon = pygame.transform.smoothscale(self.clicked.img, (side, side))
                if age > 900:                       # fade the last 300 ms
                    icon = icon.copy()
                    icon.set_alpha(int(255 * (1200 - age) / 300))
                win.blit(icon, icon.get_rect(center=(w // 2, h // 2 - 30)))
                cap = self.font.render(self.clicked.name.upper(), True, (250, 214, 110))
                win.blit(cap, cap.get_rect(center=(w // 2, h // 2 + side // 2 - 10)))
            else:
                self.clicked = None

        # --- status bar -------------------------------------------------
        bar = pygame.Surface((w, 58), pygame.SRCALPHA)
        bar.fill((0, 0, 0, 170))
        win.blit(bar, (0, 0))

        if self.phase == self.CUE:
            msg, col = f"LOOK AT:  {self.target.name.upper()}", (250, 214, 110)
        elif self.phase == self.DONE:
            msg, col = "COMPLETE - close the window", (120, 220, 160)
        else:
            msg, col = "rest...", (170, 178, 198)
        win.blit(self.font_big.render(msg, True, col), (18, 10))

        right = f"trial {min(self.trial, self.n_trials)}/{self.n_trials}"
        r = self.font.render(right, True, (220, 226, 240))
        win.blit(r, r.get_rect(topright=(w - 18, 16)))

        if self.phase in (self.CUE, self.REST) and self.started:
            left = max(0, (self.phase_end - pygame.time.get_ticks()) / 1000.0)
            t = self.font_sm.render(f"{left:4.1f}s", True, (150, 158, 180))
            win.blit(t, t.get_rect(topright=(w - 18, 38)))

        rec = "REC" if self.bci is not None else "DRY-RUN (no EEG)"
        reccol = (220, 90, 90) if self.bci is not None else (140, 146, 164)
        win.blit(self.font_sm.render(rec, True, reccol), (18, 40))

        # --- progress bar ----------------------------------------------
        if self.started and self.phase != self.DONE:
            total = self.cue_ms if self.phase == self.CUE else self.rest_ms
            left_ms = max(0, self.phase_end - pygame.time.get_ticks())
            frac = 1.0 - (left_ms / total if total else 1)
            pygame.draw.rect(win, (40, 44, 58), pygame.Rect(0, 56, w, 4))
            pygame.draw.rect(win, col, pygame.Rect(0, 56, int(w * frac), 4))

    # ------------------------------------------------------------- output
    def save_log(self, out_dir=None):
        if not self.log:
            return None
        out_dir = Path(out_dir or (ROOT / "dataset"))
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"cue_log_{self.script}_{stamp}.csv"
        with open(path, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(self.log[0].keys()))
            wr.writeheader()
            wr.writerows(self.log)
        return path


def run_example(args):
    """Execute example/<name>.py with the cue recorder attached."""
    name = args.file
    if not name.endswith(".py"):
        name += ".py"
    script = EXAMPLE_DIR / name
    if not script.exists():
        print(f"No such script: {script}")
        print(f"Available: {', '.join(sorted(p.stem for p in EXAMPLE_DIR.glob('*.py') if p.stem != '__init__'))}")
        return 1

    BCI = None
    if not args.dry_run:
        BCI = _load_bci()
        BCI.start(BID=args.board_id, port=args.serial_port, plot_domain="f")
        print(f"Recording started (board_id={args.board_id})")

    rec = CuedRecorder(BCI, trials=args.trials, cue_s=args.cue,
                       rest_s=args.rest, script=script.stem)

    print(f"\nRunning {script.name}")
    print(f"  {args.trials} trials · {args.cue}s cue · {args.rest}s rest")
    print("  the flashing button is your target - try to select it\n")

    namespace = {"__name__": "__main__", "__file__": str(script)}
    original_flip = pygame.display.flip

    def hooked_flip(*a, **k):
        """Runs once per frame, after the script has drawn, before presenting."""
        win = pygame.display.get_surface()
        if win is not None:
            if not rec.buttons:
                rec.buttons = rec.find_buttons(namespace)
                if rec.buttons:
                    rec._wrap_clicks()
                    print(f"  found {len(rec.buttons)} buttons: "
                          f"{', '.join(b.name for b in rec.buttons)}\n")
            if rec.buttons:
                rec.update(pygame.time.get_ticks())
                rec.draw(win)
        return original_flip(*a, **k)

    """
    Added by Anson
    Date: 20/9/2026
    Purpose: run from whichever directory actually contains assets/.

    The examples load images with the relative path "assets/", so the correct
    working directory depends on where assets/ lives - NOT on where the script
    lives. Running `py example/rockpaperscissors.py` from the project root means
    assets/ is expected at the root, so chdir-ing into example/ broke it.

    Look for assets/ in the likely places and use the first hit. Falling back to
    the current directory keeps the behaviour identical to launching the script
    by hand.
    """
    import os
    prev_cwd = os.getcwd()
    for candidate in (Path(prev_cwd), ROOT, EXAMPLE_DIR):
        if (candidate / "assets").is_dir():
            run_dir = candidate
            break
    else:
        run_dir = Path(prev_cwd)
    if run_dir != Path(prev_cwd):
        print(f"  assets found in {run_dir}")
    os.chdir(run_dir)

    failed = None
    try:
        pygame.display.flip = hooked_flip
        exec(compile(script.read_text(), str(script), "exec"), namespace)
    except SystemExit:
        pass
    except FileNotFoundError as exc:
        # examples load assets with relative paths; a missing asset is the
        # script's problem, not the recorder's - say so plainly
        searched = ", ".join(str(c) for c in (Path(prev_cwd), ROOT, EXAMPLE_DIR))
        failed = (f"{script.name} could not find a file it needs:\n"
                  f"    {exc}\n"
                  f"  Looked for an assets/ folder in: {searched}\n"
                  f"  Run data_record.py from the same folder you would run\n"
                  f"  the example from, or move assets/ next to the script.")
    except Exception as exc:
        failed = f"{script.name} raised {type(exc).__name__}: {exc}"
    finally:
        pygame.display.flip = original_flip
        os.chdir(prev_cwd)
        if failed:
            print(f"\n  SCRIPT ERROR\n  {failed}")
        path = rec.save_log()
        if path:
            hits = sum(r["correct"] for r in rec.log)
            print(f"\n  {hits}/{len(rec.log)} trials matched the cue")
            print(f"  trial log -> {path}")
        if BCI is not None:
            BCI.end()
            print("  EEG recording stopped")
        try:
            pygame.quit()
        except Exception:
            pass
    return 0


# =====================================================================
#  CHARACTER MODE (original behaviour, unchanged)
# =====================================================================
def main():
    import matplotlib.pyplot as plt
    BCI = _load_bci()

    #init fields
    ##game
    pygame.init()
    pygame.font.init()
    clock = pygame.time.Clock()

    win = pygame.display.set_mode((800,600))
    width, height = win.get_size()

    size = int(width*0.1)
    center = (width/2, height/2)
    my_font = pygame.font.SysFont('Comic Sans MS', size)

    show = ""
    count = 0
    show_text = False
    allow_input = True
    buffer = False
    buffer_time = 1
    show_time = 2
    cycle_start = 0
    buffer_start = 0
    ##random character list
    ###length = 0 to manual input
    length = 0
    arr = random.choices(string.ascii_letters + string.digits, k=length)

    ##game loop
    run = True
    win.fill((0,0,0))

    BCI.start(BID=-2,port=None, plot_domain="f")
    #start live plot in non blocking mode
    ani = BCI.plot_data(block=False)

    win.fill((0,0,0))
    #for logging, start_time set only when end
    #phase = start/end
    #type = buffer/cycle
    def log(phase, type, current_time, count, start_time=-1):
        if phase == "start":
            print(f"Starting {count} {type} at {current_time//1000}s")
        elif phase == "end" and start_time != -1:
            print(f"Finished {count} {type} at {current_time//1000}s in {(current_time-start_time)//1000}s")

    """
    Edit by Anson
    Date: 18/7/2026
    Changes: wrap entire game flow in try and finally, only set run=False inside
    """
    try:
        if arr:
            print(arr)
            allow_input = False
            # send start marker of show
            show = arr.pop(0)
            BCI.put_marker(show, phase="start")
            count += 1
            show_text = True

            cycle_start = pygame.time.get_ticks()
            log("start", "cycle", cycle_start, count)
            display_time = cycle_start + (show_time * 1000)

        while run:
            current_time = pygame.time.get_ticks()
            #send gui events so chart update without freezing
            plt.pause(0.001)
            for e in pygame.event.get():
                if e.type == QUIT or (e.type == KEYDOWN and e.key == K_ESCAPE):
                    run = False
                elif e.type == KEYDOWN and not arr:
                    #if key pressed is not enter, key is alphanumeric, input is allowed and arr is empty
                    if e.key != K_RETURN and e.unicode.isalnum() and allow_input:
                        show += e.unicode
                        print(f"\rKeyboard: {show: <20}", end='', flush=True)

                    elif e.key == K_BACKSPACE and allow_input:
                        show = show[:-1]
                        print(f"\rKeyboard: {show: <20}", end='', flush=True)

                    elif e.key in (K_RETURN, K_KP_ENTER) and show and allow_input:
                        print()
                        #send start marker of show
                        BCI.put_marker(show, phase="start")
                        allow_input = False
                        count += 1
                        show_text = True

                        cycle_start = current_time
                        log("start", "cycle", current_time=current_time, count=count)
                        display_time = current_time + show_time*1000

            if length != 0 and not arr and not show_text and not buffer:
                run = False

            #showing text but passed display_time(2s)            
            if show_text and current_time >= display_time:
                #send end marker of show
                BCI.put_marker(show, phase="end")
                print(show)
                log("end", "cycle",current_time=current_time, count=count, start_time=cycle_start)
                show_text = False
                buffer = True
                buffer_start = current_time
                display_time = current_time + buffer_time*1000

            elif buffer and current_time >= display_time:
                buffer = False
                allow_input = True
                show = ""
                log("end", "buffer", current_time=current_time, count=count, start_time=buffer_start)
                print("Allow input")
                if arr:
                    show = arr.pop(0)
                    BCI.put_marker(show, phase="start")
                    count += 1
                    show_text = True
                    cycle_start = current_time
                    log("start", "cycle", current_time, count)
                    display_time = current_time + (show_time * 1000)

            if show_text:
                text_surface = my_font.render(show, False, (255, 255, 255))
                textRect = text_surface.get_rect(center=center)
                win.blit(text_surface, textRect)
            else:
                win.fill((0, 0, 0))

            #update full display surface to screen
            pygame.display.flip()
            clock.tick(60)
    finally:
        BCI.end()
        pygame.quit()


def cli():
    ap = argparse.ArgumentParser(
        description="Record EEG while running an example app with cued targets")
    ap.add_argument("--file", default=None,
                    help="script in example/ to run, e.g. rockpaperscissors")
    ap.add_argument("--trials", type=int, default=10, help="number of cued trials")
    ap.add_argument("--cue", type=float, default=5.0, help="seconds per cue")
    ap.add_argument("--rest", type=float, default=2.0, help="seconds between cues")
    ap.add_argument("--board-id", type=int, default=-2,
                    help="-2 streaming board (OpenBCI GUI) · 0 Cyton · -1 synthetic")
    ap.add_argument("--serial-port", default=None, help="e.g. COM4")
    ap.add_argument("--dry-run", action="store_true",
                    help="run the UI and cues without recording EEG")
    ap.add_argument("--list", action="store_true", help="list example scripts")
    args = ap.parse_args()

    if args.list:
        print("Scripts in example/:")
        for p in sorted(EXAMPLE_DIR.glob("*.py")):
            if p.stem != "__init__":
                print("  ", p.stem)
        return 0

    if args.file:
        return run_example(args)
    main()
    return 0


if __name__ == "__main__":
    sys.exit(cli())
