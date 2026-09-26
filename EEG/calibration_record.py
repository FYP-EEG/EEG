"""
Author: Brian
Created on: 16/8/2026
Purpose: Step 3 -- protocol-driven calibration recorder that captures LABELLED
         subject-specific data INCLUDING IDLE AND DISTRACTION blocks.
Location: project_dir/EEG/calibration_record.py

WHY THIS EXISTS
---------------
Public benchmarks (Tsinghua, PhysioNet) only ever contain attentive trials: the
subject is always being asked to do something. They contain no "user isn't trying"
class. But a GUI runs continuously and the user is idle most of the wall-clock time,
so the failure mode that actually breaks the product -- false positives during idle --
is precisely the one public data cannot measure or calibrate against.

This script records the missing classes on YOUR hardware, with YOUR head geometry:

    target_0          gaze at the 15 Hz tile
    target_1          gaze at the 20 Hz tile
    idle              rest, eyes open, no tile
    idle_distracted   read text on screen / look away
    artifact          deliberate blinks

Output: dataset/calib_<subject>_<timestamp>.npz  with
    X       (n_epochs, n_channels, n_times) float32
    y       (n_epochs,) int      -- class index
    labels  (n_epochs,) str      -- block name
    fs, montage, target_freqs, channel_labels

Backends
    --backend simulate   synthetic EEG, no hardware (default; lets you rehearse
                         the whole protocol and validate the pipeline)
    --backend brainflow  live OpenBCI Cyton via BrainFlow

SAFETY
------
!! PHOTOSENSITIVE EPILEPSY WARNING !!
This script renders flickering visual stimuli at 15 and 20 Hz. Do not run it with
anyone who has photosensitive epilepsy or a history of seizures. Stop immediately
if a participant reports discomfort, dizziness, or visual disturbance. The script
prints this warning and requires explicit confirmation before starting.

Usage
    python EEG/calibration_record.py --subject S01 --backend simulate
    python EEG/calibration_record.py --subject S01 --backend brainflow --serial-port COM4
    python EEG/calibration_record.py --subject S01 --dry-run      # print protocol only
"""

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pygame_lib"))

from ML import montage as montage_mod  # noqa: E402

# --------------------------------------------------------------------- protocol
#: (block name, class index, n_repeats, seconds each, on-screen instruction)
"""
MOTOR IMAGERY PROTOCOLS
Added by Anson, 21/9/2026 (MI-only pivot)

MI is a fundamentally different task from gaze, and the protocol has to reflect
that:

  * There is no stimulus to look at. The cue tells the user WHAT to imagine, and
    the imagery itself produces the signal. So the cue must be short and the
    imagery window must be long enough for the rhythm to desynchronise.
  * The effect is a REDUCTION in mu/beta power (event-related desynchronisation),
    which takes ~0.5-1 s to develop after the cue. The first second of each trial
    is therefore partly unusable - which is why blocks are longer than the SSVEP
    ones were.
  * "Imagine squeezing your LEFT hand" works far better than "think about moving".
    Kinaesthetic imagery (feel the muscles) beats visual imagery (picture a hand).
  * REST blocks matter as much as here as they did for SSVEP: without them the
    system cannot tell imagery from ordinary thinking.

Class layout:
    0 = imagine LEFT hand   -> desynchronisation over C4 (contralateral)
    1 = imagine RIGHT hand  -> desynchronisation over C3
   -1 = rest / no command
    2 = deliberate blinks, for the artifact detector
"""

#: ~2 minute protocol for a per-session warm-up.
#: Longer than the 60 s SSVEP version because MI trials need more time each and
#: MI is inherently noisier, so more repetitions are needed for a stable estimate.
QUICK_PROTOCOL = [
    ("target_0",         0, 4, 6.0, "Imagine SQUEEZING your LEFT hand\n"
                                    "Feel the muscles - do not actually move"),
    ("target_1",         1, 4, 6.0, "Imagine SQUEEZING your RIGHT hand\n"
                                    "Feel the muscles - do not actually move"),
    ("idle",            -1, 2, 20.0, "REST. Stay still, let your mind wander.\n"
                                     "Do not imagine any movement."),
    ("idle_distracted", -1, 1, 15.0, "Count backwards from 100 in sevens.\n"
                                     "Thinking hard, but NOT moving."),
]

DEFAULT_PROTOCOL = [
    ("target_0",         0, 10, 6.0, "Imagine SQUEEZING your LEFT hand\n"
                                     "Feel the muscles - do not actually move"),
    ("target_1",         1, 10, 6.0, "Imagine SQUEEZING your RIGHT hand\n"
                                     "Feel the muscles - do not actually move"),
    ("idle",            -1,  5, 30.0, "REST. Stay still, let your mind wander.\n"
                                      "Do not imagine any movement."),
    ("idle_distracted", -1,  3, 30.0, "Count backwards from 100 in sevens.\n"
                                      "Thinking hard, but NOT moving."),
    ("artifact",         2,  4, 10.0, "Blink deliberately, about once per second."),
]

CLASS_NAMES = {0: "target_0", 1: "target_1", -1: "idle", 2: "artifact"}

DISTRACTION_TEXT = [
    "Count backwards from 100 in steps of seven, silently.",
    "100, 93, 86, 79 ...",
    "Keep your hands completely still and relaxed.",
    "This block is hard mental work WITHOUT any movement imagery,",
    "which teaches the system that thinking alone is not a command.",
]


def epoch_seconds_to_windows(seconds, fs, window, hop):
    n = int(seconds * fs)
    if n < window:
        return 0
    return 1 + (n - window) // hop


# ------------------------------------------------------------------- backends
class SimulateBackend:
    """Synthetic acquisition so the protocol can be rehearsed without hardware."""

    name = "simulate"

    def __init__(self, fs=250, n_channels=8, target_freqs=(15.0, 20.0),
                 occipital=(6, 7), frontal=(0, 1), seed=None):
        self.fs = fs
        self.n_channels = n_channels
        self.target_freqs = list(target_freqs)
        self.occipital = list(occipital)
        self.frontal = list(frontal)
        self.rng = np.random.default_rng(seed)
        self._state = None
        self._t0 = 0.0

    def start(self):
        self._t0 = 0.0

    def stop(self):
        pass

    def set_state(self, block_name, class_idx):
        self._state = (block_name, class_idx)

    def read(self, n_samples):
        """Return (n_channels, n_samples) of synthetic uV data for the current block."""
        t = self._t0 + np.arange(n_samples) / self.fs
        self._t0 += n_samples / self.fs
        x = self.rng.standard_normal((self.n_channels, n_samples)) * 3.0

        # resting alpha, always present, strongest occipitally
        a_f = self.rng.uniform(9.5, 10.5)
        a_ph = self.rng.uniform(0, 2 * np.pi)
        for ch in self.occipital:
            x[ch] += 4.5 * np.sin(2 * np.pi * a_f * t + a_ph)

        block, cls = self._state or ("idle", -1)
        if cls in (0, 1):
            freq = self.target_freqs[cls]
            ph = self.rng.uniform(0, 2 * np.pi)
            for ch in self.occipital:
                for h, amp in ((1, 1.0), (2, 0.45), (3, 0.2)):
                    x[ch] += 2.0 * amp * np.sin(2 * np.pi * h * freq * t + ph)
        elif cls == 2:
            # ~1 blink/sec: 150-250 uV frontal deflections
            for onset in range(0, n_samples, int(self.fs)):
                w = min(int(0.15 * self.fs), n_samples - onset)
                if w <= 0:
                    continue
                shape = np.hanning(w) * self.rng.uniform(150, 250)
                for ch in self.frontal:
                    x[ch, onset:onset + w] += shape
        elif block == "idle_distracted":
            # drifting eye movement + occasional muscle bursts
            x += np.linspace(0, self.rng.uniform(-8, 8), n_samples)[None, :]
            if self.rng.random() < 0.3:
                s = self.rng.integers(0, max(1, n_samples - 50))
                x[:, s:s + 50] += self.rng.standard_normal((self.n_channels, 50)) * 12
        return x


class BrainFlowBackend:
    """Live OpenBCI acquisition. Imported lazily so the module works without it."""

    name = "brainflow"

    def __init__(self, board_id=0, serial_port="COM4", fs=250, n_channels=8,
                 ip_address="225.1.1.1", ip_port=6677, master_board=-1):
        from brainflow.board_shim import BoardShim, BrainFlowInputParams  # noqa
        self.BoardShim = BoardShim
        params = BrainFlowInputParams()
        params.serial_port = serial_port
        params.ip_address = ip_address
        params.ip_port = ip_port
        params.master_board = master_board
        self.board = BoardShim(board_id, params)
        self.board_id = board_id
        self.fs = fs
        self.n_channels = n_channels
        self.eeg_channels = None

    def start(self):
        self.board.prepare_session()
        self.board.start_stream(45000)
        real_id = self.board_id
        self.eeg_channels = self.BoardShim.get_eeg_channels(real_id)[:self.n_channels]
        self.fs = self.BoardShim.get_sampling_rate(real_id)
        time.sleep(2.0)                      # let the buffer fill / settle

    def stop(self):
        try:
            if self.board.is_prepared():
                self.board.stop_stream()
                self.board.release_session()
        except Exception:
            pass

    def set_state(self, block_name, class_idx):
        """Insert a hardware marker so the CSV can be re-epoched later."""
        try:
            self.board.insert_marker(float(class_idx + 10))
        except Exception:
            pass

    def read(self, n_samples):
        """Block until n_samples are available, then return them."""
        need = n_samples
        deadline = time.time() + (need / self.fs) + 5.0
        while self.board.get_board_data_count() < need and time.time() < deadline:
            time.sleep(0.01)
        data = self.board.get_board_data(need)
        return data[self.eeg_channels, :]


# ------------------------------------------------------------------ UI (pygame)
class StimulusUI:
    """Two flickering tiles + instruction text. Frame-counted, V-synced.

    Uses DOUBLEBUF | SCALED with vsync=1 -- NOT OPENGL|DOUBLEBUF, which would make
    the surface an OpenGL context and break every blit (see instructions_verification
    section A3). Flicker toggles on exact frame counts, previewing Step 5.
    """

    def __init__(self, target_freqs=(15.0, 20.0), size=(1000, 640), refresh=None):
        import pygame
        from stimulus import StimulusEngine, open_display, detect_refresh_rate
        self.pygame = pygame
        pygame.init()
        pygame.font.init()
        self.win, self.display_info = open_display(size, refresh_hint=refresh)
        pygame.display.set_caption("BCI Calibration - keep still, follow instructions")
        self.clock = pygame.time.Clock()
        self.refresh = self.display_info["refresh"]

        # Step 5: frame-quantised flicker. The previous half-period rounding
        # collapsed 20 Hz onto 15 Hz on a 60 Hz monitor -- both tiles would have
        # flickered identically and every target_1 trial would have been mislabelled.
        self.stim = StimulusEngine(target_freqs=target_freqs, refresh_hz=self.refresh)
        self.target_freqs = [p.actual for p in self.stim.plans]
        self.requested_freqs = list(target_freqs)
        self.frame = 0
        self.font_big = pygame.font.SysFont("dejavusans,arial", 30, bold=True)
        self.font = pygame.font.SysFont("dejavusans,arial", 20)
        self.font_sm = pygame.font.SysFont("dejavusans,arial", 15)

    def pump(self):
        """Return False if the user asked to quit."""
        for e in self.pygame.event.get():
            if e.type == self.pygame.QUIT:
                return False
            if e.type == self.pygame.KEYDOWN and e.key == self.pygame.K_ESCAPE:
                return False
        return True

    def draw(self, block, instruction, remaining, progress, show_tiles=True,
             show_text=False, highlight=None):
        """
        Rewritten 21/9/2026 for motor imagery.

        There is no flicker any more. MI needs the screen to say WHICH hand to
        imagine and to hold that cue steady - a moving or flashing cue would only
        add visual evoked activity on top of the motor rhythm we want.

        highlight: 0 = left hand, 1 = right hand, None = rest / no imagery
        """
        pg = self.pygame
        w, h = self.win.get_size()
        self.win.fill((16, 18, 24))
        self.frame += 1

        cx, cy = w // 2, int(h * 0.46)

        if highlight in (0, 1):
            # big directional arrow - unambiguous at a glance, and identical on
            # every trial so it contributes the same visual response each time
            side = -1 if highlight == 0 else 1
            col = (235, 235, 245)
            head = 46                                  # half-height of the head
            tip_x = cx + side * int(w * 0.20)          # outermost point
            base_x = tip_x - side * head               # where the head meets shaft
            # shaft runs from the centre to the head base, overlapping slightly
            # so there is no seam between the two shapes
            shaft_far = cx - side * int(w * 0.04)
            left = min(base_x + side * 4, shaft_far)
            shaft = pg.Rect(left, cy - 13, abs(base_x + side * 4 - shaft_far), 26)
            pg.draw.rect(self.win, col, shaft, border_radius=4)
            pg.draw.polygon(self.win, col, [
                (tip_x, cy),
                (base_x, cy - head),
                (base_x, cy + head),
            ])
            hand = "LEFT HAND" if highlight == 0 else "RIGHT HAND"
            lab = self.font_big.render(hand, True, (250, 214, 110))
            self.win.blit(lab, lab.get_rect(center=(cx, cy + 110)))
            sub = self.font.render("imagine squeezing - do not move", True,
                                   (150, 158, 180))
            self.win.blit(sub, sub.get_rect(center=(cx, cy + 146)))
        else:
            # fixation cross for rest, so the eyes have somewhere neutral to sit
            pg.draw.line(self.win, (120, 128, 150),
                         (cx - 20, cy), (cx + 20, cy), 3)
            pg.draw.line(self.win, (120, 128, 150),
                         (cx, cy - 20), (cx, cy + 20), 3)

        if show_text:
            y = int(h * 0.72)
            for line in DISTRACTION_TEXT[:5]:
                surf = self.font_sm.render(line, True, (172, 180, 200))
                self.win.blit(surf, surf.get_rect(centerx=w // 2, top=y))
                y += 21

        title = self.font_big.render(block.replace("_", " ").upper(), True,
                                     (235, 240, 250))
        self.win.blit(title, (28, 22))
        for i, line in enumerate(instruction.split("\n")):
            srf = self.font.render(line, True, (188, 196, 216))
            self.win.blit(srf, (28, 66 + i * 26))

        t = self.font_big.render(f"{remaining:4.1f}s", True, (250, 214, 110))
        self.win.blit(t, t.get_rect(topright=(w - 28, 22)))

        bar = pg.Rect(28, h - 34, int((w - 56) * progress), 12)
        pg.draw.rect(self.win, (40, 44, 58), pg.Rect(28, h - 34, w - 56, 12),
                     border_radius=6)
        pg.draw.rect(self.win, (58, 150, 110), bar, border_radius=6)

        pg.display.flip()
        self.stim.tick()
        self.clock.tick(int(self.refresh) + 5)

    def message(self, lines, seconds):
        """Show a between-block rest screen. Returns False if quit."""
        end = time.time() + seconds
        while time.time() < end:
            if not self.pump():
                return False
            w, h = self.win.get_size()
            self.win.fill((16, 18, 24))
            for i, line in enumerate(lines):
                f = self.font_big if i == 0 else self.font
                s = f.render(line, True, (225, 232, 245) if i == 0 else (170, 178, 198))
                self.win.blit(s, s.get_rect(center=(w // 2, h // 2 - 40 + i * 38)))
            r = self.font.render(f"{end - time.time():.0f}", True, (250, 214, 110))
            self.win.blit(r, r.get_rect(center=(w // 2, h // 2 + 130)))
            self.pygame.display.flip()
            self.clock.tick(self.refresh)
        return True

    def close(self):
        self.pygame.quit()


# ------------------------------------------------------------------- recording
def run_protocol(backend, ui, protocol, fs, window, hop, rest_seconds=4.0,
                 verbose=True):
    """Execute the protocol, returning (X, y, labels)."""
    X, y, labels = [], [], []

    for block, cls, reps, secs, instruction in protocol:
        for rep in range(reps):
            if ui and not ui.message(
                    [f"{block.replace('_',' ').upper()}  ({rep+1}/{reps})",
                     *instruction.split("\n"), "", "Get ready..."], rest_seconds):
                return X, y, labels, True

            backend.set_state(block, cls)
            n_win = epoch_seconds_to_windows(secs, fs, window, hop)
            if n_win <= 0:
                continue

            # prime the buffer with one full window, then slide by `hop`
            buf = backend.read(window)
            start = time.time()
            for wi in range(n_win):
                if wi > 0:
                    new = backend.read(hop)
                    buf = np.concatenate([buf[:, hop:], new], axis=1)
                X.append(buf.astype(np.float32).copy())
                y.append(cls)
                labels.append(block)

                if ui:
                    elapsed = time.time() - start
                    if not ui.pump():
                        return X, y, labels, True
                    ui.draw(block, instruction,
                            remaining=max(0.0, secs - elapsed),
                            progress=(wi + 1) / n_win,
                            show_tiles=(block != "idle"),
                            show_text=(block == "idle_distracted"),
                            highlight=cls if cls in (0, 1) else None)

            if verbose:
                print(f"  {block:<16} rep {rep+1}/{reps}: {n_win} windows")

    return X, y, labels, False


def save(X, y, labels, subject, fs, montage_name, target_freqs, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"calib_{subject}_{stamp}.npz"
    m = montage_mod.get(montage_name)
    np.savez_compressed(
        path,
        X=np.asarray(X, dtype=np.float32),
        y=np.asarray(y, dtype=np.int64),
        labels=np.asarray(labels),
        fs=fs,
        montage=montage_name,
        target_freqs=np.asarray(target_freqs),
        channel_labels=np.asarray(m.get("labels", [])),
        subject=subject,
        created=stamp,
    )
    return path


SAFETY = """
================================================================================
  !!  PHOTOSENSITIVE EPILEPSY WARNING  !!

  This session displays flickering visual stimuli at 15 Hz and 20 Hz.
  DO NOT proceed if the participant has photosensitive epilepsy, a seizure
  history, or is unsure. Stop immediately on any discomfort, dizziness,
  headache or visual disturbance.

  Participants should be seated, well rested, and free to stop at any time.
  Press ESC at any point to abort the recording.
================================================================================
"""


def main():
    ap = argparse.ArgumentParser(description="BCI calibration recorder (Step 3)")
    ap.add_argument("--subject", default="S01")
    ap.add_argument("--backend", choices=["simulate", "brainflow"], default="simulate")
    ap.add_argument("--montage", default="cyton8_mi")
    ap.add_argument("--fs", type=int, default=250)
    ap.add_argument("--window", type=int, default=750, help="epoch length in samples")
    ap.add_argument("--hop", type=int, default=250, help="slide between epochs")
    ap.add_argument("--target-freqs", type=float, nargs=2, default=[15.0, 20.0])
    ap.add_argument("--out", default=str(ROOT / "dataset"))
    ap.add_argument("--no-ui", action="store_true", help="record without the stimulus UI")
    ap.add_argument("--dry-run", action="store_true", help="print the protocol and exit")
    ap.add_argument("--protocol", choices=["full", "quick"], default="full",
                    help="'quick' = 60s per-session protocol; 'full' = 6.3min")
    ap.add_argument("--quick", action="store_true",
                    help="tiny protocol for a pipeline smoke-test (not for real use)")
    ap.add_argument("--serial-port", default="COM4")
    ap.add_argument("--board-id", type=int, default=0)
    ap.add_argument("--yes", action="store_true", help="skip the safety confirmation")
    args = ap.parse_args()

    protocol = DEFAULT_PROTOCOL
    if args.protocol == "quick":
        protocol = QUICK_PROTOCOL
    elif args.quick:                       # smoke-test: tiny, not for real use
        protocol = [(b, c, 1, min(s, 6.0), i) for b, c, r, s, i in DEFAULT_PROTOCOL]

    total_s = sum(r * s for _, _, r, s, _ in protocol)
    total_w = sum(r * epoch_seconds_to_windows(s, args.fs, args.window, args.hop)
                  for _, _, r, s, _ in protocol)
    print("\nProtocol")
    print("-" * 68)
    for b, c, r, s, _ in protocol:
        n = r * epoch_seconds_to_windows(s, args.fs, args.window, args.hop)
        print(f"  {b:<18} class={c:>2}  {r:>2} x {s:>4.1f}s  -> {n:>4} windows")
    print("-" * 68)
    print(f"  TOTAL {total_s/60:.1f} min of recording, {total_w} windows "
          f"({args.window/args.fs:.1f}s each, {args.hop/args.fs:.1f}s hop)\n")
    if args.dry_run:
        return 0

    print(SAFETY)
    if not args.yes and args.backend == "brainflow":
        if input("Type 'yes' to confirm the participant is safe to proceed: ").strip().lower() != "yes":
            print("Aborted.")
            return 1

    if args.backend == "brainflow":
        backend = BrainFlowBackend(board_id=args.board_id, serial_port=args.serial_port,
                                   fs=args.fs)
    else:
        backend = SimulateBackend(fs=args.fs, target_freqs=args.target_freqs)

    ui = None
    actual_freqs = list(args.target_freqs)
    if not args.no_ui:
        try:
            ui = StimulusUI(target_freqs=args.target_freqs)
            ui.stim.print_report()
            # Label the recording with what the monitor ACTUALLY rendered.
            actual_freqs = list(ui.target_freqs)
            if any(abs(a - r) > 1e-6 for a, r in zip(actual_freqs, args.target_freqs)):
                print(f"\n  NOTE: requested {args.target_freqs} but this monitor renders "
                      f"{[round(f,3) for f in actual_freqs]}.\n"
                      f"  The recording is labelled with the ACTUAL frequencies.\n")
        except Exception as exc:
            print(f"UI unavailable ({exc}); continuing headless.")

    backend.start()
    aborted = False
    try:
        X, y, labels, aborted = run_protocol(backend, ui, protocol, args.fs,
                                             args.window, args.hop)
    finally:
        backend.stop()
        if ui:
            ui.close()

    if not X:
        print("No data recorded.")
        return 1

    path = save(X, y, labels, args.subject, args.fs, args.montage,
                actual_freqs, args.out)
    print(f"\n{'ABORTED - partial data' if aborted else 'Complete'}: "
          f"{len(X)} windows -> {path}")
    print(f"\nNext:  python ML/calibration.py --recording {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
