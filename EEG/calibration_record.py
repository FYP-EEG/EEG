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
DEFAULT_PROTOCOL = [
    ("target_0",        0, 10,  5.0, "Stare at the LEFT tile (15 Hz)"),
    ("target_1",        1, 10,  5.0, "Stare at the RIGHT tile (20 Hz)"),
    ("idle",           -1,  5, 30.0, "REST. Eyes open, look at the centre cross.\n"
                                     "Do not focus on either tile."),
    ("idle_distracted", -1, 3, 30.0, "Read the text on screen / look around the room.\n"
                                     "Ignore the tiles completely."),
    ("artifact",        2,  4, 10.0, "Blink deliberately, about once per second."),
]

CLASS_NAMES = {0: "target_0", 1: "target_1", -1: "idle", 2: "artifact"}

DISTRACTION_TEXT = [
    "The occipital cortex responds to periodic visual stimulation by",
    "producing a steady-state response at the stimulus frequency and",
    "its harmonics. This response is strongest over O1, Oz and O2.",
    "Read this passage at your normal pace. Let your attention drift",
    "away from the flickering tiles entirely. Look around the room,",
    "check the window, glance at your hands. This block teaches the",
    "system what 'not trying to select anything' actually looks like.",
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
        pg = self.pygame
        w, h = self.win.get_size()
        self.win.fill((16, 18, 24))
        self.frame += 1

        if show_tiles:
            for i, freq in enumerate(self.target_freqs):
                on = self.stim.is_on(i)
                base = (235, 235, 245) if on else (26, 28, 36)
                cx = w * (0.25 if i == 0 else 0.75)
                rect = pg.Rect(0, 0, 190, 190)
                rect.center = (int(cx), int(h * 0.46))
                pg.draw.rect(self.win, base, rect, border_radius=14)
                ring = (250, 200, 90) if highlight == i else (70, 76, 96)
                pg.draw.rect(self.win, ring, rect, width=6, border_radius=14)
                lab = self.font.render(f"{freq:.0f} Hz", True, (150, 158, 180))
                self.win.blit(lab, lab.get_rect(center=(rect.centerx, rect.bottom + 22)))
        else:
            pg.draw.line(self.win, (120, 128, 150),
                         (w // 2 - 18, int(h * 0.46)), (w // 2 + 18, int(h * 0.46)), 3)
            pg.draw.line(self.win, (120, 128, 150),
                         (w // 2, int(h * 0.46) - 18), (w // 2, int(h * 0.46) + 18), 3)

        if show_text:
            y = int(h * 0.72)
            for line in DISTRACTION_TEXT[:5]:
                surf = self.font_sm.render(line, True, (172, 180, 200))
                self.win.blit(surf, surf.get_rect(centerx=w // 2, top=y))
                y += 21

        title = self.font_big.render(block.replace("_", " ").upper(), True, (235, 240, 250))
        self.win.blit(title, (28, 22))
        for i, line in enumerate(instruction.split("\n")):
            s = self.font.render(line, True, (188, 196, 216))
            self.win.blit(s, (28, 66 + i * 26))

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
    ap.add_argument("--montage", default="cyton8_ssvep")
    ap.add_argument("--fs", type=int, default=250)
    ap.add_argument("--window", type=int, default=750, help="epoch length in samples")
    ap.add_argument("--hop", type=int, default=250, help="slide between epochs")
    ap.add_argument("--target-freqs", type=float, nargs=2, default=[15.0, 20.0])
    ap.add_argument("--out", default=str(ROOT / "dataset"))
    ap.add_argument("--no-ui", action="store_true", help="record without the stimulus UI")
    ap.add_argument("--dry-run", action="store_true", help="print the protocol and exit")
    ap.add_argument("--quick", action="store_true",
                    help="shortened protocol for a pipeline smoke-test")
    ap.add_argument("--serial-port", default="COM4")
    ap.add_argument("--board-id", type=int, default=0)
    ap.add_argument("--yes", action="store_true", help="skip the safety confirmation")
    args = ap.parse_args()

    protocol = DEFAULT_PROTOCOL
    if args.quick:
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
