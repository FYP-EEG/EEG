"""
Author: Anson
Created on: 23/8/2026
Purpose: The shipping app -- a CALIBRATION page and a REAL page, with per-user profiles.

    python example/bci_app.py                     # pick user, calibrate, then use
    python example/bci_app.py --user anson        # jump straight to that user
    python example/bci_app.py --source hardware --serial-port COM4

WHY TWO PAGES
-------------
Calibration is per USER, not per install. The confidence threshold depends on an
individual's alpha amplitude and exactly where the electrodes sit today, so a
threshold fitted to one person makes the speller unresponsive or trigger-happy for
the next. The app therefore always opens on the calibration page and will not let an
uncalibrated user reach the speller.

PAGES
  USERS       pick or create a user
  CALIBRATE   run the 6-minute protocol, fit the threshold, save the profile
  READY       show the fitted numbers, confirm before going live
  SPELLER     the real EEG-to-Text page, using THIS user's threshold

Recordings and profiles are written to profiles/<user>/ which mirrors the Drive
layout (drive/anson/...), so the folder can be synced verbatim for training.
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pygame_lib"))
sys.path.insert(0, str(ROOT / "EEG"))

import numpy as np
import pygame

import bci_sdk
from bci_sdk.profiles import UserProfile, list_users

BG = (18, 20, 28)
PANEL = (30, 33, 44)
ACCENT = (58, 110, 165)
GOOD = (58, 150, 110)
WARN = (196, 132, 32)
TEXT = (232, 236, 244)
MUTED = (150, 158, 180)


class App:
    PAGE_USERS = "USERS"
    PAGE_CALIBRATE = "CALIBRATE"
    PAGE_READY = "READY"
    PAGE_SPELLER = "SPELLER"

    def __init__(self, source="sim", user=None, serial_port="COM4",
                 quick=False, size=(1024, 700)):
        pygame.init()
        pygame.font.init()
        self.win, self.display_info = bci_sdk.open_display(size, resizable=True)
        pygame.display.set_caption("BCI Communicator")
        self.clock = pygame.time.Clock()

        self.f_big = pygame.font.SysFont("dejavusans,arial", 34, bold=True)
        self.f_med = pygame.font.SysFont("dejavusans,arial", 22, bold=True)
        self.f = pygame.font.SysFont("dejavusans,arial", 17)
        self.f_sm = pygame.font.SysFont("dejavusans,arial", 14)

        self.source = source
        self.serial_port = serial_port
        self.quick = quick
        self.users = list_users()
        self.sel = 0
        self.typing = False
        self.new_name = ""
        self.profile = None
        self.page = App.PAGE_USERS
        self.status = "Select a user"
        self.calib = None

        if user:
            self.profile = UserProfile(user)
            self.page = (App.PAGE_READY if self.profile.is_calibrated
                         else App.PAGE_CALIBRATE)

    # ================================================================ USERS
    def draw_users(self):
        w, h = self.win.get_size()
        self.win.blit(self.f_big.render("Who is using the system?", True, TEXT), (60, 50))
        self.win.blit(self.f.render(
            "Calibration is per person - each user needs their own.", True, MUTED),
            (60, 96))

        y = 150
        for i, p in enumerate(self.users):
            box = pygame.Rect(60, y, w - 120, 58)
            pygame.draw.rect(self.win, PANEL if i != self.sel else ACCENT, box,
                             border_radius=10)
            self.win.blit(self.f_med.render(p.name, True, TEXT), (80, y + 15))
            if p.is_calibrated:
                age = p.age_days or 0
                stale = p.is_stale()
                txt = (f"calibrated {age:.1f}d ago"
                       f"{'  - STALE, recalibrate' if stale else ''}")
                self.win.blit(self.f_sm.render(txt, True, WARN if stale else GOOD),
                              (w - 380, y + 22))
            else:
                self.win.blit(self.f_sm.render("not calibrated", True, WARN),
                              (w - 380, y + 22))
            y += 68

        box = pygame.Rect(60, y, w - 120, 58)
        pygame.draw.rect(self.win, ACCENT if self.sel == len(self.users) else PANEL,
                         box, border_radius=10)
        label = f"New user: {self.new_name}_" if self.typing else "+ New user"
        self.win.blit(self.f_med.render(label, True, TEXT), (80, y + 15))

        self.win.blit(self.f_sm.render(
            "UP/DOWN select    ENTER confirm    ESC quit", True, MUTED), (60, h - 40))

    def users_key(self, e):
        if self.typing:
            if e.key == pygame.K_RETURN and self.new_name.strip():
                self.profile = UserProfile(self.new_name.strip())
                self.typing = False
                self.new_name = ""
                self.page = App.PAGE_CALIBRATE
            elif e.key == pygame.K_BACKSPACE:
                self.new_name = self.new_name[:-1]
            elif e.key == pygame.K_ESCAPE:
                self.typing = False
            elif e.unicode and e.unicode.isprintable():
                self.new_name += e.unicode
            return
        if e.key == pygame.K_DOWN:
            self.sel = min(self.sel + 1, len(self.users))
        elif e.key == pygame.K_UP:
            self.sel = max(self.sel - 1, 0)
        elif e.key == pygame.K_RETURN:
            if self.sel == len(self.users):
                self.typing = True
            else:
                self.profile = self.users[self.sel]
                self.page = (App.PAGE_READY if self.profile.is_calibrated
                             else App.PAGE_CALIBRATE)

    # ============================================================ CALIBRATE
    def draw_calibrate(self):
        w, h = self.win.get_size()
        self.win.blit(self.f_big.render(
            f"Calibration - {self.profile.name}", True, TEXT), (60, 50))

        lines = [
            "This teaches the system what YOUR brain looks like:",
            "",
            "  1. Stare at the LEFT tile (15 Hz)        10 x 5s",
            "  2. Stare at the RIGHT tile (20 Hz)       10 x 5s",
            "  3. REST - look at the cross, relax        5 x 30s",
            "  4. Read text / look away                  3 x 30s",
            "  5. Blink deliberately                     4 x 10s",
            "",
            "Steps 3-5 matter most: they teach the system when you are NOT",
            "  selecting, which is what stops it typing while you think.",
            "",
            "Takes about 6 minutes. Press ESC during it to abort.",
        ]
        y = 120
        for ln in lines:
            self.win.blit(self.f.render(ln, True,
                                        MUTED if ln.startswith("  ") else TEXT), (60, y))
            y += 26

        if self.profile.is_calibrated:
            self.win.blit(self.f_sm.render(
                f"Existing calibration from {self.profile.age_days:.1f} days ago "
                f"- ENTER re-runs it, S skips.", True, WARN), (60, y + 10))

        pygame.draw.rect(self.win, GOOD, pygame.Rect(60, h - 90, 320, 46),
                         border_radius=10)
        self.win.blit(self.f_med.render("ENTER  Start calibration", True, TEXT),
                      (78, h - 78))
        self.win.blit(self.f_sm.render(
            "!! Photosensitive epilepsy: flickering light. Do not proceed if at risk.",
            True, WARN), (60, h - 34))

    def run_calibration(self):
        """Run the Step 3 protocol, fit a threshold, save into profiles/<user>/."""
        import calibration_record as cr
        from ML import calibration as cal

        self.status = "calibrating..."
        backend = (cr.BrainFlowBackend(serial_port=self.serial_port)
                   if self.source == "hardware" else cr.SimulateBackend())
        protocol = cr.DEFAULT_PROTOCOL
        if self.quick:
            protocol = [(b, c, 1, min(s, 6.0), i)
                        for b, c, r, s, i in cr.DEFAULT_PROTOCOL]

        ui = cr.StimulusUI(target_freqs=(15.0, 20.0))
        actual = list(ui.target_freqs)
        backend.start()
        try:
            X, y, labels, aborted = cr.run_protocol(
                backend, ui, protocol, 250, 750, 250, verbose=False)
        finally:
            backend.stop()
            ui.close()

        # reopen our own window (the recorder owned the display)
        self.win, self.display_info = bci_sdk.open_display((1024, 700), resizable=True)
        pygame.display.set_caption("BCI Communicator")

        if not X:
            self.status = "calibration produced no data"
            return False

        out = self.profile.recording_path()
        cr.save(X, y, labels, self.profile.slug, 250, "cyton8_ssvep", actual,
                out.parent)
        rec = sorted(out.parent.glob("calib_*.npz"))[-1]

        prof = cal.analyse(str(rec), percentile=95.0,
                           out=self.profile.dir / "analysis.json")
        if not prof:
            self.status = "could not fit a threshold (no clean idle data)"
            return False
        self.profile.update_from_analysis(prof, recording_path=rec)
        self.status = f"calibrated: threshold {prof['confidence_threshold']:.4f}"
        self.page = App.PAGE_READY
        return True

    # ================================================================ READY
    def draw_ready(self):
        w, h = self.win.get_size()
        p = self.profile
        self.win.blit(self.f_big.render(f"Ready - {p.name}", True, TEXT), (60, 50))

        box = pygame.Rect(60, 120, w - 120, 210)
        pygame.draw.rect(self.win, PANEL, box, border_radius=12)
        pygame.draw.rect(self.win, GOOD, box, width=2, border_radius=12)

        m = p.data.get("metrics", {})
        rows = [
            ("confidence threshold", f"{p.threshold:.4f}" if p.threshold else "-"),
            ("stimulus", f"{p.target_freqs[0]:.1f} / {p.target_freqs[1]:.1f} Hz"),
            ("calibrated", f"{p.age_days:.1f} days ago" if p.age_days is not None else "-"),
            ("idle recorded", f"{p.data.get('idle_minutes', 0):.1f} min"),
            ("false cmds/min idle", f"{m.get('fp_per_min', float('nan')):.2f}"),
            ("precision", f"{m.get('precision', 0)*100:.1f}%"),
        ]
        y = 145
        for k, v in rows:
            self.win.blit(self.f.render(k, True, MUTED), (85, y))
            self.win.blit(self.f.render(str(v), True, TEXT), (420, y))
            y += 30

        if p.is_stale():
            self.win.blit(self.f.render(
                "This calibration is over a week old - electrodes drift. "
                "Consider recalibrating (C).", True, WARN), (60, 350))

        pygame.draw.rect(self.win, GOOD, pygame.Rect(60, h - 110, 300, 50),
                         border_radius=10)
        self.win.blit(self.f_med.render("ENTER  Open speller", True, TEXT),
                      (78, h - 96))
        self.win.blit(self.f_sm.render(
            "C recalibrate     U switch user     ESC quit", True, MUTED), (60, h - 42))

    # ============================================================== SPELLER
    def open_speller(self):
        """Launch the real page with THIS user's calibrated threshold."""
        from example.bci_speller import Speller
        kwargs = dict(source=("engine" if self.source == "hardware" else "sim"),
                      serial_port=self.serial_port)
        if self.profile.is_calibrated:
            kwargs["profile"] = str(self.profile.path)
        app = Speller(**kwargs)
        text = app.run()
        self.win, self.display_info = bci_sdk.open_display((1024, 700), resizable=True)
        pygame.display.set_caption("BCI Communicator")
        self.status = f'typed: "{text.strip()[:40]}"' if text.strip() else "speller closed"
        self.page = App.PAGE_READY

    # ================================================================= loop
    def run(self):
        running = True
        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.VIDEORESIZE:
                    self.win, self.display_info = bci_sdk.open_display(
                        e.size, resizable=True)
                elif e.type == pygame.KEYDOWN:
                    if self.page == App.PAGE_USERS:
                        if e.key == pygame.K_ESCAPE and not self.typing:
                            running = False
                        else:
                            self.users_key(e)
                    elif self.page == App.PAGE_CALIBRATE:
                        if e.key == pygame.K_ESCAPE:
                            self.page = App.PAGE_USERS
                        elif e.key == pygame.K_RETURN:
                            self.run_calibration()
                        elif e.key == pygame.K_s and self.profile.is_calibrated:
                            self.page = App.PAGE_READY
                    elif self.page == App.PAGE_READY:
                        if e.key == pygame.K_ESCAPE:
                            running = False
                        elif e.key == pygame.K_RETURN:
                            self.open_speller()
                        elif e.key == pygame.K_c:
                            self.page = App.PAGE_CALIBRATE
                        elif e.key == pygame.K_u:
                            self.users = list_users()
                            self.page = App.PAGE_USERS

            self.win.fill(BG)
            if self.page == App.PAGE_USERS:
                self.draw_users()
            elif self.page == App.PAGE_CALIBRATE:
                self.draw_calibrate()
            elif self.page == App.PAGE_READY:
                self.draw_ready()

            w, h = self.win.get_size()
            self.win.blit(self.f_sm.render(
                f"{self.page}   src={self.source}   {self.status}", True, MUTED),
                (w - 470, 18))
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


def main():
    ap = argparse.ArgumentParser(description="BCI Communicator (calibration + speller)")
    ap.add_argument("--source", choices=["sim", "hardware"], default="sim")
    ap.add_argument("--user", default=None)
    ap.add_argument("--serial-port", default="COM4")
    ap.add_argument("--quick", action="store_true",
                    help="shortened calibration, for testing the flow")
    a = ap.parse_args()
    App(source=a.source, user=a.user, serial_port=a.serial_port, quick=a.quick).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
