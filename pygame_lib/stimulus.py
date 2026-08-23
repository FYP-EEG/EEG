"""
Author: Anson
Created on: 16/8/2026
Purpose: Step 5 -- V-synced, frame-accurate SSVEP stimulus engine (instructions.md Gap 5).

TWO CORRECTIONS TO THE HANDOFF INSTRUCTIONS
-------------------------------------------
1. `instructions.md` says to use `pygame.OPENGL | pygame.DOUBLEBUF` with vsync=1.
   That makes the display an OpenGL context: `Surface.blit()` stops working and every
   existing Button/TileButton render silently disappears. The correct incantation for
   a 2D blitting app is:
        pygame.display.set_mode(size, pygame.DOUBLEBUF | pygame.SCALED, vsync=1)
   SCALED is what actually enables the vsync flag in SDL2.

2. `instructions.md` says "on a 60Hz monitor, a 10Hz target flashes every 6 frames",
   i.e. a symmetric half-period of refresh/(2f). That only works when refresh/(2f) is
   an integer. On a 60 Hz monitor:

        request 15 Hz -> half_period 2 -> ACTUAL 15.00 Hz   ok
        request 20 Hz -> half_period 2 -> ACTUAL 15.00 Hz   ** WRONG **

   Rounding collapses 20 Hz onto 15 Hz -- both tiles would flicker identically and the
   classifier could never separate them. The fix is to quantise the *full period* to an
   integer number of frames and allow an ASYMMETRIC duty cycle:

        20 Hz @ 60 Hz -> period 3 frames -> 2 on / 1 off
        FFT of [1,1,0] repeated at 60 fps -> dominant component 20.00 Hz  (verified)

   A square wave's fundamental is set by its period, not by its symmetry, so an
   asymmetric duty cycle still drives a clean SSVEP at the intended frequency.

FRAME-COUNTED, NOT TIME-BASED
-----------------------------
State is derived from an integer frame counter (`frame % period < on_frames`), never
from `time.time()`. Wall-clock timers drift against the display's actual refresh and
smear the stimulus spectrum -- exactly the jitter Gap 5 exists to remove.
"""

import time
from collections import deque

import pygame


# --------------------------------------------------------------------- helpers
def detect_refresh_rate(default=60.0):
    """Best-effort monitor refresh rate in Hz."""
    try:
        info = pygame.display.Info()
        rate = getattr(info, "refresh_rate", 0) or 0
        if rate and rate > 0:
            return float(rate)
    except Exception:
        pass
    try:
        modes = pygame.display.get_desktop_refresh_rates()
        if modes and modes[0] > 0:
            return float(modes[0])
    except Exception:
        pass
    return float(default)


def measure_refresh_rate(surface_flip, n_frames=120, warmup=20):
    """Empirically time actual flips. Returns (measured_hz, frame_times_ms)."""
    for _ in range(warmup):
        surface_flip()
    times = []
    last = time.perf_counter()
    for _ in range(n_frames):
        surface_flip()
        now = time.perf_counter()
        times.append((now - last) * 1000.0)
        last = now
    mean_ms = sum(times) / len(times)
    return (1000.0 / mean_ms if mean_ms > 0 else 0.0), times


def open_display(size, refresh_hint=None, resizable=False, vsync=True):
    """Open a V-synced, blit-compatible display.

    Returns (surface, info_dict). Falls back gracefully if vsync is unavailable
    (common on virtual/headless drivers) and reports what actually happened.
    """
    flags = pygame.DOUBLEBUF | pygame.SCALED
    if resizable:
        flags |= pygame.RESIZABLE
    info = {"vsync": False, "flags": "DOUBLEBUF|SCALED", "fallback": None}
    surf = None
    if vsync:
        try:
            surf = pygame.display.set_mode(size, flags, vsync=1)
            info["vsync"] = True
        except Exception as exc:
            info["fallback"] = f"vsync unavailable ({exc})"
    if surf is None:
        try:
            surf = pygame.display.set_mode(size, flags)
        except Exception as exc:
            info["fallback"] = f"SCALED unavailable ({exc})"
            surf = pygame.display.set_mode(size, pygame.DOUBLEBUF)
            info["flags"] = "DOUBLEBUF"
    info["refresh"] = refresh_hint or detect_refresh_rate()
    return surf, info


# ------------------------------------------------------------------ flicker
class FlickerPlan:
    """Frame-quantised schedule for one flicker frequency.

    period_frames = round(refresh / freq); on_frames = ceil(period/2).
    The realised frequency is refresh / period_frames -- report this, not the request.
    """

    __slots__ = ("requested", "refresh", "period_frames", "on_frames", "phase_offset")

    def __init__(self, requested_hz, refresh_hz, phase_offset=0):
        self.requested = float(requested_hz)
        self.refresh = float(refresh_hz)
        self.period_frames = max(2, int(round(refresh_hz / requested_hz)))
        self.on_frames = (self.period_frames + 1) // 2      # ceil -> asymmetric ok
        self.phase_offset = int(phase_offset)

    @property
    def actual(self):
        return self.refresh / self.period_frames

    @property
    def error_hz(self):
        return self.actual - self.requested

    @property
    def error_pct(self):
        return 100.0 * self.error_hz / self.requested if self.requested else 0.0

    @property
    def duty(self):
        return self.on_frames / self.period_frames

    @property
    def is_exact(self):
        return abs(self.error_hz) < 1e-9

    @property
    def is_symmetric(self):
        return self.period_frames % 2 == 0

    def is_on(self, frame):
        return ((frame + self.phase_offset) % self.period_frames) < self.on_frames

    def describe(self):
        sym = "symmetric" if self.is_symmetric else "asymmetric"
        flag = "exact" if self.is_exact else f"{self.error_pct:+.2f}% ERROR"
        return (f"{self.requested:5.1f} Hz -> {self.period_frames} frames "
                f"({self.on_frames} on/{self.period_frames - self.on_frames} off, "
                f"{sym}, duty {self.duty:.2f}) = {self.actual:6.2f} Hz [{flag}]")

    def __repr__(self):
        return f"<FlickerPlan {self.requested}Hz->{self.actual:.2f}Hz>"


def suggest_frequencies(refresh_hz, lo=13.0, hi=32.0, max_n=6):
    """Frequencies that are EXACTLY renderable at this refresh, outside the alpha band.

    Use this when moving to a >2-command speller: picking exactly-renderable targets
    costs nothing and removes a whole class of stimulus error.
    """
    out = []
    for period in range(2, int(refresh_hz) + 1):
        f = refresh_hz / period
        if lo <= f <= hi:
            out.append(round(f, 4))
    out.sort(reverse=True)
    return out[:max_n]


class StimulusEngine:
    """Owns the frame counter and per-target flicker state, and audits jitter."""

    def __init__(self, target_freqs=(15.0, 20.0), refresh_hz=60.0, history=600,
                 phase_offsets=None):
        self.refresh = float(refresh_hz)
        offsets = phase_offsets or [0] * len(target_freqs)
        self.plans = [FlickerPlan(f, refresh_hz, o)
                      for f, o in zip(target_freqs, offsets)]
        self.frame = 0
        self._times = deque(maxlen=history)
        self._last = None
        self.dropped_frames = 0

    # ------------------------------------------------------------- per frame
    def tick(self):
        """Advance one rendered frame. Call once immediately after display.flip()."""
        self.frame += 1
        now = time.perf_counter()
        if self._last is not None:
            dt = (now - self._last) * 1000.0
            self._times.append(dt)
            # a frame taking >1.5 nominal periods means the flip missed a refresh
            if dt > 1.5 * (1000.0 / self.refresh):
                self.dropped_frames += 1
        self._last = now

    def is_on(self, index):
        return self.plans[index].is_on(self.frame)

    def states(self):
        return [p.is_on(self.frame) for p in self.plans]

    def reset(self):
        self.frame = 0
        self._times.clear()
        self._last = None
        self.dropped_frames = 0

    # -------------------------------------------------------------- auditing
    def timing_report(self):
        """Frame-time statistics -- the evidence that flicker is jitter-free."""
        if len(self._times) < 2:
            return None
        ts = list(self._times)
        n = len(ts)
        mean = sum(ts) / n
        var = sum((t - mean) ** 2 for t in ts) / n
        sd = var ** 0.5
        nominal = 1000.0 / self.refresh
        srt = sorted(ts)
        return {
            "frames": self.frame,
            "measured_fps": 1000.0 / mean if mean else 0.0,
            "nominal_fps": self.refresh,
            "mean_ms": mean,
            "nominal_ms": nominal,
            "sd_ms": sd,
            "jitter_pct": 100.0 * sd / nominal if nominal else 0.0,
            "min_ms": srt[0],
            "max_ms": srt[-1],
            "p99_ms": srt[min(n - 1, int(0.99 * n))],
            "dropped": self.dropped_frames,
            "drop_rate": self.dropped_frames / max(1, self.frame),
        }

    def print_report(self):
        print("\nFlicker plan")
        print("-" * 74)
        print(f"  monitor refresh: {self.refresh:.1f} Hz")
        for p in self.plans:
            print("  " + p.describe())
        bad = [p for p in self.plans if not p.is_exact]
        if bad:
            print(f"  !! {len(bad)} target(s) NOT exactly renderable at this refresh.")
            print(f"     Exactly renderable nearby: {suggest_frequencies(self.refresh)}")
        dup = {p.period_frames for p in self.plans}
        if len(dup) < len(self.plans):
            print("  !! COLLISION: two targets share a frame period -- they will look\n"
                  "     identical and the classifier cannot separate them.")

        r = self.timing_report()
        if r:
            print("\nFrame timing (V-sync audit)")
            print("-" * 74)
            print(f"  frames rendered  : {r['frames']}")
            print(f"  measured fps     : {r['measured_fps']:.2f} "
                  f"(nominal {r['nominal_fps']:.1f})")
            print(f"  frame time       : {r['mean_ms']:.3f} ms "
                  f"(nominal {r['nominal_ms']:.3f} ms)")
            print(f"  jitter (sd)      : {r['sd_ms']:.3f} ms "
                  f"({r['jitter_pct']:.1f}% of a frame)")
            print(f"  min / p99 / max  : {r['min_ms']:.2f} / {r['p99_ms']:.2f} "
                  f"/ {r['max_ms']:.2f} ms")
            print(f"  dropped frames   : {r['dropped']} ({r['drop_rate']:.2%})")

    # ------------------------------------------------------------ validation
    def spectrum_check(self, index, n_frames=1200):
        """Simulate the on/off sequence and FFT it.

        Confirms the rendered square wave's dominant component really is the
        intended frequency -- this is what caught the 20 Hz -> 15 Hz collapse.
        """
        import numpy as np
        p = self.plans[index]
        seq = np.array([1.0 if p.is_on(f) else 0.0 for f in range(n_frames)])
        seq -= seq.mean()
        spec = np.abs(np.fft.rfft(seq))
        freqs = np.fft.rfftfreq(n_frames, 1.0 / self.refresh)
        peak = float(freqs[int(np.argmax(spec))])
        return {"requested": p.requested, "planned": p.actual, "fft_peak": peak,
                "matches": abs(peak - p.actual) < (self.refresh / n_frames) * 2}
