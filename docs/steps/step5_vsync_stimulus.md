# Step 5 — Hardware-synced visual stimulus engine

**Covers:** `instructions.md` Gap 5 (both items) + correction **A3** from
`docs/instructions_verification.md`.
**Status:** ✅ complete. **Tests:** 37 assertions in `tests/test_step5_stimulus.py`.

---

## Two corrections to the handoff instructions

### A. `pygame.OPENGL | pygame.DOUBLEBUF` would have broken the entire UI
The instructions specify `OPENGL|DOUBLEBUF` with `vsync=1`. That turns the display into an
OpenGL context where `Surface.blit()` no longer works — every `Button` and `TileButton` would
silently stop rendering. The correct form for a 2D blitting app is:

```python
pygame.display.set_mode(size, pygame.DOUBLEBUF | pygame.SCALED, vsync=1)
```
`SCALED` is what actually lets SDL2 honour the vsync flag. A test asserts `blit()` still works.

### B. 🐛 "flash every N frames" silently collapsed 20 Hz onto 15 Hz

The instructions say *"on a 60Hz monitor, a 10Hz target flashes every 6 frames"* — a symmetric
half-period of `refresh/(2f)`. That only holds when `refresh/(2f)` is an integer. On 60 Hz:

```
request 15 Hz -> half_period 2 frames -> ACTUAL 15.00 Hz    ok
request 20 Hz -> half_period 2 frames -> ACTUAL 15.00 Hz    ** WRONG **
```

**Both tiles would have flickered identically at 15 Hz.** The classifier could never separate
them, and — worse — Step 3's calibration recorder used exactly this formula, so every
`target_1` trial would have been recorded at 15 Hz while labelled 20 Hz. That poisons the
threshold fit, the profile, and any TRCA template built from it. Found before any hardware
recording happened.

**Fix:** quantise the **full period** to an integer number of frames and allow an
**asymmetric duty cycle**. A square wave's fundamental is set by its period, not its symmetry:

```
15.0 Hz -> 4 frames (2 on/2 off, symmetric,  duty 0.50) = 15.00 Hz [exact]
20.0 Hz -> 3 frames (2 on/1 off, asymmetric, duty 0.67) = 20.00 Hz [exact]
```

Verified by FFT of the actual rendered sequence, not by assertion:
`target 1: FFT peak 20.00Hz == 20.0Hz`.

---

## Why V-sync matters — measured, not asserted

`tests/exp_jitter_impact.py` renders the stimulus under three timing regimes, converts it to an
evoked response, and runs the **real** FBCCA classifier:

| Rendering | Accuracy | Mean confidence | Rejected as `NO_ACTION` |
|---|---|---|---|
| **A) V-synced, frame-counted (Step 5)** | **100.0%** | **0.753** | **0.0%** |
| B) time-based timer (2 ms sd) | 98.3% | 0.227 | 28.3% |
| C) no vsync / loop stalls (6 ms sd) | 75.0% | 0.071 | **100.0%** |

The accuracy column understates the damage. What jitter really destroys is **confidence**:
0.753 → 0.071. Because Step 2's engine rejects anything below the idle-calibrated threshold,
condition C would reject **every single window** — the speller would appear completely dead
while "accuracy" still read 75%. This is the clearest example yet of why bare accuracy is the
wrong metric for a GUI.

---

## What was built

### `pygame_lib/stimulus.py`

| Component | Purpose |
|---|---|
| `FlickerPlan` | Frame-quantised schedule for one frequency. Reports `actual`, `error_pct`, `duty`, `is_exact`, `is_symmetric`. |
| `StimulusEngine` | Owns the frame counter, per-target state, jitter auditing, FFT self-check. |
| `open_display()` | V-synced, blit-compatible display with graceful fallback; reports what it actually got. |
| `suggest_frequencies()` | Exactly-renderable frequencies outside the alpha band for a given refresh. |
| `measure_refresh_rate()` | Empirically times flips rather than trusting the driver. |

**Frame-counted, never time-based:** state is `(frame + offset) % period < on_frames`. Wall-clock
timers drift against the real refresh and smear the spectrum — precisely the jitter Gap 5 targets.

### Honest reporting on non-60 Hz monitors

```
144 Hz:  15.0 Hz -> 10 frames = 14.40 Hz [-4.00% ERROR]
          20.0 Hz ->  7 frames = 20.57 Hz [+2.86% ERROR]
          Exactly renderable nearby: [28.8, 24.0, 20.5714, 18.0, 16.0, 14.4]
 75 Hz:  20.0 Hz ->  4 frames = 18.75 Hz [-6.25% ERROR]
```

It also **detects collisions** — two targets landing on the same frame period — and says so.

### The classifier now tracks what is *actually* rendered

A subtle but important integration point: on a 144 Hz panel the speller renders 14.40 Hz, so
the CCA reference signals must be built for **14.40 Hz, not 15.0 Hz**. `Speller.actual_freqs`
is passed into `BCIEngine`, and Step 3 recordings are now labelled with the actual rendered
frequencies. Mismatching these would quietly cost accuracy with no visible symptom.

---

## Files

| File | Change |
|---|---|
| `pygame_lib/stimulus.py` | **new** — the stimulus engine |
| `example/bci_speller.py` | `--flicker`, V-synced display, frame-driven tiles, live jitter readout |
| `EEG/calibration_record.py` | **bug fixed** — uses `StimulusEngine`; records actual freqs |
| `tests/test_step5_stimulus.py` | **new** — 37 assertions |
| `tests/exp_jitter_impact.py` | **new** — the jitter experiment |
| `docs/step5_speller_flicker.png` | speller with flicker active |

## Run
```bash
python example/bci_speller.py --flicker
python example/bci_speller.py --flicker --source sim --profile dataset/profile_*.json
python example/bci_speller.py --flicker --refresh 144        # override detection
python tests/test_step5_stimulus.py
python tests/exp_jitter_impact.py
```
On exit the speller prints the flicker plan and a frame-timing audit — **put that table in your
report** as evidence the stimulus was hardware-synced.

⚠️ **Photosensitive epilepsy:** `--flicker` renders 15/20 Hz flashing. Same precautions as Step 3.

## Carried forward
- All timing here was measured on the **dummy SDL driver**, which does not vsync — the numbers
  prove the *logic*, not your monitor. Re-run `--flicker` on the real display and check
  `measured fps ≈ nominal` with `dropped 0%`.
- **Check your monitor's refresh before recording.** On anything other than 60/120 Hz, 15/20 Hz
  are not exactly renderable; use the printed `suggest_frequencies()` list instead.
- Row-scanning currently alternates the two frequencies across rows. Moving to 4–6
  exactly-renderable targets (60 Hz gives 30/20/15) would cut selection time substantially —
  a natural Step 6+ improvement now that the stimulus engine supports arbitrary counts.
