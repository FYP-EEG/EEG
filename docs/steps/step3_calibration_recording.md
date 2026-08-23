# Step 3 — Subject-specific calibration recording (with idle & distraction)

**Covers:** `instructions.md` Gap 2 item 3 (calibration sequence) + blocker **B6**
(labelled stimulus recorder) from `docs/instructions_verification.md`.
**Status:** ✅ code complete and validated end-to-end on the simulate backend.
🧪 **Requires your hardware to produce the real profile.**
**Tests:** 27 assertions in `tests/test_step3_calibration.py`, all passing.

---

## Why this step is the highest-value one

Public benchmarks (Tsinghua, PhysioNet) contain **only attentive trials** — the subject is
always being asked to do something. They contain no "user isn't trying" class. But a GUI runs
continuously and the user is idle most of the wall-clock time, so the failure mode that
actually breaks the product — false positives during idle — is exactly the one public data
**cannot** measure or calibrate against.

`EEG/data_record.py` flashes typed characters; it cannot record "user stared at the 15 Hz tile
for 5 s" or "user was reading and not selecting". That recorder is what Step 3 adds.

---

## What was built

### 1. `EEG/calibration_record.py` — protocol-driven labelled recorder

```
  target_0           class= 0  10 x  5.0s  ->   30 windows
  target_1           class= 1  10 x  5.0s  ->   30 windows
  idle               class=-1   5 x 30.0s  ->  140 windows
  idle_distracted    class=-1   3 x 30.0s  ->   84 windows
  artifact           class= 2   4 x 10.0s  ->   32 windows
  TOTAL 6.3 min of recording, 316 windows (3.0s each, 1.0s hop)
```

Note the deliberate imbalance: **240 s of idle vs 100 s of target**. That ratio mirrors real
GUI usage and is the whole point — most of the data teaches the system what *not* to react to.

- **Two backends.** `--backend simulate` (default, no hardware, lets you rehearse the whole
  protocol and validate the pipeline) and `--backend brainflow` (live Cyton). BrainFlow is
  imported lazily, so the module runs without it installed.
- **Stimulus UI** with frame-counted flicker, `DOUBLEBUF | SCALED, vsync=1` — a working preview
  of Step 5, and **not** the `OPENGL|DOUBLEBUF` from `instructions.md` that would break blits.
  At 60 Hz: 15 Hz toggles every 2 frames, 20 Hz every 1.5 (rounded).
- **Distraction block** renders real prose to read, so "distracted" is a genuine cognitive
  state rather than an empty screen.
- Sliding-window epoching (750 samples, 250 hop) matching the engine exactly.
- `--dry-run` prints the protocol, `--quick` shortens it for smoke tests, `ESC` aborts and
  still saves partial data.
- **Photosensitive-epilepsy warning** printed with explicit confirmation required before a
  live run.

Output: `dataset/calib_<subject>_<timestamp>.npz` with `X, y, labels, fs, montage,
target_freqs, channel_labels`.

### 2. `ML/calibration.py` — fits the profile and reports GUI metrics

Real output from the validated simulate run:

```
CONFIDENCE SEPARATION (this is what makes NO_ACTION possible)
  attentive  n=  60  mean=0.4316  sd=0.2315  p05=0.1092
  idle       n= 224  mean=0.0593  sd=0.0186  p95=0.0891
  separation (Cohen's d) = 2.27
  artifact windows rejected: 32/316  (blink block caught: 32/32)

GUI USABILITY METRICS  (accuracy alone does not predict these)
   pct   thresh  FP/min idle  precision   recall  ITR b/min
    80   0.0726         0.27      95.2%    33.3%       20.0
    90   0.0836         0.27      95.2%    33.3%       20.0
    95   0.0891         0.27      95.2%    33.3%       20.0
  97.5   0.0978         0.00     100.0%    31.7%       19.0
    99   0.1183         0.00     100.0%    23.3%       14.0
```

This is the metrics table from `docs/accuracy_strategy.md` §2 made operational: you pick a
threshold by the **false-positives-per-minute** you can tolerate, not by accuracy. Note the
p97.5 row — 0 FP/min at 100% precision, costing only ~2 points of recall.

`--sweep` shows the trade-off; the chosen threshold is written to
`dataset/profile_<subject>.json`.

### 3. `BCIEngine.from_profile(path)`

```python
eng = BCIEngine.from_profile("dataset/profile_calib_S01_....json")
```
The threshold now comes from *this user's own idle recording*, replacing the
simulation-derived 0.15 placeholder from Step 2.

---

## 🐛 Real bug this step uncovered

The first calibration run reported **`blink block caught: 0/32`** — artifact detection was
completely broken, and Step 2's tests had not caught it.

**Cause:** blinks are 0.5–3 Hz slow waves, but `predict_proba()` ran `artifact_detection()` on
data already through the **6 Hz SSVEP highpass**, which deletes them. Measured on a real
recorded blink window:

```
RAW peak Fp1/Fp2      : 221.6 uV     <- well over the 100 uV threshold
after 6-88 Hz filter  :  60.2 uV     <- under it. Never fires.
threshold             : 100.0
```

Step 2's test passed only because `sim_stream.inject_blink()` adds a *square* 180 µV step,
which has enough broadband energy to survive the highpass. A realistic Hanning-shaped blink
does not. **The synthetic test was easier than reality** — exactly the gap this step exists to
close.

**Fix:** `artifact_detection(data, is_raw=True)` now applies its own 0.5–8 Hz band, and
`predict_proba()` checks the **raw** signal *before* the SSVEP filter. Result:

```
artifact     artifact-detected 32/32
idle         artifact-detected 0/140
target_0     artifact-detected 0/30
```

Step 1 (27) and Step 2 (23) regressions still pass.

---

## 🧪 How to run the real recording

```bash
# 1. rehearse without hardware
python EEG/calibration_record.py --dry-run
python EEG/calibration_record.py --subject S01 --backend simulate

# 2. real session (Cyton on COM4)
python EEG/calibration_record.py --subject S01 --backend brainflow --serial-port COM4

# 3. fit the profile
python ML/calibration.py --recording "dataset/calib_S01_*.npz" --sweep

# 4. use it
#    BCIEngine.from_profile("dataset/profile_calib_S01_....json")
```

**Recording checklist**
- Dim, consistent room lighting; no other flickering light sources.
- Electrodes on O1/O2 (SSVEP) and Fp1/Fp2 (blinks) per `ML/montage.py:CYTON_8_SSVEP`.
- Check impedance before starting; re-gel anything noisy.
- Seat the participant ~60 cm from the screen, same distance every session.
- Run the idle blocks **honestly** — actually rest and let attention wander. Idle data that is
  secretly attentive will set the threshold too high and make the speller unresponsive.
- Re-run calibration per session; electrode placement drifts day to day.

⚠️ **Photosensitive epilepsy:** do not run with anyone who has a seizure history. Stop on any
discomfort, dizziness, or visual disturbance.

---

## Files

| File | Change |
|---|---|
| `EEG/calibration_record.py` | **new** — protocol, 2 backends, stimulus UI, safety gate |
| `ML/calibration.py` | **new** — threshold fitting, FP/min, precision, ITR, profile export |
| `ML/bci_engine.py` | `from_profile()` classmethod |
| `ML/hybrid_classifier.py` | artifact band fix (`is_raw`, 0.5–8 Hz) |
| `tests/test_step3_calibration.py` | **new** — 27 assertions incl. the blink regression |
| `docs/calib_target.png`, `docs/calib_distracted.png` | stimulus screenshots |

## Caveat

Every number above comes from the **simulate** backend, which is cleaner and better behaved
than real EEG. It validates the *pipeline*, not your hardware. Expect lower separation, a
higher threshold, and lower recall on real data — that's the point of recording it.

## Next
**Step 4 — acquisition→engine bridge** (blocker B5). `calibration_record.py` already contains a
working BrainFlow read loop; Step 4 generalises it into a threaded ring buffer so the speller
can consume live windows without blocking the UI.
