# Step 2 — Stimulus frequencies, confidence scores, and the idle state

**Covers:** `instructions.md` Gap 4 (all three items) + blockers **B1**, **B3**, **B4**, **B2**
from `docs/instructions_verification.md`.
**Status:** ✅ complete — 23 assertions in `tests/test_gap4_idle.py`, all passing.

---

## Why this came before Gap 2's ML refactor

Gap 4 asks for confidence thresholding, but the classifier physically could not provide a
confidence: `cca_ssvep_detection()` ended in `np.argmax(class_scores)` and threw the
correlation values away. Every "idle state" feature was therefore unimplementable until the
classifier returned scores. That made this the true unblocker, ahead of TRCA/FBCSP.

---

## Changes

### 1. Stimulus frequencies 10/12 Hz → **15/20 Hz**  (`ML/hybrid_classifier.py`)
10 Hz sits inside the occipital alpha band (8–13 Hz), which appears whenever the user relaxes
their eyes. The classifier was being asked to separate an *evoked* 10 Hz response from a
*spontaneous* 10 Hz rhythm at the same electrodes — the cause of the 10 Hz bias in the Colab
run (295 vs 125 predictions, 73.10%).

`tests/exp_frequency_choice.py`, 120 simulated trials:

| Condition | 10/12 Hz | 15/20 Hz |
|---|---|---|
| Moderate alpha | 91.7% | **100%** |
| Strong alpha (drowsy) | 80.8% (12 Hz recall **61.7%**) | **100%** |

Both new targets are exact 60 Hz monitor divisors (15 Hz = every 4 frames, 20 Hz = every 3),
so Step 5's frame-accurate flicker becomes trivial.

> Simulated signals are cleaner than real EEG — 100% will not survive contact with hardware.
> The **mechanism** (alpha collision) is what's real and well documented.

### 2. Sub-bands 8/18/28 Hz → **12/22/32 Hz**
The old `low_f = 8.0 + band*10.0` put band 0's edge at 8 Hz, admitting alpha directly.
Filters are now pre-built once in `__init__` instead of per-trial per-class (was 6 Butterworth
designs per window).

### 3. `predict_proba()` → `(label, confidence, scores)`  — blocker B4
```python
label, conf, scores = clf.predict_proba(window)
# label: 0 | 1 | NO_ACTION (-1) | "BLINK"
```
`predict()` is kept as a back-compat wrapper so the Colab notebook still runs.

### 4. `NO_ACTION` — the key change for a distraction-tolerant GUI
When the winning score falls below `confidence_threshold`, the classifier returns
`NO_ACTION` instead of inventing a command. Attentive vs idle scores separate cleanly
(**0.430 ± 0.092** vs **0.053 ± 0.018**, ~4 sd apart).

### 5. `calibrate_threshold(idle_windows, percentile=95)`
Sets the gate from the user's *own* idle recording. This is the per-subject adaptation that
public benchmark data structurally cannot provide, and it is what Step 3 feeds.

### 6. `ML/montage.py` — blocker B1
The Colab run indexes a 64-channel Tsinghua cap (`[53…63]`); the live rig is an 8-channel
Cyton. Those indices would raise a bare `IndexError` deep inside CCA. Channels are now
selected by **montage name** (`cyton8_ssvep`, `benchmark64`, `cyton8_motor`), and
`montage.validate()` raises an explanatory error instead:

> `Montage 'benchmark64' needs >= 64 channels but data has 8. Are you feeding 8-channel Cyton data to a 64-channel config?`

### 7. `BCIEngine` filter bypass fixed — blocker B3
`process_frame()` called `cca_ssvep_detection(raw_trial_data)` directly, so the 1–45 Hz
bandpass **never ran on the live path** — the opposite of what the notebook validated. It now
routes through `predict_proba()`, which filters. Asserted by a test that greps the source.

### 8. Consecutive-run debounce (Gap 4 item 2)
The old `Counter`-based majority vote could fire on 3-of-5 mixed noise. Now a **run** of N
consecutive agreeing windows is required; any disagreement resets the run to zero.

### 9. Artifact lockout + refractory (Gap 4 item 3)
A blink on Fp1/Fp2 pauses output for `artifact_lockout_ms` (default 800 ms). A
`refractory_ms` gap (default 1200 ms) stops one long gaze from repeating a key ten times.
Engine states are now explicit: `NO_ACTION`, `BLINK`, `LOCKOUT`, `BUILDING`, `REFRACTORY`.

### 10. MI unit scaling + band — blocker B2
`mi_model_8ch.joblib` was trained on MNE EDF data in **volts**, band-passed 8–30 Hz.
BrainFlow delivers **microvolts**, and the old `predict()` accepted `auto_filter` and ignored
it. Features were off by ~1e6 and out of band. Now scaled and filtered to match training.

---

## Measured effect

`tests/test_gap4_idle.py`, 150-window session, 30% attentive / 70% idle-distracted:

```
BEFORE (forced argmax):  100% of idle windows emitted a command
                         -> 141/141 garbage keystrokes per 200 windows

AFTER:                   0 false commands / 105 idle windows
                         no_action_rate 70%
                         real commands still dispatched
                         idle -> NO_ACTION on 25/25 windows
```

---

## Files

| File | Change |
|---|---|
| `ML/hybrid_classifier.py` | rewritten — freqs, sub-bands, `predict_proba`, `NO_ACTION`, calibration, MI fixes |
| `ML/bci_engine.py` | rewritten — filtered path, run-debounce, lockout, refractory, `report()` |
| `ML/montage.py` | **new** — electrode→index maps + validation |
| `ML/sim_stream.py` | 8-ch Cyton default, 15/20 Hz, `set_idle()`, realistic occipital alpha |
| `example/bci_speller.py` | consumes engine states; 15 Hz = NEXT, 20 Hz = SELECT |
| `tests/test_gap4_idle.py` | **new** — 23 assertions |
| `tests/exp_frequency_choice.py` | **new** — the alpha-collision experiment |
| `tests/exp_idle_and_metrics.py` | **new** — accuracy-vs-usability experiment |

## Run
```bash
python tests/test_gap4_idle.py        # 23 assertions
python tests/test_speller.py          # 27 assertions (Step 1 regression)
python tests/exp_frequency_choice.py  # reproduces the 10 Hz bias
```

## Carried forward
`--source engine` still cannot run: no acquisition→engine bridge exists (blocker B5, Step 4).
The confidence threshold default (0.15) is a simulation-derived placeholder — **Step 3 replaces
it with a value calibrated from your own idle recording.**
