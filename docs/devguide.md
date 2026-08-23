# Developer guide — understanding this codebase

For a new team member, or a third-party developer who wants brain-controlled buttons
in their own app. Written to be read top to bottom in about an hour.

---

## 1. The one-paragraph mental model

A screen shows **flickering tiles**. When you stare at a tile flickering at 15 Hz, your
visual cortex produces electrical activity at 15 Hz (an **SSVEP**). Electrodes at the
back of your head (O1, O2) pick it up. The software slices that signal into 3-second
windows, correlates each window against 15 Hz and 20 Hz reference patterns, and if one
matches *confidently enough, several times in a row*, it fires a UI event. Two
frequencies give two commands — enough to drive a scanning keyboard.

**The hard part is not detecting a frequency. It is knowing when the user isn't trying.**
Most of the design exists to avoid typing garbage while someone reads the screen.

---

## 2. The data path

```
  eyes                                                        UI event
   │                                                              ▲
   ▼                                                              │
[flickering tiles]  StimulusEngine (frame-accurate, V-synced)     │
   │                                                              │
   ▼                                                              │
[EEG electrodes] → BrainFlowBackend ─┐                            │
                   SyntheticBackend ─┼→ StreamBridge ─→ BCIEngine ┘
                   ReplayBackend    ─┘  (thread +        (filter, classify,
                                         ring buffer,     confidence gate,
                                         750/250 windows) debounce, lockout)
```

Read those five boxes and you understand the system.

| Stage | File | One-line job |
|---|---|---|
| Stimulus | `pygame_lib/stimulus.py` | flash tiles at *exactly* the right Hz |
| Acquisition | `EEG/stream_bridge.py` | samples → overlapping windows, off-thread |
| Classification | `ML/hybrid_classifier.py` | window → (class, confidence) or `NO_ACTION` |
| Decision | `ML/bci_engine.py` | confidence gate + debounce + blink lockout |
| Application | `bci_sdk/` | `Command` → your callback |

---

## 3. Read the code in this order

**Hour 1 — get it running**
```bash
pip install -e ".[ui]"
python example/bci_app.py           # THE APP: users -> calibrate -> speller
python example/sdk_demo.py          # 4 buttons, synthetic EEG, no hardware
python example/bci_speller.py       # the speller page on its own
```

`bci_app.py` is the shipping entry point: it always opens on user selection and will
not let an uncalibrated user reach the speller, because the confidence threshold is
personal.

**Then read, in order:**

1. **`bci_sdk/session.py`** (~300 lines) — start here. `BCISession.poll()` is the
   entire runtime in 30 lines: get a window, classify, fire callbacks.
2. **`ML/bci_engine.py`** (~200) — `process_frame()`. Every guard against false
   positives lives here, each one commented with why.
3. **`ML/hybrid_classifier.py`** (~240) — `predict_proba()`. The actual signal
   processing: bandpass → filter bank → CCA → confidence.
4. **`EEG/stream_bridge.py`** (~545) — `RingBuffer` then `_emit_ready()`. How a
   continuous stream becomes discrete overlapping windows.
5. **`pygame_lib/stimulus.py`** (~300) — `FlickerPlan`. Why 20 Hz is "2 frames on,
   1 frame off" and not a symmetric blink.
6. **`bci_sdk/widgets.py`** (~290) — `Scanner`. How two commands address 53 keys.

Everything else (`calibration.py`, `evaluate.py`, `trca.py`, `mi_advanced.py`,
`datasets.py`) is offline tooling — read it when you need it.

---

## 4. Five concepts that explain most design decisions

### `NO_ACTION` is a first-class answer
`argmax` always returns *something*. In a GUI that means typing a character every time
the user blinks or thinks. The classifier returns `NO_ACTION` when the winning score is
below a threshold. **Measured: forced argmax emitted a command in 100% of idle windows;
with the gate, 0 out of 105.**

Corollary: `session.poll()` returning `None` is the normal, healthy case.

### Confidence, not just a label
`predict_proba()` returns `(label, confidence, scores)`. Everything downstream —
thresholding, debouncing, idle rejection — needs the score. An earlier version threw it
away inside `np.argmax`, which made the whole idle-handling layer impossible to build.

### Debounce = *consecutive* agreement
Not a majority vote (3-of-5 noise can win that). N windows in a row must agree, and any
disagreement resets the counter to zero.

### Frame counts, never wall-clock timers
Flicker state is `(frame + offset) % period < on_frames`. Timers drift against the
monitor's real refresh and smear the frequency you're trying to produce. **Measured:
6 ms of jitter dropped mean confidence from 0.753 to 0.071 — every window below
threshold, i.e. a speller that appears dead.**

### Avoid 8–13 Hz
That's the occipital alpha band — your brain produces ~10 Hz *spontaneously* whenever
you relax your eyes. A 10 Hz target is indistinguishable from resting alpha. The
original 10/12 Hz design measured 73.1% with a heavy 10 Hz bias; 15/20 Hz fixed it.

---

## 5. Add brain control to your own app

```python
import bci_sdk

session = bci_sdk.BCISession(source="sim")     # "hardware" when you have a headset

button = bci_sdk.BrainButton("LIGHTS", rect=(50, 50, 200, 80))

@button.on_brain_select
def lights_on(btn):
    print("selected", btn.label)

scanner = bci_sdk.Scanner([[button, other_button]])
session.on_command(scanner.handle_command)

with session:
    while running:
        session.poll()          # once per frame, non-blocking
        scanner.update()        # dwell timers
        scanner.update_highlight()
        scanner.draw(screen)
```

`example/sdk_demo.py` is exactly this, complete, using no internal imports.

**Choosing a paradigm:**

| | `ButtonGroup` (direct gaze) | `Scanner` (row/column) |
|---|---|---|
| Commands needed | one frequency per button | **two**, total |
| Practical max | 4–6 buttons at 60 Hz | unlimited |
| Speed | fast | slower, but scales |

---

## 6. Calibration and models — what you actually need

Three levels. **Most people only need level 1.**

**Level 1 — nothing (works today).** FBCCA is training-free.
```python
bci_sdk.BCISession(source="hardware")
```

**Level 2 — per-user calibration (recommended).** 6 minutes, fits the confidence
threshold to *your* brain and *your* electrode placement. Easiest via the app:
```bash
python example/bci_app.py --user anson --source hardware --serial-port COM4
```
Recordings land in `profiles/anson/`, which mirrors the Drive layout. In code:
```python
p = bci_sdk.UserProfile("anson")
bci_sdk.BCISession(source="hardware", **p.as_session_kwargs())
```

**Level 3 — a trained TRCA model (only with more electrodes).**
```python
bci_sdk.BCISession(source="hardware", model="ssvep_trca_8ch.joblib")
```
See `docs/GOOGLE_DRIVE_SETUP.md` (`ML/` goes at the Drive root). **But note Step 6's measurement: on the current
8-channel montage TRCA scored 57.6% against FBCCA's 100%**, because only O1/O2 are
occipital and TRCA is a spatial filter with nothing to work with. Add Oz/PO3/PO4 before
bothering with level 3.

---

## 7. Testing and evidence

```bash
python tests/test_speller.py                 # 27   Step 1
python tests/test_gap4_idle.py               # 23   Step 2
python tests/test_step3_calibration.py       # 27   Step 3
python tests/test_step4_bridge.py            # 33   Step 4
python tests/test_step5_stimulus.py          # 37   Step 5
python tests/test_step6_models.py            # 38   Step 6
python tests/test_step7_sdk.py               # 63   Step 7
python tests/test_step8_datasets_models.py   # 34   datasets + model I/O
python tests/test_step9_profiles_app.py      # 29   profiles + app flow
```
**311 assertions.** All run headless (`SDL_VIDEODRIVER=dummy`), no hardware.

`tests/exp_*.py` are **experiments**, not tests — they print evidence tables behind the
design choices, and they're the material for your report:

| Experiment | Question answered |
|---|---|
| `exp_frequency_choice.py` | why not 10/12 Hz |
| `exp_idle_and_metrics.py` | why accuracy is the wrong metric |
| `exp_jitter_impact.py` | why V-sync matters |
| `exp_leakage.py` | why overlapping windows need grouped CV |
| `exp_trca_channels.py` | why TRCA needs more electrodes |

---

## 8. Debugging

| Symptom | Likely cause | Check |
|---|---|---|
| No commands ever | threshold too high, or user genuinely idle | `session.report()["engine_no_action_rate"]` |
| Random commands | threshold too low, no calibration | run Level 2 calibration |
| One class always wins | targets in the alpha band, or a stimulus collision | `stim.print_report()` |
| Everything is `BLINK` | `artifact_threshold` too low, bad frontal contact | check Fp1/Fp2 impedance |
| `IndexError` in CCA | montage/channel-count mismatch | `montage.validate()` message |
| Speller feels dead | frame jitter, or wrong rendered frequency | `stim.timing_report()` |
| Windows dropped | render loop blocked | `bridge.report()["drop_rate"]` |

Two diagnostics worth memorising:
```python
session.print_report()   # source, fs error, windows, drops, idle rate
stim.print_report()      # flicker plan, exactness, collisions, jitter
```

---

## 9. Conventions

- Shapes are always **(trials, channels, samples)**; a single window is
  **(channels, samples)**. 8 channels, 750 samples, 250 Hz.
- Units are **microvolts** at the boundary. `MotorImageryClassifier` converts to volts
  internally because it was trained on MNE data.
- Artifact detection runs on **raw** data (`is_raw=True`) — blinks are 0.5–3 Hz and the
  6 Hz SSVEP highpass deletes them.
- Never do heavy work in the render loop; acquisition is already threaded.
- Every user callback is wrapped in try/except — a third-party bug must not take down
  an assistive device mid-sentence.

---

## 10. Where to look next

- `docs/steps/README.md` — one design doc per step: what changed, why, what it measured.
- `docs/instructions_verification.md` — audit of the original handoff, incl. three
  instructions that were wrong.
- `docs/accuracy_strategy.md` — why accuracy is the wrong metric, with experiments.
- `docs/ROADMAP.md` — what's done and what remains.

**Honest status:** the whole pipeline is validated end-to-end on synthetic and replayed
data, with 282 assertions. `BrainFlowBackend` is written against the documented API but
**has never run against a physical Cyton**. First contact will need adjustment to
`board_id` and channel ordering. Every performance number in these docs comes from
simulation or replay unless explicitly stated otherwise.
