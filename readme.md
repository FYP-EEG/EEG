# bci_sdk — EEG-controlled button selection for pygame-ce

## 🥜 In a nutshell

**We are building a way to click buttons by looking at them.**

Imagine a button on a screen that blinks very fast — 15 times a second. When your eyes look at
it, something surprising happens: **your brain starts pulsing at exactly 15 times a second
too.** It copies the blinking. You cannot feel it, but it really happens.

Now put a headband on with little sensors that listen to the back of your head. The computer
hears that 15-times-a-second pulse and thinks:

> *"They must be looking at the 15 button!"*

So it presses that button for you. **No hands. No mouse. Just looking.**

Put a second button next to it blinking 20 times a second, and now the computer can tell the
two apart — because your brain pulses at whichever one you are staring at.

### Why bother?

Some people cannot move their hands or speak. If looking at something is enough to choose it,
they can talk, play, and control things again.

### What this project actually is

We are not building one app. **We are building the toolbox** so that *other* programmers can
add this to *their* apps in a few lines:

```python
button = bci_sdk.BrainButton("YES", rect=(50, 50, 200, 80), freq=15.0)

@button.on_brain_select
def chosen(btn):
    print("they looked at YES!")
```

The programmer makes the buttons. Our library does the hard part — listening to the brain and
working out which button was chosen.

### The tricky bits (why this is hard)

- 🧠 **The brain is noisy.** The signal we want is tiny, buried under everything else the brain
  is doing. It is like hearing one person tap a spoon during a loud party.
- 😴 **Brains hum at ~10 pulses a second all on their own** when you relax your eyes. So we
  must never use a 10-blink button — the computer could not tell "looking" from "resting".
- 🤫 **The computer must be allowed to say "nothing".** If it always guesses, it will press
  buttons while you are just reading or blinking. Knowing when you are *not* choosing is the
  hardest part.
- 👤 **Every brain is different.** So each person does a 60-second warm-up first, and the
  computer tunes itself to them.

### Where we are now

The whole toolbox is **built**, and it works on pretend (computer-generated) brain signals.

**It has never been tried on a real person's brain yet.** That is the next step — and until
that happens, we genuinely do not know how well it works.

---

A Python library that lets developers add **brain-controlled button selection** to their
pygame-ce applications. The user looks at a flickering button; the library detects which one
and fires that button's callback.

```python
import bci_sdk

session = bci_sdk.BCISession(source="hardware")
button  = bci_sdk.BrainButton("FIRE", rect=(50, 50, 200, 80), freq=15.0)

@button.on_brain_select
def fire(btn):
    print("selected by brain:", btn.label)
```

---

## ⚠️ Project status — read first

This is a **final-year research prototype**, not a working product.

| | |
|---|---|
| Code written | ✅ ~5,300 lines |
| Runs end to end on synthetic data | ✅ |
| **Ever processed a real brain signal** | ❌ **no** |
| **`pip install` works** | ❌ **no — see below** |
| Published to PyPI | ❌ no |

**Two known blockers:**

1. **`pip install` is broken.** `pyproject.toml` declares `packages = ["bci_sdk", "ML"]`, but
   `bci_sdk` also imports from `EEG/` and `pygame_lib/`, which are not packages (no
   `__init__.py`). A built wheel installs and then fails on `import bci_sdk`. It works from a
   source checkout only, via a `sys.path` shim. ~1 hour to fix properly.
2. **No EEG from a human has ever been recorded.** `BrainFlowBackend` was written from
   BrainFlow's documentation and has never been connected to a board. Every performance number
   in `docs/` comes from simulation or replayed data.

The only real measurements in this project are two pre-existing baselines:
**61.40%** (CSP+LDA on PhysioNet) and **73.10%** (CCA on the Tsinghua Benchmark).

See `docs/progress/progress.md` for the honest assessment, and
`docs/LAB_SESSION_CHECKLIST.md` for what to do at the headset.

---

## How it works

The screen flickers a button at a precise frequency. Stare at a 15 Hz flicker and your visual
cortex produces electrical activity at 15 Hz — a real, measurable effect called **SSVEP**
(Steady-State Visually Evoked Potential). Electrodes at the back of your head pick it up, and
the classifier matches the recording against each candidate frequency.

```
flickering button → visual cortex oscillates → electrode → 3 s window
  → blink check → bandpass → filter-bank CCA → confidence scores
  → threshold + debounce → button.on_brain_select() fires
```

Two frequencies = two commands. `docs/explain/CLASSIFIER_EXPLAINED.md` walks through every
step with diagrams.

**Default classifier: FBCCA** (Filter Bank Canonical Correlation Analysis). It is
*training-free* — the reference waves are generated mathematically, so no model file and no
training data are required to get started. TRCA is available as an optional trained
alternative.

---

## Install

No PyPI release yet. Clone and run from source:

```bash
git clone https://github.com/FYP-EEG/EEG.git
cd EEG
pip install -r requirements.txt
```

`pygame-ce` is required for the UI. `brainflow` is only needed for real hardware — everything
else runs without it.

> Uses **pygame-ce**, a drop-in fork of pygame. Install one or the other, never both.

---

## Try it without a headset

```bash
python example/sdk_demo.py                 # 4 brain-controlled buttons, synthetic EEG
python example/bci_app.py                  # user select → calibrate → speller
python example/rockpaperscissors.py        # mouse only; demos Button.flash()
```

Arrow keys work throughout, so the whole system is drivable with no hardware.

---

## Using the library

### `BCISession` — the acquisition + decoding pipeline

```python
session = bci_sdk.BCISession(
    source="hardware",                       # "sim" | "replay" | "hardware"
    profile="profiles/anson/profile.json",   # per-user calibration (recommended)
)
session.on_command(lambda cmd: print(cmd.name, cmd.confidence))
session.start()
while running:
    session.poll()        # NON-BLOCKING, call once per frame
session.stop()
```

`poll()` returns `None` most frames. Windows arrive about once per second, not once per
frame — that is normal, not an error.

### `BrainButton` — the widget

```python
button = bci_sdk.BrainButton("YES", rect=(50, 50, 200, 80), freq=15.0, dwell=1.0)

@button.on_brain_select
def chosen(btn): ...

button.on_focus(lambda b: ...)   # gaze/highlight arrived
button.on_blur(lambda b: ...)    # left before committing
```

`freq` is the flicker frequency and is **required** for direct gaze selection — a button that
does not flicker produces nothing to detect.

### Two selection paradigms

| | `ButtonGroup` (direct gaze) | `Scanner` (row/column) |
|---|---|---|
| Commands needed | one frequency per button | **two**, total |
| Practical maximum | **4–6 buttons** on a 60 Hz monitor | unlimited |
| Speed | fast | slower, but scales |

```python
group = bci_sdk.ButtonGroup(buttons, freqs=bci_sdk.suggest_frequencies(60))
session.on_command(group.handle_command)

scanner = bci_sdk.Scanner([row1, row2, row3])
session.on_command(scanner.handle_command)
```

**The 4–6 button limit is a hard physical constraint**, not an implementation choice. On a
60 Hz monitor only 30 / 20 / 15 Hz are exactly renderable outside the alpha band. More
buttons than that require `Scanner`.

---

## Calibration — per user, every session

The confidence threshold depends on an individual's alpha amplitude and exactly where the
electrodes sit today. A threshold fitted to one person makes the system unresponsive or
trigger-happy for the next.

```bash
# 60-second protocol
python EEG/calibration_record.py --subject anson --backend brainflow \
       --serial-port COM4 --protocol quick --out profiles/anson

python ML/calibration.py --recording "profiles/anson/calib_*.npz" --sweep
```

```python
p = bci_sdk.UserProfile("anson")
session = bci_sdk.BCISession(source="hardware", **p.as_session_kwargs())
```

The protocol deliberately spends **half its time recording idle**, because the hardest
problem is not detecting a frequency — it is knowing when the user *isn't* trying.

---

## Recording labelled data from a real app

`EEG/data_record.py` runs any script from `example/`, cues a target button with a ripple, and
records EEG with markers around each trial:

```bash
python EEG/data_record.py --file rockpaperscissors --trials 20 --cue 5
python EEG/data_record.py --file rockpaperscissors --dry-run    # rehearse, no EEG
python EEG/data_record.py --list
```

It writes a CSV recording what was cued versus what was actually clicked. **Trials where they
disagree should be dropped during analysis** — the user wasn't attending to the cued target,
so that EEG is mislabelled.

> Note: `Button.flash()` is a *cosmetic* ripple for cueing and click feedback. It is not an
> SSVEP stimulus. Frequency-accurate flicker lives in `pygame_lib/stimulus.py`
> (`StimulusEngine`), which is frame-counted and V-synced.

---

## Things that will bite you

**The system is allowed to say nothing.** `NO_ACTION` is a normal state. A user reading the
screen should produce zero events. Don't build a UI that assumes a command is always coming.

**Match your stimulus to your monitor.** On a 60 Hz display only 30/20/15 Hz are exactly
renderable. Requesting 12 Hz silently gets you 15 Hz.
```python
bci_sdk.suggest_frequencies(60.0)      # -> [30.0, 20.0, 15.0]
stim.print_report()                    # flags inexact targets and collisions
```
Build your classifier around the **actual** rendered frequency, not the requested one.

**Avoid 8–13 Hz.** That is the occipital alpha band — the brain produces ~10 Hz spontaneously
whenever the eyes relax. A 10 Hz target is indistinguishable from resting alpha. This project
measured exactly that failure: 73.10% accuracy with a heavy 10 Hz bias.

**Never block the render loop.** SSVEP depends on frame-accurate flicker; a stalled loop
corrupts the signal you are trying to read. Acquisition already runs on its own thread.

**⚠️ Photosensitive epilepsy.** Flickering stimuli can trigger seizures. Screen participants,
warn them, and let them stop at any time.

---

## Repository layout

| Path | Contents |
|---|---|
| `bci_sdk/` | public API — `BCISession`, `BrainButton`, `Scanner`, `UserProfile` |
| `ML/` | classifiers (FBCCA, TRCA, CSP/FBCSP/Riemannian), engine, calibration, dataset loaders |
| `EEG/` | acquisition — stream bridge, BrainFlow backend, calibration + cued recorders |
| `pygame_lib/` | UI primitives — `Button`, `TileButton`, V-synced `StimulusEngine` |
| `example/` | `sdk_demo` (reference app), `bci_app` (calibration + speller), `rockpaperscissors` |
| `notebooks/` | Colab training script for public datasets |
| `tools/` | `sync_to_drive.py` — stage files for Colab |
| `docs/` | design documents, progress report, lab checklist |

`example/bci_speller.py` is a 53-key EEG-to-Text keyboard, kept as a **backup / reference
implementation**. `sdk_demo.py` is the better example of the library's intended use.

---

## Tests

```bash
python tests/test_speller.py                 # 27
python tests/test_gap4_idle.py               # 23
python tests/test_step3_calibration.py       # 27
python tests/test_step4_bridge.py            # 33
python tests/test_step5_stimulus.py          # 37
python tests/test_step6_models.py            # 38
python tests/test_step7_sdk.py               # 63
python tests/test_step8_datasets_models.py   # 34
python tests/test_step9_profiles_app.py      # 29
```

**These verify that the code executes and its internal contracts hold — nothing more.** They
say nothing about whether the system works on a real brain. Only a hardware session can
establish that.

`tests/exp_*.py` are experiments, not pass/fail tests. They generate the evidence tables
behind the design decisions documented in `docs/`.

---

## Documentation

| Document | Purpose |
|---|---|
| `docs/progress/progress.md` | full progress report with charts and time estimates |
| `docs/DEVELOPER_GUIDE.md` | learn the codebase in about an hour |
| `docs/explain/CLASSIFIER_EXPLAINED.md` | how the classifier works, from zero |
| `docs/LAB_SESSION_CHECKLIST.md` | what to do at the headset |
| `docs/HARDWARE_CYTON_HEADBAND.md` | wiring for Cyton + EEG Headband Kit |
| `docs/GOOGLE_DRIVE_SETUP.md` | what to upload for Colab training |

---

## Hardware

Built for **OpenBCI Cyton** (8 channels, 250 Hz) with the **EEG Headband Kit**.

**Channels 6 and 7 must be occipital (O1/O2).** OpenBCI's default headband wiring puts P8 on
channel 6, which has almost no SSVEP. Channels 0 and 1 are Fp1/Fp2, used for blink detection.
See `docs/HARDWARE_CYTON_HEADBAND.md`.

## Authors

Anson Li · Brian Chan— Final Year Project
