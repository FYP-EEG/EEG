# What to upload to Google Drive for model training

Colab has **no display and no headset**, so it only needs the data and the maths.
Uploading the UI layer just slows down mounting and invites import errors.

---

## Layout

Everything sits at the **MyDrive root**, matching the layout your existing notebook
already uses (`sys.path.insert(0, '/content/drive/MyDrive')`).

```
MyDrive/
├── ML/                          ← UPLOAD (the whole package)
│   ├── __init__.py
│   ├── datasets.py              ← loaders for .mat / .edf / .npz
│   ├── hybrid_classifier.py     ← FBCCA + artifact detection
│   ├── trca.py                  ← TRCA + save()/load()
│   ├── mi_advanced.py           ← FBCSP / Riemannian (MI only)
│   ├── evaluate.py              ← grouped CV (prevents leakage)
│   ├── montage.py               ← electrode maps
│   └── bci_engine.py            ← (import target of __init__)
│
├── benchmark_downloads/         ← UPLOAD if pre-training SSVEP
│   ├── S1.mat … S35.mat
│   └── Freq_Phase.mat           ← REQUIRED to verify your freq_map
│
├── mi_eegmmidb/                 ← UPLOAD only if training Motor Imagery
│   └── S001/S001R03.edf …
│
├── anson/                       ← UPLOAD your own data (one folder PER USER)
│   ├── calib_anson_*.npz        ← from example/bci_app.py
│   └── profile.json             ← fitted threshold
│
├── notebooks/
│   └── train_ssvep_colab.py     ← UPLOAD (the training script)
│
└── models/                      ← OUTPUT: .joblib lands here, download it
```

## Upload / don't upload

| Path | Upload? | Why |
|---|---|---|
| `ML/` | ✅ **yes** | at the **Drive root** — training code *and* the classes needed to load the model |
| `notebooks/train_ssvep_colab.py` | ✅ yes | the script itself |
| `profiles/<user>/` | ✅ yes | your own data — the most valuable input |
| Benchmark `S*.mat` + `Freq_Phase.mat` | ✅ if pre-training | ~1 GB; `Freq_Phase.mat` is mandatory to verify labels |
| PhysioNet `*.edf` | ⬜ only for MI | MI is not on the demo path (see Step 6) |
| `bci_sdk/` | ❌ no | runtime façade; imports pygame |
| `pygame_lib/` | ❌ no | UI; Colab has no display |
| `example/` | ❌ no | apps |
| `EEG/stream_bridge.py`, `calibration_record.py` | ❌ no | need hardware/display |
| `tests/` | ❌ no | run locally |

> `ML/mi_synth.py` and `ML/sim_stream.py` are optional — only if you want to
> reproduce the Step 6 experiments in Colab.

---

## Stage it automatically

```bash
python tools/sync_to_drive.py --out ~/GoogleDrive/MyDrive --user anson
```
Copies `ML/`, the training script and `profiles/anson/` into the right places, and
skips the UI layer. Benchmark `.mat` and PhysioNet `.edf` you place by hand (too large).

## Run it

```python
# Colab cell
!pip install numpy scipy scikit-learn joblib
%run /content/drive/MyDrive/notebooks/train_ssvep_colab.py
```

The script does `sys.path.insert(0, '/content/drive/MyDrive')`, so `import ML` resolves
to `MyDrive/ML/` — the same convention as your existing notebook. To train for a
different person, set `BCI_USER` first:

```python
import os; os.environ["BCI_USER"] = "brian"
```

Then download `MyDrive/models/ssvep_trca_8ch.joblib` and use it locally:

```python
bci_sdk.BCISession(source="hardware", model="ssvep_trca_8ch.joblib")
```

---

## Three traps that will waste your afternoon

**1. Channel count.** The Benchmark is a 64-channel cap; your Cyton has 8. A model
trained on 64 channels **cannot** run on your headset. `train_ssvep_colab.py` calls
`pick_channels(Xp, BENCHMARK_PICK)` to reduce public data to your 8 electrodes first.
Confirm `BENCHMARK_PICK` matches how you actually wired the board.

**2. Frequency coverage.** The Benchmark's 40 targets span **8.0–15.8 Hz**. If you
flicker at 15/20 Hz, **20 Hz does not exist in the public data** — you can only
pre-train the 15 Hz class from it. This is not a flaw in the pipeline; it is why your
own recording is required, not optional.

**3. `freq_map` is now derived, not guessed.** The script calls `build_freq_map()` on
your `Freq_Phase.mat`:
```python
freq_map, missing = build_freq_map("Freq_Phase.mat", (15.0, 20.0))
# -> {7: 0}   missing: [20.0]
```
This matters: an earlier version of this script hard-coded `{35: 0}` for 15 Hz, but
index 35 is **11.8 Hz**. That would have trained on mislabelled data and produced a
meaningless accuracy figure. Never hard-code these indices.

---

## Which model should you actually train?

From Step 6's measurements, on **your current 8-channel montage**:

- **TRCA scored 57.6% vs FBCCA's 100%** — because TRCA is a spatial filter and
  `cyton8_ssvep` gives it only 2 occipital channels (O1, O2) to work with.
- **FBCCA needs no training at all.** It is the default and currently the better choice.

So: train a model **only** if you have added occipital electrodes (Oz, PO3, PO4), or
if you want the per-subject threshold. If you just want a working speller today, run
Step 3 calibration and use the resulting `profile_*.json` with plain FBCCA — no Drive,
no Colab, no training.

```python
bci_sdk.BCISession(source="hardware", profile="dataset/profile_calib_S01_*.json")
```
