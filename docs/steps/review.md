# Steps 1–7: summary and readiness review

---

## Part 1 — What each step did

| Step | Delivered | Gap | Tests |
|---|---|---|---|
| **1** | **BCI Speller** — 53-tile keyboard, `TextBuffer`, live output panel, row/column scanning | Gap 1 | 27 |
| **2** | **Confidence + idle state** — 15/20 Hz targets, `predict_proba()`, `NO_ACTION`, run-debounce, blink lockout, montage map | Gap 4 + B1–B4 | 23 |
| **3** | **Calibration recorder** — 6-min protocol (240 s idle vs 100 s target), 2 backends, threshold fitting, FP/min + ITR metrics | Gap 2 (calib) + B6 | 27 |
| **4** | **Stream bridge** — threaded ring buffer, 750/250 windows, oldest-first backpressure, 3 backends | B5 (*in no gap*) | 33 |
| **5** | **V-sync stimulus** — period-quantised flicker, jitter audit, FFT self-check, collision detection | Gap 5 + A3 | 37 |
| **6** | **TRCA / FBCSP / Riemannian** + leakage-safe grouped CV | Gap 2 (models) | 38 |
| **7** | **`bci_sdk`** — `BCISession`, `BrainButton.on_brain_select`, `Scanner`, typed errors, packaging, README | Gap 3 + B10 | 63 |
| **+** | **Dataset loaders + model persistence** (added during this review — see Part 2) | — | 34 |

**282 assertions, all passing, all headless.**

### Eight bugs caught by building it

1. **10 Hz collided with occipital alpha** → the 73.1% figure and its 10 Hz bias (Step 2)
2. **`BCIEngine` skipped the bandpass** on the live path, unlike the validated notebook (Step 2)
3. **64-channel indices on 8-channel hardware** → `IndexError` on first real data (Step 2)
4. **Blinks deleted by the SSVEP highpass** → 0/32 detected; artifacts need their own band (Step 3)
5. **Phase discontinuity between stream chunks** → every window read as `NO_ACTION` (Step 4)
6. **20 Hz silently rendered as 15 Hz** on a 60 Hz monitor — would have mislabelled every
   `target_1` calibration trial (Step 5)
7. **Overlapping windows inflated accuracy by +11%** under a naive split (Step 6)
8. **`Command.confidence` always 0.000** — the engine never exposed it (Step 7)

Items 4 and 6 would have silently corrupted the calibration data you are about to record.

---

## Part 2 — Review: is this a usable library for other devs?

You asked for three things. Here is the honest assessment.

### ✅ 1. Brain-controlled button selection — **yes, ready**

```python
button = bci_sdk.BrainButton("YES", rect=(50, 50, 200, 80))

@button.on_brain_select
def chosen(btn):
    print("selected", btn.label)
```

Two paradigms (`ButtonGroup` direct-gaze, `Scanner` two-command), a dwell bar so users can
cancel, blink rejection, and an idle state so nothing fires while reading. The acceptance
test is enforced in CI: `example/sdk_demo.py` is a complete app that imports **only**
`bci_sdk`, verified by a test that greps for internal imports.

### ⚠️ 2. Trained model from public + your own data — **was broken, now fixed**

When I reviewed the uploaded files against your goal, this did **not** work:

- **No dataset loaders.** Nothing could read Tsinghua `.mat` or PhysioNet `.edf`.
- **No model persistence.** `TRCAClassifier` had no `save()`/`load()` — a model trained in
  Colab could never reach your laptop. That silently blocks the entire workflow.
- **No way to use a model at runtime.** `BCISession` had no `model=` parameter.

Added and tested (34 assertions):

```python
# ML/datasets.py — one shape for every source: (trials, channels, samples)
Xp, yp, subs, _ = bci_sdk.load_benchmark_dir("benchmark_downloads", freq_map={35: 0})
Xp = bci_sdk.pick_channels(Xp, [0,1,24,28,42,46,60,62])   # 64ch -> your 8
Xo, yo, meta = bci_sdk.load_calibration_npz("dataset/calib_S01_*.npz")
X, y, src = bci_sdk.combine((Xp, yp), (Xo, yo))           # public + your own

model.save("ssvep_trca_8ch.joblib")                        # train in Colab
bci_sdk.BCISession(source="hardware", model="ssvep_trca_8ch.joblib")   # use locally
```

Guards included: refuses to save an unfitted model, rejects foreign model files, and raises
if a model's training frequencies don't match the session's stimulus.

**One honest caveat.** The Tsinghua Benchmark's 40 targets span **8.0–15.8 Hz**. Your
stimulus is 15/20 Hz, so **20 Hz does not exist in the public data** — you can pre-train
the 15 Hz class only. Public data cannot substitute for your own recording here.

### ✅ 3. Per-user calibration — **yes, ready**

```bash
python EEG/calibration_record.py --subject S01 --backend brainflow --serial-port COM4
python ML/calibration.py --recording "dataset/calib_S01_*.npz" --sweep
```
```python
bci_sdk.BCISession(source="hardware", profile="dataset/profile_calib_S01_*.json")
```

The protocol deliberately records more idle than active time, because the hard problem is
knowing when the user *isn't* trying. Output includes false-positives/minute, precision and
ITR — the metrics that predict whether a GUI is usable.

### The recommendation you may not expect

**Don't train a model yet.** Step 6 measured, on your current montage:

| | accuracy |
|---|---|
| FBCCA (training-free) | **100%** |
| TRCA (trained) | 57.6% |

TRCA is a *spatial* filter and `cyton8_ssvep` gives it only O1 and O2 — two free
parameters. Verified not to be a bug: TRCA scores 90% resubstitution, and beats FBCCA once
given 4+ occipital channels.

So the fastest path to a working speller is **Level 2** (calibration profile + FBCCA) — no
Drive, no Colab, no training. Train only after adding **Oz, PO3, PO4**, which is also the
cheapest genuine accuracy upgrade available to you.

---

## Part 3 — What is still not proven

| Item | Status |
|---|---|
| `BrainFlowBackend` | Written against the documented API, **never run against a physical Cyton**. Expect to adjust `board_id` and channel ordering. |
| All performance numbers | From **synthetic or replayed** data unless stated otherwise. |
| V-sync timing | Measured on the dummy SDL driver, which does not vsync. Re-check `measured fps ≈ nominal`, `dropped 0%` on your display. |
| Packaging | `pyproject.toml` is valid and installs editable; **no wheel built or installed into a clean venv**. |
| MI pipelines | Ranked on synthetic data. Confirm against PhysioNet in Colab. |
| `freq_map` for the Benchmark | **Must** be verified against `Freq_Phase.mat`. The original notebook assumed index 10 = 10 Hz without checking. |

### Dependency note
Your `requirements_loose.txt` uses **pygame-ce**; my `pyproject.toml` had specified
**pygame**. These are competing forks and installing both shadows one another. I aligned
everything to **pygame-ce** and re-ran the full suite — 282/282 still pass.

---

## Part 4 — Recommended next actions

1. **Record calibration data on your own head** (`--backend brainflow`). Everything else is
   waiting on this.
2. **Verify the flicker on your real monitor** — run `--flicker` and check the timing audit.
3. **Add Oz/PO3/PO4 electrodes** — the highest-value hardware change.
4. **Then, optionally, train** using `docs/GOOGLE_DRIVE_SETUP.md`.
5. **Report FP/min, precision and ITR**, not bare accuracy.

## Documents
- `docs/DEVELOPER_GUIDE.md` — how to learn the codebase (~1 hour read)
- `docs/GOOGLE_DRIVE_SETUP.md` — what to upload for training, and what not to
- `docs/steps/` — one design doc per step
- `docs/accuracy_strategy.md` — why accuracy is the wrong metric
- `docs/instructions_verification.md` — audit of the original handoff
