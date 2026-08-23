# How to actually increase accuracy — and are we heading toward a realistic GUI library?

**Short answer: partly. The direction is right, but the target metric is wrong, and one
hardware/design decision is costing you more accuracy than any classifier upgrade can win back.**

Two experiments in `tests/` back everything below, run against your real
`HybridSSVEPClassifier`.

---

## 1. Your biggest accuracy win is not an algorithm — it's the stimulus frequencies

`exp_frequency_choice.py`, 120 trials, realistic resting-alpha background:

| Condition | Targets | Accuracy | recall low | recall high |
|---|---|---|---|---|
| Moderate alpha | **10 / 12 Hz (current)** | 91.7 % | 100 % | 83.3 % |
| Moderate alpha | **15 / 20 Hz** | **100 %** | 100 % | 100 % |
| Strong alpha (drowsy) | **10 / 12 Hz (current)** | 80.8 % | 100 % | **61.7 %** |
| Strong alpha (drowsy) | **15 / 20 Hz** | **100 %** | 100 % | 100 % |

This reproduces your Colab bias exactly: the 10 Hz class gets 100 % recall while 12 Hz
collapses, and the classifier predicts class-0 69 % of the time on a balanced set — the same
signature as your 295-vs-125 split at 73.10 %.

**The cause is a design choice, not a bug.** 10 Hz sits in the middle of the occipital alpha
band (8–13 Hz), which is present *whenever the user relaxes their eyes*. You are asking the
classifier to distinguish an evoked 10 Hz response from a spontaneous 10 Hz rhythm at the same
electrodes. No amount of FBCCA/TRCA fixes that — you're fighting physiology.

**Action: move the SSVEP targets out of the alpha band.** 15 Hz and 20 Hz are both outside
8–13 Hz, and both are exact divisors of a 60 Hz monitor (every 4 frames / every 3 frames), so
they can be rendered jitter-free — which is also what Gap 5 needs. This is a one-line change to
`target_freqs` plus the flicker code, and it is worth more than the entire Gap 2 ML refactor.

*Caveat: simulated signals are cleaner than real ones, so 100 % will not hold on hardware. The
**direction and mechanism** are what matter — the alpha-band collision is real and well documented
in the SSVEP literature.*

---

## 2. "Accuracy" is the wrong metric for a GUI library — and this is the core of your question

You asked whether public controlled-experiment data transfers to a realistic GUI with
distractions. `exp_idle_and_metrics.py` answers it. Simulated 200-window session where the user
is only intentionally selecting 30 % of the time (realistic: they read, think, blink, look away):

**1) Forced argmax — what your code does today**
```
accuracy on attentive trials : 100.0%   <- the number you'd report
commands emitted while idle  : 141/141 = 100.0%
```
Perfect benchmark accuracy, **completely unusable GUI**. It types 141 garbage characters per 200
windows. This is the exact gap between a controlled experiment and your product: the benchmark
*never shows the classifier a trial where the answer is "nothing"*, so it cannot measure the only
failure mode that matters live.

**2) Confidence threshold + N consecutive agreeing windows**
```
threshold + 3 consecutive:  wrong commands 0,  false-idle 0/141 = 0.0%,  precision 100%
threshold + 2 consecutive:  wrong commands 0,  false-idle 4/141 = 2.8%,  precision 69.2%
```
The scores separate cleanly — attentive mean **0.430** vs idle mean **0.053**, ~4 sd apart. Idle
rejection is very achievable *if the classifier stops throwing the correlation scores away*.

> Caveat on "intentional windows ignored": that count is inflated by my simulation interleaving
> idle windows randomly between attentive ones, breaking up consecutive runs. On real hardware a
> user gazes continuously for 2–4 s, so runs form naturally. Treat the precision/false-positive
> columns as meaningful and the miss count as pessimistic.

**Metrics you should report instead of bare accuracy:**
- **False positives per minute of idle time** (the headline number for a GUI library)
- **Precision** of issued commands
- **ITR** (bits/min) — the standard BCI metric, captures the speed/accuracy trade-off
- Accuracy *conditional on a command being issued*

---

## 3. On public data vs. your controlled-experiment problem

You're right to be suspicious. Three distinct transfer gaps:

| Gap | Consequence | Fix |
|---|---|---|
| **Benchmark has no idle class** | Cannot learn or validate "do nothing" | Record your own idle/distraction data — this is the one thing public data can never give you |
| **64-ch lab cap → 8-ch Cyton** | `ssvep_channels=[54,55,60,61,62]` `IndexError`s on live data (blocker B1) | Map to O1/O2/Oz; re-validate the benchmark using *only* the 8 electrodes you actually own |
| **Cross-subject MI** | 61.4 % ≈ chance; CSP filters are subject-specific | Per-subject calibration, or drop MI from the critical path |

**Recommendation on MI: don't put it in the demo.** 61.4 % cross-subject means roughly 2 of every
5 commands are wrong. Ship SSVEP + blink as the reliable control, keep MI as a documented
research branch. A 2-command SSVEP speller that works beats a 3-command one that doesn't.

**Most valuable thing you can do:** record ~20 minutes on your own Cyton with a labelled protocol
that *includes* idle blocks — gaze at target, rest with eyes open, read text on screen, look away,
blink deliberately, converse. Public data validates your algorithm; only your own data validates
your product. This also unblocks TRCA (which needs subject-specific training data) and calibration.

---

## 4. Are we working toward a realistic distraction-tolerant GUI library?

**Directionally yes, structurally not yet.** Honest scorecard:

| Requirement for a realistic EEG-controlled GUI | Status |
|---|---|
| Multi-tile GUI with real application semantics | ✅ Step 1 speller |
| Text buffer / output decoupled from rendering | ✅ `TextBuffer` is pygame-free |
| Blink/artifact rejection exists | ✅ `artifact_detection` works (verified on injected 180 µV spike) |
| **Idle / NO_ACTION state** | ❌ **forced argmax — cannot say "nothing"** |
| **Classifier returns confidence** | ❌ `np.argmax()` discards the scores (blocker B4) |
| Debounce requiring consecutive agreement | ⚠️ exists but votes on *labels*, not confidence |
| Distraction handling (reading, looking away) | ❌ untested, no data |
| Jitter-free stimulus | ❌ Gap 5 |
| Live acquisition → engine bridge | ❌ blocker B5 |
| Metrics that reflect GUI usability | ❌ still reporting bare accuracy |

The single structural change that unlocks the whole "realistic GUI" goal is making the
classifiers return `(label, confidence)` instead of `argmax`. Everything in Gap 4 — idle state,
thresholding, meaningful debounce — is currently *unimplementable* without it. That's why I put
it as step 2 in the verification doc, ahead of the Gap 2 ML refactor.

Your `example/realistic_ui.py` ("a distraction of background scenary") shows you already have the
right instinct. The missing half is that visual distraction must be paired with a classifier
allowed to answer "no command".

---

## Revised priority order

1. **Change target frequencies to 15/20 Hz** — biggest accuracy win, ~1 line, also serves Gap 5.
2. **`(label, confidence)` return + `NO_ACTION`** — unblocks everything about distraction tolerance.
3. **Fix the 8-channel map + the `BCIEngine` filter bypass** — required before any live run.
4. **Record your own labelled data *including idle/distraction blocks*.**
5. Acquisition→engine bridge; then Gap 5 V-sync flicker.
6. Only then Gap 2's TRCA/FBCSP — with calibration data it can actually use.
7. Report FP/min + ITR + precision, not just accuracy.

Steps 1–3 are small, and I'd expect them to move live usability more than the entire Gap 2
refactor would.
