# Testing Guide

How to tell whether this system works — and, when it doesn't, which part is
broken.

```bash
python tools/layer_tests.py --layer 1 --user anson    # hardware
python tools/layer_tests.py --layer 2 --user anson    # signal
python tools/layer_tests.py --layer 3 --user anson    # model
python tools/layer_tests.py --layer 4                 # online protocol
python tools/layer_tests.py --layer 5                 # usability protocol
python tools/layer_tests.py --all --user anson
```

---

## Why layered testing

A BCI can fail in five independent places. **All five look identical from the
outside: you try, and nothing happens.**

| Layer | Fails when | Looks like |
|---|---|---|
| 1 Hardware | electrode not contacting scalp | nothing happens |
| 2 Signal | no measurable ERD in this person | nothing happens |
| 3 Model | classifier can't learn it | nothing happens |
| 4 Interface | works offline, not in real time | nothing happens |
| 5 Product | works but too slow or tiring | user gives up |

A single accuracy number cannot distinguish these. Worse, **testing a high layer
on data broken at a low layer produces confident nonsense** — you can spend a
week tuning a classifier that was fed noise from a disconnected electrode.

**Always run in order. Fix the lowest failure first.**

---

# Layer 1 · Hardware and signal quality

**Question:** is the rig recording brain activity at all?

```bash
python tools/layer_tests.py --layer 1 --user anson
```

### 1.1 Per-channel amplitude

| Reading | Meaning | Action |
|---|---|---|
| 5–50 µV | normal EEG | ✅ |
| < 1 µV | channel dead / not connected | check the pin and lead |
| 50–100 µV | noisy | re-seat, check hair under the electrode |
| > 200 µV | saturated | bad contact, movement, or floating reference |

### 1.2 Flat-line detection
Counts unique values per channel. A stuck ADC or unplugged lead produces a near
constant value that *looks* plausible on a plot but carries no information.

### 1.3 Mains interference
Ratio of 48–52 Hz power (50 Hz in Hong Kong) to 8–30 Hz power. Above 2× means
mains dominates the band we classify on.
**Fixes:** move away from chargers and power strips, check the ground/bias
electrode, unplug the laptop charger.

### 1.4 Electrode bridging
Pairwise correlation across channels. *r* > 0.98 means two electrodes are
electrically joined — usually gel smeared between adjacent sites. Bridged
channels destroy every spatial filter, because CSP works by weighting channels
*differently*.

### 1.5 Alpha blocking — the definitive test ⭐

Record 15 s eyes open, then 15 s eyes closed, labelled `eyes_open` /
`eyes_closed`. Occipital 8–13 Hz power should rise **≥ 1.5×** with eyes closed.

**Why this test matters more than any other in this document:** alpha blocking is
large, universal, and needs no training. If you cannot see it, the rig is not
recording usable EEG and every later number is noise. It takes two minutes and
saves entire wasted sessions.

If it fails: electrodes not posterior enough, insufficient scalp contact through
hair, or a reference problem.

---

# Layer 2 · Signal separability

**Question:** does *this person's* imagined movement produce a measurable
difference — **before any classifier is involved**?

```bash
python tools/layer_tests.py --layer 2 --user anson
```

This layer exists because a classifier cannot recover information that was never
recorded. Testing it separately prevents blaming the model for missing signal.

### 2.1 Class balance
≥ 8 trials per class, ratio > 0.8. Imbalance biases accuracy and every derived
metric.

### 2.2 Band power per channel
Log-variance in 8–30 Hz per channel per class, with Cohen's *d* for each.
Shows *which electrodes carry information* — if only frontal channels separate,
you are probably classifying eye movement, not motor imagery.

### 2.3 Lateralisation direction ⭐

ERD is **contralateral**: imagining the *left* hand attenuates the *right* motor
cortex (C4), and vice versa.

The test verifies both directions. **A reversed result is diagnostic, not noise**
— it usually means C3 and C4 leads are swapped, or the cue labels are inverted in
the recorder. Without this check, swapped electrodes look like a failed
experiment.

### 2.4 Separability — the go/no-go gate ⭐

Reports best single-channel Cohen's *d* and multivariate Mahalanobis distance.

| Cohen's *d* | Interpretation | Action |
|---|---|---|
| **> 1.5** | strongly separable | proceed to Layer 3 |
| **0.8 – 1.5** | marginal | more trials; check electrode placement |
| **< 0.8** | not separable | **stop.** Fix Layer 1, or reconsider the paradigm |

Measured *before* training deliberately. If *d* < 0.8, no classifier will rescue
it and time spent on Layer 3 is wasted.

### 2.5 Statistical reality
Two-sample *t*-test on the best channel. *p* < 0.05 required — a visible
difference in a small sample is often chance.

### 2.6 Idle separability
Compares resting windows against imagery. Needed so the system can output
`NO_ACTION`. If rest and intent overlap, no confidence threshold can separate
them and the interface will fire constantly (see Layer 4.2).

---

# Layer 3 · Offline model accuracy

**Question:** can a classifier learn it, and is the score real?

```bash
python tools/layer_tests.py --layer 3 --user anson --perm 200
```

### 3.1 Enough independent blocks ⭐

**Needs ≥ 3 separate recording blocks per class.**

The protocol records each class as one continuous block, so a single session
gives only 1–2 blocks per class. Cross-validation then trains on one block and
tests on another, producing meaningless extremes.

> Measured on a real 2-session recording: `fbcsp` scored **100.0%** and
> `riemann_svm` **26.6%** (below chance). Both were artefacts of having 4 blocks
> — not real performance.

**Record several shorter sessions rather than one long one.**

### 3.2 Grouped cross-validation ⭐

Windows are 750 samples with a 250-sample hop, so consecutive windows share
**66.7%** of their data:

```python
window[0][:, 250:] == window[1][:, :500]   # True
```

A random split places near-duplicates in both train and test.

> **Measured inflation from a random split: +11.0 percentage points.**

All results use grouped *k*-fold with blocks confined to single folds.

### 3.3 Binomial test against chance
Is the accuracy distinguishable from a coin flip given the sample size?
62% on 40 trials is not.

### 3.4 Permutation test ⭐
Shuffles labels *M* times and **rebuilds the entire pipeline** each time,
producing an empirical null distribution.

$$p = \frac{1 + |\{m : a(\pi_m) \ge a_{\text{obs}}\}|}{M + 1}$$

This is the strictest available test — it accounts for optimism introduced by
feature selection and hyperparameter choices, which a binomial test misses.

### 3.5 Bootstrap confidence interval
Report intervals, never bare point estimates:

```
54.7% (95% CI 50.0% – 59.4%)
```

If the lower bound touches 50%, the result is not solid regardless of the mean.

### 3.6 Learning curve
Accuracy vs. amount of data. Still rising → record more. Flat → at the ceiling
for this montage.

### 3.7 Cross-session stability
Train on day 1, test on day 2. Within ~10 points means electrode placement is
repeatable. A large drop means recalibration is required every session — a
usability finding, not a bug.

### Worked example — the test catching a false positive

```
best pipeline csp_lda at 54.7%
[FAIL] binomial test    p = 0.3179
[FAIL] permutation test p = 0.1765
[WARN] 54.7% (95% CI 50.0% – 59.4%)
```

54.7% looks like a result. It isn't. **Without §3.3–3.5 this would have been
reported as a finding.**

---

# Layer 4 · Online performance

**Question:** does it work in real time, with a human in the loop?

Offline accuracy uses clean labelled trials. Online adds feedback, fatigue, and
the need to *withhold* commands.

### 4.1 Cued online accuracy
```bash
python EEG/data_record.py --file realistic_ui --trials 20 --cue 6
```
Proportion of trials where the cued button was selected first. **Target ≥ 70%.**
The generated `cue_log_*.csv` records intended vs actual for each trial.

### 4.2 False activations per minute of rest ⭐ — the headline metric

**Protocol:** wear the headset and do nothing for 5 minutes. Rest, read, look
around, blink normally.
**Target: < 0.5 commands/min.**

> Why this outranks accuracy: a classifier using unconstrained `argmax` measured
> **100% accuracy** on attentive trials while emitting a command on **141 of 141**
> idle windows. Accuracy alone would have called it a success. It was unusable.

### 4.3 Time to selection
Median and 90th percentile, including retries. A technically accurate system
taking 90 s per press is not usable.

### 4.4 Information transfer rate
Wolpaw ITR (bits/min) combines accuracy and speed — the standard BCI metric, and
the one that exposes a system trading one for the other with no net gain.

### 4.5 Artifact rejection
20 deliberate blinks, 10 jaw clenches, 10 head turns during rest.
**Target: ≥ 90% flagged, 0 producing a command.** Jaw clenches matter — EMG is
broadband and lands squarely in the 8–30 Hz band.

### 4.6 Error recovery
After a wrong selection, can the user correct within 2 attempts? A UI with no
path back is unusable regardless of accuracy.

### 4.7 Drift over a session
Repeat 4.1 at minutes 0, 10 and 20. **Target: < 15 point drop.** Catches drying
electrodes and mental fatigue — the usual reason a demo works once and fails on
the second run.

---

# Layer 5 · Usability and product

### 5.1 Task completion
Select 5 specific HUD actions in order. **Target ≥ 80% unaided.**

### 5.2 Subjective workload ⭐
NASA-TLX or a 1–5 fatigue scale after each block. **Target: mental demand ≤ 3/5
after 10 minutes.**

Motor imagery is genuinely tiring. A system that works for 2 minutes and
exhausts the user by 10 has failed, and **no objective metric captures this**.

### 5.3 Learning effect
Same task across 3 days. BCI control is a trainable skill; improvement is
expected and is itself a result.

### 5.4 Multi-participant validation ⭐
**3–5 participants, each with their own calibration.** Report per-user accuracy
and the spread.

An estimated **15–30% of people show little usable motor imagery** ("BCI
illiteracy"), so a small study may legitimately include a non-responder. Report
it — a single-subject success is not evidence of a general library.

### 5.5 Developer integration
Someone unfamiliar builds a brain-controlled button using only the public API and
README. Automated proxy: `example/sdk_demo.py` imports only `bci_sdk`, enforced
by a test that greps for internal imports.

### 5.6 Installation
```bash
python -m build && pip install dist/*.whl && python -c "import bci_sdk"
```
**Current status: FAILS.** The wheel omits `EEG/` and `pygame_lib/`. Known,
documented, ~1 hour to fix.

### 5.7 Safety and comfort
Session length limits, electrode pressure, skin condition after removal, and a
clearly communicated stop instruction.

---

# Automated regression tests

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

**~311 assertions. These verify that the code runs and its internal contracts
hold — nothing more.** They say nothing about whether the system works on a
brain. Both kinds of testing are necessary; conflating them would be misleading.

---

# Acceptance criteria

| # | Criterion | Threshold | Layer |
|---|---|---|---|
| 1 | Alpha blocking present | qualitative, all subjects | 1 |
| 2 | Class separability | Cohen's *d* > 0.8 | 2 |
| 3 | Offline accuracy | > 65%, *p* < 0.05 | 3 |
| 4 | Online cued accuracy | ≥ 70% | 4 |
| 5 | False activations at rest | < 0.5 / min | 4 |
| 6 | Cross-session stability | within 10 points | 3 |
| 7 | Multi-participant | ≥ 3 subjects meeting #3 | 5 |
| 8 | Library installs and runs | `pip install` → working demo | 5 |

---

# Interpreting results

### Compound reliability

Single-decision accuracy misleads. For *n* sequential decisions at per-step
accuracy *P*:

$$P_{\text{task}} = P^{\,n}$$

| Per-step | 4-step selection | Attempts needed |
|---|---|---|
| 61.4% | 14.2% | ~7 |
| 79.2% | 39.4% | ~2.5 |
| 90% | 65.6% | ~1.5 |

**61% per step sounds acceptable and is not.** This is why interface design
(fewer buttons, shorter scan paths) is evaluated alongside the classifier.

### Negative results

If separability fails across all participants **with Layer 1 passing**, the
defensible conclusion is that 8-channel dry-electrode motor imagery is unsuitable
for real-time discrete selection.

That is a legitimate, useful finding — provided hardware has been independently
verified. The layered structure exists precisely so that conclusion can be drawn
with confidence rather than guessed at.

---

# Test schedule

| Phase | Tests | Requires |
|---|---|---|
| Bench | regression suites | nothing |
| First hardware session | Layer 1, 2 | headset |
| Data collection | Layer 2, 3.1–3.5 | headset, 3+ sessions |
| Model selection | Layer 3.6–3.7 | accumulated recordings |
| Online evaluation | Layer 4 | headset + trained model |
| User study | Layer 5 | 3–5 participants |
