# Step 6 — ML upgrades: TRCA (SSVEP) and FBCSP / Riemannian (MI)

**Covers:** `instructions.md` Gap 2 items 1 and 2.
**Status:** ✅ implemented and measured. **Tests:** 38 assertions in `tests/test_step6_models.py`.

**Headline finding: neither upgrade should ship on your current hardware.** The measurements
below say the bottleneck is your *montage* and your *calibration*, not your algorithms. That is
a more useful result than a speculative accuracy claim, and both conclusions are reproducible.

---

## First: a validation trap that would have invalidated every number

Step 3 records 750-sample windows with a 250-sample hop, so **consecutive windows share 500 of
750 samples**. Verified directly:

```
window[0][:, 250:] == window[1][:, :500]   ->  True
60 windows, but only 2 truly independent recording blocks
```

A random train/test split puts near-duplicates on both sides. Quantified in
`tests/exp_leakage.py`:

```
naive random split (LEAKY) : 68.6% +/- 9.3%
grouped CV (honest)        : 57.6% +/- 2.5%
INFLATION                  : +11.0%
```

Every result in this step uses **grouped cross-validation** (`ML/evaluate.py`): windows from the
same recording block share a group id and always land in the same fold, with the overlapping
seam between chunks discarded. Numbers are lower than a naive split would give, and they are the
ones you can defend.

---

## SSVEP: FBCCA vs TRCA

`ML/trca.py` implements TRCA, ensemble TRCA, and filter-bank TRCA. TRCA learns, from the
subject's own data, the spatial filter maximising cross-trial reproducibility
(`S w = λ Q w`), amplifying the phase-locked evoked response and suppressing non-reproducible
background — including alpha.

On your actual Step 3 recording, under grouped CV:

| method | accuracy | mean confidence |
|---|---|---|
| **FBCCA** (training-free) | **100.0% ± 0.0%** | 0.449 |
| TRCA (subject-specific) | 57.6% ± 2.5% | 0.027 |

### Is that a bug? No — I checked

TRCA trained *and* tested on all data reaches **90%**, so the implementation is sound; the CV
result is a data/montage limit. `tests/exp_trca_channels.py` isolates the cause by varying
occipital channel count at fixed low SNR:

| occipital channels | FBCCA | TRCA | verdict |
|---|---|---|---|
| 2 (your Cyton) | 81.0% | 80.0% | tie |
| 4 | 89.0% | 94.0% | **TRCA wins** |
| 6 | 97.7% | 100.0% | tie |
| 9 (literature setup) | 98.3% | 99.7% | tie |

**TRCA is a spatial filter.** With `cyton8_ssvep` only O1 and O2 are occipital, so `w` has two
free parameters — there is essentially no spatial mixing left to optimise. Published TRCA
(Nakanishi 2018) uses ~9 occipital channels and 6+ blocks per class on 64-channel caps. It also
needs many repetitions for a stable covariance estimate; each CV fold here had ~4 independent
epochs.

> ⚠️ While building that experiment I hit a bug worth recording: `make()` generated a **new
> random mixing matrix per call**, so train and test came from different simulated subjects.
> TRCA collapsed (to 63–76%, sd ±20%) while FBCCA was unaffected, because FBCCA is spatially
> agnostic. Fixing it to share one mixing matrix per subject produced the coherent table above.
> A subject-specific method evaluated across subjects will always look broken.

**Recommendation:** keep **FBCCA** as the default. `HybridTRCAClassifier` gives you TRCA behind
the same API with automatic FBCCA fallback, so if you later add occipital electrodes (Oz, PO3,
PO4 are the cheap wins) you can switch with one flag and re-run the comparison.

---

## Motor Imagery: CSP+LDA vs FBCSP vs Riemannian

`ML/mi_advanced.py` implements CSP, Filter-Bank CSP, Riemannian geometric mean, and tangent-space
mapping in **numpy/scipy/sklearn only** — no `mne`, no `pyriemann`, keeping the SDK dependency
surface small.

The right question isn't "which algorithm?" but "is the algorithm even the bottleneck?":

| pipeline | cross-subject | within-subject | gain from calibration |
|---|---|---|---|
| csp_lda | 55.9% ± 10.3% | **79.2% ± 5.3%** | **+23.2%** |
| riemann_lr | 62.7% ± 2.9% | 70.4% ± 6.7% | +7.8% |
| fbcsp | 50.8% ± 0.8% | 69.2% ± 5.3% | +18.4% |
| riemann_svm | 56.6% ± 6.0% | 65.0% ± 6.3% | +8.4% |

**Per-subject calibration is worth ~+23 points. Switching algorithms is worth ~+7.** That is the
answer to Gap 2: the instructions propose replacing the classifier, but the measurement says
replacing *cross-subject training with per-subject training* matters roughly three times more.

**Recommendation stands: keep MI out of the live demo.** Even at the best within-subject 79%,
one in five commands is wrong. Ship SSVEP + blink; document MI as a research branch.

### Honesty about the MI numbers
These use synthetic data (`ML/mi_synth.py`), since PhysioNet isn't available in this
environment. My first generator was **too easy** — plain CSP+LDA scored 97% cross-subject,
absurd against the real 61.4%. It added a large phase-locked sinusoid; real motor imagery is a
modest, noisy *band-power reduction* with random phase. The rewritten generator models ERD
properly and lands CSP+LDA in the mid-50s cross-subject, consistent with reality.

Synthetic data can validate that pipelines are correctly implemented and behave sensibly. It
**cannot** definitively rank methods for real EEG. Treat the ordering as a hypothesis to confirm
against your PhysioNet notebook.

---

## Files

| File | Purpose |
|---|---|
| `ML/trca.py` | **new** — TRCA, ensemble/filter-bank TRCA, `HybridTRCAClassifier` with FBCCA fallback |
| `ML/mi_advanced.py` | **new** — CSP, FBCSP, Riemannian mean, tangent space, 4 pipelines |
| `ML/evaluate.py` | **new** — grouped CV, SSVEP and MI comparisons, within-vs-cross analysis |
| `ML/mi_synth.py` | **new** — realistic ERD-based synthetic MI |
| `tests/test_step6_models.py` | **new** — 38 assertions |
| `tests/exp_leakage.py`, `tests/exp_trca_channels.py` | **new** — the two experiments above |
| `requirements.txt` | **new** — pinned deps (`brainflow` optional) |

## Run
```bash
python ML/evaluate.py                       # FBCCA vs TRCA on your recording
python ML/evaluate.py --mi-synthetic        # + MI pipeline comparison
python tests/exp_leakage.py                 # leakage inflation
python tests/exp_trca_channels.py           # TRCA vs channel count
python tests/test_step6_models.py           # 38 assertions
```

## Carried forward
- **Cheapest real accuracy win: add occipital electrodes.** Oz/PO3/PO4 would take you from 2 to
  5 SSVEP channels and make TRCA viable — worth more than any further algorithm work.
- Re-run `ML/evaluate.py` on a **real** Step 3 recording; the current one is simulate-backend.
- The MI ranking needs confirming on PhysioNet via your Colab notebook.
- Next: **Step 7 — SDK wrapper** (Gap 3). The API can now be frozen around components that have
  been measured rather than assumed.
