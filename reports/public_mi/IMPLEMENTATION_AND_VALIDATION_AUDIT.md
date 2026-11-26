# Motor-imagery implementation and validation audit

## Bottom line

The implementation contains **CSP/FBCSP** and **Riemannian covariance/tangent-space pipelines**. It uses **z-scoring selectively**, rather than applying one global raw-EEG z-score: Riemannian pipelines standardize tangent-space features from training-fold statistics; optional deep models standardize each EEG channel from training-set statistics; and trial-quality code uses robust median/MAD z-scores only to reject outliers. The selected production FBCSP-LDA hand model has no `StandardScaler`. There is no distinct **Euclidean Alignment (EA)** implementation. A **report-only Riemannian Alignment (RA)** option whitens each subject with that subject's unlabeled geometric-mean covariance, but it is transductive for held-out subjects and is not used by the production public models. The code does not cap or clip EEG amplitudes; suspect signals are rejected. The deployed-named “Riemannian classifiers” are more precisely tangent-space feature pipelines followed by an ordinary linear SVM or logistic regression, not direct manifold MDM classifiers.

The public models are correctly marked `runtime_ready=false`. Their offline, cued balanced accuracies do **not** validate spontaneous/asynchronous intention detection.

## 1. Technique and code audit

| Technique | Finding | Where and what it does |
|---|---|---|
| **Z-score / standardization** | **Implemented selectively** | `riemann_svm` and `riemann_lr` use sklearn `StandardScaler` after tangent mapping, so each of the 36 features is transformed using training-fold mean and standard deviation. Optional deep MI models compute one mean and standard deviation per channel over training trials and time, then apply the same stored statistics at inference. `ML/quality.py` also uses a robust median/MAD z-score of log band power for outlier rejection only. Raw EEG in the classical covariance path is channel-demeaned but not divided by its standard deviation, and the selected production `fbcsp_lda` hand pipeline has no scaler. |
| **CSP** | **Implemented** | `ML/mi_advanced.py`: binary CSP estimates class-average regularized covariances, solves a generalized eigenproblem, retains filters from both eigenvalue extremes, and outputs normalized log-variance features. |
| **FBCSP** | **Implemented and selected for hand** | `FilterBankCSP`/`OneVsRestFilterBankCSP` apply CSP in multiple mu/beta sub-bands. `ML/train_public_mi.py` uses five bands from 8–30 Hz, feature selection, and shrinkage LDA. The selected production hand model is `fbcsp_lda`. |
| **Euclidean Alignment (EA)** | **Not found as a distinct method** | There is no implementation of the usual subject-wise arithmetic-reference transform, e.g. `X' = R_arithmetic^(-1/2) X`, under an EA path. Feature `StandardScaler` is not EA. |
| **Riemannian Alignment (RA)** | **Present only in report experiments** | `ML/public_mi_experiments.py::riemannian_align_by_group` computes regularized trial covariances, their subject-specific geometric mean, and applies `R_subject^(-1/2)` to channel-demeaned trials. It uses no labels, but the complete unlabeled epoch batch of each held-out subject is used, making reported held-out use transductive. Production public training does not apply it. |
| **Whitening** | **Yes, in two covariance-geometric senses** | (1) RA applies subject-reference congruence whitening to trials. (2) The tangent mapper forms `M^(-1/2) C M^(-1/2)` around the training Fréchet mean. Neither is raw-amplitude clipping or ordinary per-trial z-scoring. |
| **Amplitude clipping/capping** | **No EEG clipping/capping** | `EEG/signal_preflight.py` and `ML/quality.py` flag/reject non-finite, flat, railed, amplitude-outlier, or jump-outlier data. The `np.clip` in `_trace_svg` only bounds display coordinates. Other `np.clip` calls control synthetic-signal parameters, not recorded EEG. |
| **Riemannian classifiers** | **Implemented, with a naming qualification** | `riemann_svm` and `riemann_lr` compute SPD covariance and tangent-space features, standardize those features, then fit a linear SVM or logistic regression. The classifier itself is Euclidean; the representation is Riemannian. No direct minimum-distance-to-Riemannian-mean classifier is used. |

### Important implementation qualifications

- `TangentSpaceMapper` is inside an sklearn pipeline, so its Fréchet mean is fitted on each training fold rather than on test labels/data.
- The report-only RA is different: it is computed before evaluation splitting and uses each held-out subject's full unlabeled batch to estimate that subject's reference.
- Standardized public archives now default to a broad 1–40 Hz import passband. Classical FBCSP and tangent pipelines still apply their own 8–30 Hz feature filters inside training folds; archive preservation and model feature selection are separate decisions.
- Offline filtering uses `sosfiltfilt`, which is zero-phase and non-causal. A live implementation must replace it with stateful causal filtering and revalidate performance and latency.
- The production hand model excludes `idle` and `both_hands`; it solves only left-hand versus right-hand imagery. It therefore cannot decide whether an intention is present.

## 2. Exact eight-channel SPD/tangent-space derivation

For the Riemannian route, one trial uses the ordered eight-channel montage

`[F3, C3, Cz, P3, Pz, F4, C4, P4]`

and can be written as

`X ∈ R^(8×T)`.

In the public archives, a typical three-second trial at 250 Hz has `T = 750`, but the derivation is valid for any `T`.

### Step 1 — band-pass and channel centering

The code applies a fourth-order Butterworth 8–30 Hz band-pass with `sosfiltfilt`, then removes each channel's temporal mean:

`Xc = X − mean_time(X)`.

### Step 2 — sample covariance

`C0 = Xc Xc^T / (T−1)`,

so `C0 ∈ R^(8×8)` and is symmetric positive semidefinite.

### Step 3 — trace normalization and shrinkage

`Cbar = C0 / tr(C0)`

and, with `reg = 0.05`,

`C = 0.95 Cbar + 0.05 I8 / 8`.

The identity term makes the result strictly positive definite in normal finite data, so each trial is a point on the SPD(8) manifold. Trace normalization removes global covariance scale; it does not clip the underlying voltage samples.

### Step 4 — training-fold Fréchet mean

For training covariances `{Ci}`, the code iteratively estimates the affine-invariant geometric/Fréchet mean

`M = arg min_P Σ_i d_R(P, Ci)^2`.

Only the training fold reaches this `fit` step.

### Step 5 — map each covariance around the mean

For each trial,

`A = M^(-1/2) C M^(-1/2)`

and

`S = log(A)`.

The eigendecompositions used for the inverse square root and logarithm floor eigenvalues at `1e−12` for numerical stability.

### Step 6 — symmetric vectorization

An 8×8 symmetric matrix has

`8(8+1)/2 = 36`

unique upper-triangular entries. The code outputs all 36, multiplying off-diagonal values by `sqrt(2)` and leaving diagonal values unchanged. This weighting preserves the symmetric matrix's Frobenius inner product after vectorization.

Thus the path is

`8×T EEG → 8×8 SPD covariance → 8×8 log-tangent matrix → 36 real features`.

A `StandardScaler`, fitted only on training features, then feeds either a linear SVM or logistic regression. This 36-feature result applies to the single-band Riemannian path; FBCSP has a different feature construction and dimensionality.

## 3. Public-dataset comparison

**Metric labels:** SH-BA = subject-held-out balanced accuracy; DH-BA = leave-one-source-dataset-out balanced accuracy. Project `±` values are standard deviations across evaluation folds, not confidence intervals and not necessarily across all individual subjects. Published source values below are ordinary accuracy or Cohen's kappa under their own protocols, not project BA.

| Public source | Original/associated published result and protocol | This project's eight-channel result |
|---|---|---|
| **OpenNeuro ds008446** | The 2026 release contains 20 participants, five finger classes, 62 EEG channels, and 512 Hz data. The associated FeatureStability code specifies separate-condition five-class nested three-fold RBF-SVM analyses, but no authoritative final numeric accuracy is publicly exposed in the release/README. No number should be invented. [1](https://openneuro.org/datasets/ds008446) · [2](https://github.com/lauramdamm/FeatureStability) | Five-class finger model, 5 subjects/1,500 trials in this run: **20.467% ± 2.579% SH-BA**, Riemannian LR; nominal chance **20%**. This is approximately chance and is not a positive finger-decoding result. |
| **PhysioNet EEGMMIDB** | The BCI2000 paper cited with the dataset describes the acquisition platform, not an original EEGMMIDB classification benchmark. An associated Cho et al. reanalysis reported **60.42% ± 11.68 percentage points** over 109 subjects with within-subject CSP+FLDA. [5](https://moabb.neurotechx.com/docs/generated/moabb.datasets.PhysionetMI.html) · [3](https://academic.oup.com/gigascience/article/6/7/gix034/3796323) | Feet idle-vs-both-feet, 20 subjects/898 trials: **61.281% ± 3.959% SH-BA**, Riemannian SVM. In the cross-source hand model, the leave-PhysioNet-out fold scored **60.652% DH-BA** on 900 left/right trials. |
| **BNCI2014-001 / BCI Competition IV 2a** | The competition winner obtained **κ = 0.57** on unseen evaluation sessions: four classes, 9 subjects, 22 EEG channels, subject-specific training/evaluation, FBCSP with multiclass extensions. Kappa must not be relabelled as accuracy. [1](https://www.bbci.de/competition/iv/results/) · [1](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2012.00039/full) | The project changes the task to binary left-vs-right and holds out the entire dataset while training on the other sources: **67.014% DH-BA** on 1,440 trials. BNCI feet was excluded because it lacks a harmonized idle class; retaining it would make source identity a shortcut. |
| **Ofner2017** | With 61 EEG channels and low-frequency time-domain/MRCP processing, the original within-subject study reported **27%** for six imagined movements and **73%** for imagined movement-versus-rest. [1](https://pubmed.ncbi.nlm.nih.gov/28797109/) | Seven classes including idle, 5 subjects/2,100 trials: **18.667% ± 1.092% SH-BA**, FBCSP-LDA; chance **14.286%**. The historical audited run used 8–30 Hz archives. New imports default to 1–40 Hz, which is broader but still omits the paper's 0.3–1 Hz MRCP content. Neither is a like-for-like reproduction of the 0.3–3 Hz analysis. |
| **Cho2017** | Binary left/right hand, within-subject CSP+FLDA with repeated random 70/30 subset splits: **67.46% ± 13.17 percentage points** over 50 usable subjects and 64 EEG channels. [2](https://pmc.ncbi.nlm.nih.gov/articles/PMC5493744/) | Leave-Cho-out hand fold, trained on the other three sources: **65.400% DH-BA** on 1,000 trials. This tests source transfer rather than subject-specific within-session fitting. |
| **Lee2019 / OpenBMI** | Subject-specific offline-training/online-test and session analyses with CSP-family methods reported MI mean **71.1%** (SD about 15 percentage points); the paper's table reports session means around **70.1%** and **72.2%**. It used 62 EEG channels and 54 subjects. [6](https://pmc.ncbi.nlm.nih.gov/articles/PMC6501944/) | Leave-OpenBMI-out hand fold, trained on the other three sources: **67.200% DH-BA** on 1,000 trials. Again, this is dataset transfer with eight channels, not subject-specific offline-to-online performance. |

The selected cross-source hand model overall is **66.213% ± 5.846% SH-BA** and **65.066% ± 2.643% DH-BA** over 35 subjects and 4,340 binary trials. It is FBCSP-LDA. These are the project's most defensible public-MI numbers, but they should not be presented as exceeding or reproducing the original studies because the channels, task definitions, participant subsets, frequency content, metrics, and validation splits differ.

## 4. Citation correction

The alleged “2015 IEEE TNSRE Riemannian EEG” citation does not resolve to one matching publication.

- The closest TNSRE review is **Yger, Bérar, and Lotte, “Riemannian Approaches in Brain-Computer Interfaces: A Review,” IEEE TNSRE 25(10), 1753–1762**. Its DOI is `10.1109/TNSRE.2016.2627016`; it was available online in 2016 and appeared in the 2017 issue—not 2015. [1](https://hal.science/hal-01394253)
- A foundational motor-imagery covariance-manifold paper is **Barachant et al., “Multiclass Brain–Computer Interface Classification by Riemannian Geometry,” IEEE Transactions on Biomedical Engineering 59(4), 920–928 (2012)**, DOI `10.1109/TBME.2011.2172210`. It is TBME, not TNSRE. [2](https://hal.science/hal-00681328)

Use the former for a review citation and the latter for the core covariance/tangent-space approach; do not cite a nonexistent 2015 TNSRE item.

## 5. Does the current model support spontaneous intention detection?

**No.** The current results validate fixed, cue-locked epoch classification only. They do not establish a detector that can decide, continuously and without a cue, whether the user intends to move.

The present gap is structural:

1. The hand and finger models do not contain a genuine no-intent class.
2. Cued idle epochs are not representative of long natural non-control periods.
3. Abrupt cues can introduce visual/anticipatory structure absent during self-paced operation.
4. `sosfiltfilt` uses future samples and is non-causal.
5. Report-only RA uses the complete unlabeled held-out-subject batch and is not a causal online adaptation rule.
6. Balanced accuracy on pre-cut trials does not measure false commands accumulated over minutes or hours.
7. The metadata correctly records `runtime_ready=false` for all public models.

Published asynchronous work illustrates why trial accuracy is insufficient. Ofner et al. reported TPRs of **26.6% and 36.9%** with **3.2 and 3.6 false positives/min**, while class accuracy among true detections was **66.0% and 70.8%**. [2](https://pmc.ncbi.nlm.nih.gov/articles/PMC6509331/) A 2025 self-paced study obtained **31.8% TPR** at a calibrated maximum of **1 FP/min**, explicitly evaluating event-by-event and using a refractory period. [4](https://pmc.ncbi.nlm.nih.gov/articles/PMC11850329/)

## 6. Required validation and runtime logic

### Data and validation

- Record self-paced/asynchronous sessions with long, realistic non-intent periods—not only instructed rest.
- Include confounds: blinks, jaw/face activity, speaking, posture changes, cable/electrode disturbances, ordinary arm movements, attention changes, and environmental variation.
- Obtain independent movement-onset ground truth with EMG, IMU, motion capture, or a switch; define the acceptable detection window before analysis.
- Separate calibration, threshold tuning, and final testing by session/day. Include cap replacement and, ideally, environment changes.
- Personalize the model and thresholds, then test on untouched continuous recordings. Report per-subject results and uncertainty rather than only a pooled mean.
- If alignment/adaptation is retained, estimate it from calibration data or past streaming data only. Never use future samples or the complete test batch.

### Causal runtime state machine

A safe minimum design is:

1. Stateful causal filtering and resampling.
2. Sliding windows, e.g. every 100–250 ms, with a declared history length.
3. Signal-quality/artifact and out-of-distribution gate; reject rather than force a command.
4. First-stage `intent` versus `no-intent` probability.
5. A calibrated threshold plus persistence/dwell rule, such as N of the last K windows, with hysteresis.
6. Only after an intent trigger, a second-stage movement/command classifier.
7. Confidence/abstention logic for the command class.
8. Refractory/cooldown period, duplicate-event suppression, and a neutral fail-safe state.
9. Causal adaptation with bounded updates and rollback; log every window, rejection, trigger, and emitted command.

### Metrics that must be reported

- Event-level true-positive rate/sensitivity and miss rate.
- False activations per minute and per hour during genuine no-intent time.
- Detection latency distribution relative to ground-truth onset.
- Precision/positive predictive value and event-level F1.
- The full TPR-versus-FP/min operating curve across thresholds, not one cherry-picked threshold.
- Conditional movement-class confusion matrix/BA **after** a true intent detection, kept separate from detector performance.
- Abstention/rejection rate, artifact-trigger rate, uptime, and probability calibration such as Brier score or ECE.
- Closed-loop task success, command rate, recovery from false commands, and safety-related outcomes.

Until those tests pass with a predeclared acceptable operating point, the public models should remain benchmark/pretraining artifacts rather than spontaneous-control models.
