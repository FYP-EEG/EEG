# Public motor-imagery datasets and category-specialist models

Edited by Anson  
Date: 10/11/2026  
Purpose: document licensed public finger, hand, upper-limb and feet EEG support without confusing public benchmarks with personal Cyton validation.

## What this integration does

Public data is stored separately from personal calibration:

```text
dataset/public/raw/<dataset_id>/                 downloaded source files (do not commit)
dataset/public/processed/<dataset_id>/<category>/epochs.npz
models/public/<category>/category_model.joblib   benchmark/pretraining only
profiles/<user>/motor_intent/<task-set>/         exact personal game model
profiles/<user>/motor_intent/_categories/<category>/intent_model.joblib
```

The four model categories are:

- `finger`: individual fingers. Unsided public finger labels remain explicitly unsided.
- `hand`: gross left/right/both-hand imagery.
- `upper_limb`: arm, elbow, forearm and hand-open/hand-close movement classes.
- `feet`: left/right/both-feet imagery.

A live game resolves its model in this order:

1. exact personal task-set model;
2. compatible personal category specialist whose trained intents cover the game;
3. otherwise fail and request calibration.

A public category model is **not** served directly. It may initialize a compatible EEGConformer or TCFormer, after which a personal head/model must be fine-tuned and saved under the user's profile. This prevents public cross-subject accuracy from being represented as real-user Cyton accuracy.

## Registry

The machine-readable registry is `ML/public_mi.py::PUBLIC_MI_DATASETS`.

| ID | Category | True task labels | Access / license | Important restriction |
|---|---|---|---|---|
| `openneuro_ds008446` | finger | thumb, index, middle, ring, pinky, **side not published in task labels** | public, CC0 1.0 | Imported as `finger_*_unsided`; never duplicate into left and right classes. |
| `physionet_eegmmidb` | hand, feet | imagery runs 04/08/12: left/right fist; imagery runs 06/10/14: both fists/both feet | public, ODC-By 1.0 | T1/T2 depend on run. Execution runs are rejected. |
| `bnci2014_001` | hand, feet | left hand, right hand, both feet, tongue | public, CC-BY-ND 4.0 | Tongue is excluded. Do not redistribute derived archives under NoDerivatives terms. |
| `ofner2017` | upper_limb | right elbow flex/extend, right forearm supinate/pronate, right hand open/close, rest | public, CC-BY 4.0 | Import the motor-imagery session only; all movement classes are right-sided. |
| `cho2017` | hand | left/right kinesthetic hand finger-movement imagery | public, CC-BY 4.0 | This is a hand class, not an individual-finger class. |
| `lee2019_mi` | hand | left/right hand grasp imagery | public EEG, CC-BY 4.0 | Labelled training runs only; unlabelled test runs are excluded. GPL-3.0 applies to the OpenBMI software and is inherited by NEMAR derivative metadata. |
| `zenodo_18470278` | upper_limb | left arm, right arm, extra virtual arm | restricted, license unstated | Catalogued but automated import is disabled until access and reuse terms are verified. |

Sources and citations:

- OpenNeuro ds008446 v1.0.1: <https://openneuro.org/datasets/ds008446/versions/1.0.1>, DOI `10.18112/openneuro.ds008446.v1.0.1`.
- PhysioNet EEGMMIDB v1.0.0: <https://physionet.org/content/eegmmidb/1.0.0/>, DOI `10.13026/C28G6P`.
- BCI Competition IV 2a / BNCI2014-001: <https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_001.html>.
- Ofner2017: <https://moabb.neurotechx.com/docs/generated/moabb.datasets.Ofner2017.html>, paper DOI `10.1371/journal.pone.0182578`.
- Cho2017: <https://moabb.neurotechx.com/docs/generated/moabb.datasets.Cho2017.html>, data DOI `10.5524/100295`.
- Lee2019 MI / OpenBMI: <https://moabb.neurotechx.com/docs/generated/moabb.datasets.Lee2019_MI.html>, data DOI `10.5524/100542`.
- Restricted arm/XVA record: <https://zenodo.org/records/18470278>.

Always check the source record again before redistribution. Registry metadata is a safeguard, not legal advice.

## Verified source formats and channel contracts

The importers are checked against published release sidecars, current NEMAR sourcedata manifests, and MOABB 1.7.2 loader metadata before standardized archives are written.

| ID | Source actually consumed | Native EEG | Hz | Cyton target handling |
|---|---|---:|---:|---|
| `openneuro_ds008446` | BIDS EDF + TSV v1.0.1 | 62 scalp + 2 references; marker separate in TSV | 512 | all eight exact |
| `physionet_eegmmidb` | EDF+ annotations | 64 | 160 | all eight exact |
| `bnci2014_001` | per-session MAT via MOABB/NEMAR | 22 | 250 | C3/Cz/Pz/C4 exact; F3/P3/F4/P4 spherical-spline interpolated |
| `ofner2017` | GDF via MOABB/NEMAR, imagery session only | 61 | 512 | all eight exact |
| `cho2017` | per-subject MAT via MOABB/NEMAR | 64 | 512 | all eight exact |
| `lee2019_mi` | two-session MAT via MOABB/NEMAR | 62 (+4 EMG excluded) | 1000 | all eight exact |

BNCI is the only dataset that does not natively contain the complete confirmed Cyton montage. Nearby BNCI electrodes are **not renamed** as missing nodes. Its 22 real sensor locations are retained while MNE spherical-spline interpolation reconstructs only `F3,P3,F4,P4`; those names are recorded in archive provenance. Any missing channel outside that explicit BNCI allow-list fails the import. This preserves the required eight-channel order without pretending that `FC3` is `F3` or that `CP3` is `P3`.

Every provider is checked for its expected event vocabulary, verified native sampling rate or documented subject-specific rate exception, EEG channel count, required metadata, output labels, finite values, and final `(trials, 8, 750)` shape. The standardized archive passband defaults to **1–40 Hz** and is stored in preprocessing provenance. Processed archives are reused only when their license, units, 250 Hz rate, channel order, sample shape, category, source, subject count **and passband** still match the current contract; older 8–30 Hz archives are therefore re-imported rather than silently reused.

The detailed audit is written to `reports/public_mi/data_format_audit.md`.

## Install optional support

From the project root on Windows:

```powershell
python -m pip install -e ".[public-mi]"
```

For EEGConformer and TCFormer:

```powershell
python -m pip install -e ".[transformer]"
```

Install the PyTorch build appropriate for the computer's CPU/GPU first when needed. The core SDK and classical personal models do not require PyTorch, Braindecode or MOABB.

## List datasets

```powershell
python -m ML.public_mi --list
```

## One-command automated workflow

`tools/prepare_public_mi.py` is the recommended full workflow. It:

1. displays every source license and skips the restricted Zenodo record;
2. downloads OpenNeuro ds008446 with `openneuro-py`;
3. downloads only PhysioNet imagery runs and verifies their published SHA-256 checksums;
4. confines MOABB downloads to workflow-owned staging;
5. imports standardized, separate category archives;
6. trains each requested candidate on subject-held-out folds;
7. selects the best candidate separately for `finger`, `hand`, `upper_limb`, and `feet`;
8. publishes `models/public/<category>/category_model.joblib` plus `models/public/model_manifest.json`;
9. verifies that every public model has `runtime_ready=False`;
10. generates the full cohort/source/literature report, actual archive inventory and class-separation figures while the archives still exist;
11. applies the selected cleanup policy only after models and evidence pass verification.

The default `staging-only` policy preserves standardized archives. The explicit
`models-reports` policy removes both workflow staging and the six named processed
dataset directories after verified evidence generation, retaining selected models,
reports/figures and all personal recordings/profiles. It never trains a universal
all-body classifier.

Inspect the complete plan without downloading:

```powershell
python tools/prepare_public_mi.py --dry-run
```

Run all public datasets with classical candidates while keeping processed archives:

```powershell
python tools/prepare_public_mi.py --accept-licenses
```

For the requested local-machine policy—retain only selected models and complete
reports after success—use:

```powershell
python tools/prepare_public_mi.py `
  --accept-licenses `
  --cleanup-policy models-reports `
  --hash-evidence-inputs
```

The second command creates the evidence report and figures **before** deleting the
processed archives. On failure it retains staging and processed data for diagnosis
unless `--cleanup-on-failure` is explicitly supplied. Do not combine
`--cleanup-policy models-reports` with `--keep-downloads`.

`--pipelines auto` is the default. It always compares `riemann_lr`, `riemann_svm`, and the audited 8–30 Hz `fbcsp_lda`; if both PyTorch and Braindecode are installed, it also compares EEGConformer and TCFormer. To request them explicitly:

```powershell
python tools/prepare_public_mi.py `
  --accept-licenses `
  --pipelines riemann_lr,riemann_svm,eegconformer,tcformer `
  --epochs 40
```

A smaller pilot can limit each dataset, although enough complete subjects are still required for held-out folds:

```powershell
python tools/prepare_public_mi.py `
  --accept-licenses `
  --subjects-per-dataset 5
```

Verified processed archives are reused on later runs unless `--refresh` is supplied; a limited pilot archive or an archive with stale license/channel/sample metadata is not mistaken for a current import. Before downloading, a conservative subject-scaled disk-space estimate is checked against current sourcedata manifests. The current full-source planning values sum to about **112 GB**, not 80 GB (approximately 5.5 GB OpenNeuro, 1.7 GB PhysioNet imagery-only, 1 GB BNCI, 28 GB Ofner, 11 GB Cho, and 65 GB Lee), and the default peak free-space threshold is roughly **154 GB** to allow conversion and candidate models. Provider sizes may change. `--min-free-gb` can override that threshold only after the user checks actual provider sizes. Successful runs clean staging by default. Use `--keep-downloads` only when raw/cache retention is intentional. On failure, staging is retained for diagnosis and resumable HTTP downloads; `--cleanup-on-failure` opts into deleting it.

OpenNeuro ds008446 v1.0.1 requires release-specific event reconstruction. Its `_events.tsv` sidecars contain one row per EEG sample with repeated marker values, and their published `onset` values carry an extra sampling-rate divisor. For each EDF, the importer reads the matching sidecar, collapses consecutive marker blocks, keeps finger cues `3` through `7`, and reconstructs the EDF sample as `round(onset * sfreq**2)`. Every reconstructed cue is checked against that EDF's sample count before epoching. This avoids both the earlier `No matching events found for 3` failure and the later empty-epoch failure caused by relying on the EDF annotation/marker channel.

After an import failure, rerun the same preparation command **without** `--refresh`. The retained `dataset/public/work/auto/downloads/` staging files are reused, so an already completed OpenNeuro download is not fetched again.

The MOABB importer supplies both the selected `events` and an explicit matching `n_classes`. This is required by MOABB releases whose `MotorImagery.used_events()` does not infer `n_classes`; without it, BNCI2014-001 can stop with `TypeError: '<' not supported between instances of 'int' and 'NoneType'`. Before downloading/epoching continues, the importer also checks the selected dataset's verified event and channel contract. After loading, it checks MOABB metadata, sampling, labels and dimensions, then performs the explicit BNCI-only channel interpolation described above. Its epoch extraction is compatible with both the MNE 1.5 and current `get_data()` signatures. Updating the importer is sufficient—do not uninstall MOABB or delete its download cache.

Reports are written under `reports/public_mi/`, including `latest_workflow.json`. The restricted `zenodo_18470278` source is always skipped because its files and reuse terms are not publicly automatable.

## Report-only improvement experiment

`tools/compare_public_mi.py` compares stronger, explicitly documented hypotheses without replacing anything under `models/public/`:

```powershell
python tools/compare_public_mi.py
```

It tests category-specific bands, filter-bank CSP, and an unlabeled per-subject Riemannian-alignment variant. Hand trials are harmonized to the common `left_hand` versus `right_hand` task. Feet trials use only sources that independently contain both `idle` and `both_feet`, followed by equal sampling in every source/label cell. This currently excludes BNCI feet from the binary feet comparison because BNCI contributes only `both_feet`; combining that with PhysioNet-only idle would let source identity predict the class.

Every candidate receives subject-held-out evaluation. Leave-one-dataset-out evaluation is reported only when at least two datasets each contain every evaluated class. It is therefore meaningful for harmonized hand data, but unavailable rather than fabricated for the current single-source finger/upper-limb tasks and source-complete feet task. Riemannian alignment is reported as **transductive**: it uses all epochs from each held-out subject to estimate an unlabeled geometric-mean covariance, never their labels.

The normal archives are imported with a broad **1–40 Hz** passband. This preserves delta activity from 1 Hz upward plus the full mu/beta range, while classical FBCSP and tangent-space candidates may still apply their own training-fold 8–30 Hz feature filters. The default archives still cannot recover Ofner's 0.3–1 Hz movement-related cortical-potential content. The comparison detects this from archive provenance and marks a recipe requiring 0.3 Hz `skipped` rather than producing misleading numbers.

To test the upper-limb low-frequency recipes, create a separate broad-band experiment archive. Do not overwrite the successful normal archives or selected models:

```powershell
python tools/prepare_public_mi.py `
  --accept-licenses `
  --subjects-per-dataset 5 `
  --pipelines riemann_lr `
  --import-bandpass 0.3,40 `
  --processed-root dataset/public/experiments/broadband/processed `
  --model-root dataset/public/experiments/broadband/baseline-models `
  --work-root dataset/public/experiments/broadband/work `
  --report-root reports/public_mi/experiments/broadband-preparation

python tools/compare_public_mi.py `
  --processed-root dataset/public/experiments/broadband/processed `
  --report-root reports/public_mi/experiments/broadband-comparison
```

Because the earlier successful workflow cleaned its staging downloads, the first broad-band import must download source data again. Later runs reuse the separate processed archives only when their recorded passband and all other archive contracts match. Experiment outputs are timestamped JSON and CSV files under `reports/public_mi/experiments/`; no experiment candidate is promoted automatically.

## Full evidence report and class-separation figures

`tools/generate_public_mi_evidence.py` generates one reproducible report from the standardized archives and selected-model metadata:

```powershell
python tools/generate_public_mi_evidence.py
```

Outputs are written under `reports/public_mi/evidence/`:

- `EVIDENCE_REPORT.md` — authoritative dataset source, participant cohort details (including explicit missing demographics), source trial design, published comparator result, actual imported counts, historical audit values and recomputed grouped results;
- `evidence_report.json` — cohort metadata, archive provenance/fingerprints, split policy, fold details, model-info availability, and plot fit boundaries;
- `dataset_source_literature_comparison.csv` — source, cohort, literature protocol/metric and project comparison table;
- `dataset_archive_inventory.csv` — actual imported participants, sessions, trials, channels, samples per trial and total channel-sample count for every dataset/category;
- `figures/<category>_csp_class_separation.png` — classes in distinct colors;
- tangent-feature PCA, grouped out-of-fold confusion matrix, class covariance summaries and complete-subject fold scores.

The class-separation figures use one declared representative subject-held-out fold. CSP, the tangent Fréchet mean, feature scaling and PCA are fitted on training subjects only; colored points are held-out subjects. Balanced accuracy is computed separately from all grouped out-of-fold predictions. The report never converts published kappa to accuracy or equates a within-subject literature score with project cross-subject/source balanced accuracy.

If `dataset/public/processed/` is absent, the generator still writes the authoritative source/literature and explicitly labelled historical-audit table, but emits no synthetic plots or recomputed metrics. Restore the actual `epochs.npz` archives and rerun. Add `--hash-inputs` when full archive SHA-256 fingerprints are required.

## Manual download and import

### Finger: OpenNeuro ds008446

Download BIDS data from the source page into:

```text
dataset/public/raw/openneuro_ds008446/
```

Then run:

```powershell
python -m ML.public_mi --dataset openneuro_ds008446 --input dataset/public/raw/openneuro_ds008446
```

The importer pairs every `*_eeg.edf` with its `*_events.tsv`, reconstructs v1.0.1 finger-cue block onsets with the release-specific squared-rate correction, keeps only the requested motor channels, resamples to 250 Hz, stores MNE-scaled volts, and writes:

```text
dataset/public/processed/openneuro_ds008446/finger/epochs.npz
```

The five labels remain `finger_thumb_unsided` through `finger_pinky_unsided` because the released task labels do not identify a left or right hand.

### Hand and feet: PhysioNet EEGMMIDB

Download from PhysioNet into `dataset/public/raw/physionet_eegmmidb/`. Then:

```powershell
python -m ML.public_mi --dataset physionet_eegmmidb --input "dataset/public/raw/physionet_eegmmidb/S*/*.edf"
```

Only imagery runs `04, 06, 08, 10, 12, 14` are accepted. The importer writes separate `hand` and `feet` archives and context-qualifies every T0/T1/T2 event by run number.

The release is not uniformly 160 Hz: imagery/execution EDFs for subjects **S088, S092 and S100 are 128 Hz**, a known exception also documented by MOABB. The importer validates 128 Hz only for those three subject IDs, validates 160 Hz for the others, and resamples each recording with its events to the common 250 Hz archive rate. The 1–40 Hz archive passband remains below the exceptional recordings' 64 Hz Nyquist frequency. Provenance records the native rates actually observed. See <https://moabb.neurotechx.com/docs/generated/moabb.datasets.PhysionetMI.html>.

### MOABB datasets

Examples:

```powershell
python -m ML.public_mi --dataset bnci2014_001 --subjects 1,2,3
python -m ML.public_mi --dataset ofner2017 --subjects 1,2,3
python -m ML.public_mi --dataset cho2017 --subjects 1,2,3
python -m ML.public_mi --dataset lee2019_mi --subjects 1,2,3
```

MOABB manages its download cache. Imported archives still go under `dataset/public/processed/`. Omitting `--subjects` imports every available subject and can require substantial disk space and time.

## Standardized archive schema

Every processed NPZ contains:

- `X`: `(trials, channels, samples)`;
- `labels`: canonical labels that preserve source meaning;
- `source_labels`: original event labels;
- `subjects` and `sessions`;
- `source_dataset`, `category`, `channel_labels`, `fs`, and `units`;
- `imagery_or_execution`;
- `license`, `citation`, and download source;
- JSON preprocessing metadata and schema version.

The writer rejects category mixing, undeclared source mappings, implicit units, duplicate channels, NaN/inf values, and unsupported paradigms. The archive loader defaults to imagery only.

## Train a public benchmark/pretraining model

The public hand head is deliberately binary: `left_hand` versus `right_hand`. Public `idle` and `both_hands` epochs stay in the provenance-rich archives but are excluded during hand-model fitting because they are not shared by all four hand providers and do not match the personal universal hand vocabulary. The model bundle records this as training policy `cross-source-left-versus-right-hand-v1`.

The public feet head is also guarded against source-label confounding. BNCI contributes `both_feet` but no equivalent idle class, so its feet-only trials are not pooled with PhysioNet-only idle. Production fitting retains only providers that independently contain both `idle` and `both_feet`, then deterministically balances every source/label cell. With the current registry this means an honest PhysioNet-only binary feet model; the excluded BNCI archive remains preserved. The bundle records policy `source-complete-idle-versus-both-feet-v1`.

Candidate bundles record both whole-subject CV and leave-one-dataset-out CV. When at least two source datasets are class-complete—as they are for binary hand imagery—the automated workflow selects by dataset-held-out balanced accuracy first. Single-source categories continue to use subject-held-out accuracy. This prevents a small gain on a provider-specific subject split from outranking stronger cross-provider generalization.

Classical baseline:

```powershell
python -m ML.train_public_mi dataset/public/processed/cho2017/hand/epochs.npz --category hand --pipeline riemann_lr
```

EEGConformer or TCFormer:

```powershell
python -m ML.train_public_mi dataset/public/processed/cho2017/hand/epochs.npz --category hand --pipeline eegconformer --epochs 40
python -m ML.train_public_mi dataset/public/processed/bnci2014_001/hand/epochs.npz --category hand --pipeline tcformer --epochs 40
```

Public evaluation holds out entire subjects. Random epoch splitting is not used because it leaks subject-specific EEG patterns. The saved bundle has `runtime_ready=False`.

EEGConformer is the maintained Braindecode convolution-plus-Transformer implementation described at <https://braindecode.org/stable/generated/braindecode.models.EEGConformer.html>. TCFormer is the maintained temporal-convolutional Transformer described at <https://braindecode.org/stable/generated/braindecode.models.TCFormer.html> and the authors' repository <https://github.com/Altaheri/TCFormer>.

## Fine-tune into a personal category specialist

First collect personal sessions with the normal recorder. Then use only a public model with exactly matching channel labels/order, sampling rate and epoch length:

```powershell
python ML/train_intent_mi.py anson --intents left_hand,right_hand --category-specialist --pipelines eegconformer --pretrained models/public/hand/category_model.joblib --transformer-epochs 30
```

The command validates category, channel order, sample rate and epoch length before transfer. Matching weight tensors initialize the personal model; an incompatible classification head is skipped and refit. Personal evaluation remains session-held-out.

For small personal datasets, keep the classical CSP/Riemannian baselines and compare on the same held-out sessions:

```powershell
python ML/train_intent_mi.py anson --intents left_hand,right_hand --category-specialist --pipelines riemann_svm,riemann_lr
```

A Transformer trained from scratch on a few eight-channel personal sessions is not assumed to be better.

## Runtime behavior

Existing game APIs do not change:

```python
bci = bci_sdk.LiveBCI(
    buttons,
    source="intent-mi",
    user="anson",
    intent_map={"left": "left_hand", "right": "right_hand"},
)
```

`LiveBCI` calls `MotorIntentModel.load_resolved()`. It selects personal category models from the game's intent map. A finger-only game loads the finger specialist; a hand-plus-feet game loads both specialists through a router and uses the highest accepted prediction. If a specialist predicts another same-category class not mapped by the game, that output is suppressed as no action. An exact task-set model is used only as an in-app-calibration fallback when category models are missing. Public benchmark models are never served directly.

## Claims that must not be made

- A successful download/import proves only data plumbing.
- Public subject-held-out accuracy is not this user's Cyton accuracy.
- Synthetic/dry-run success is not motor-imagery validation.
- Motor execution is not motor imagery.
- Physical movement during an imagery trial is contamination.
- Eight-channel fine-finger decoding remains a research goal; software support does not establish biological separability.
