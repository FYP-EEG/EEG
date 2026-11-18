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
| `lee2019_mi` | hand | left/right hand grasp imagery | public, CC0 1.0 | Labelled training runs only; unlabelled test runs are excluded. |
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

## Download and import

### Finger: OpenNeuro ds008446

Download BIDS data from the source page into:

```text
dataset/public/raw/openneuro_ds008446/
```

Then run:

```powershell
python -m ML.public_mi --dataset openneuro_ds008446 --input dataset/public/raw/openneuro_ds008446
```

The importer keeps only the requested motor channels, resamples to 250 Hz, stores MNE-scaled volts, and writes:

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

`LiveBCI` calls `MotorIntentModel.load_resolved()`. It loads an exact task-set model first, then a compatible personal category model. If a category specialist predicts another same-category class not mapped by this game, that output is suppressed as no action. It never falls back to an unrelated all-body classifier or serves a public benchmark directly.

## Claims that must not be made

- A successful download/import proves only data plumbing.
- Public subject-held-out accuracy is not this user's Cyton accuracy.
- Synthetic/dry-run success is not motor-imagery validation.
- Motor execution is not motor imagery.
- Physical movement during an imagery trial is contamination.
- Eight-channel fine-finger decoding remains a research goal; software support does not establish biological separability.
