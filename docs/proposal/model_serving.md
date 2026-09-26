# Serving the model

Author: Anson · 26/9/2026 · describes `ML/sync.py`

Requirement: no raw data online, no server-side training, contributions optional.

---

## 1 · Models cannot be pushed and pulled

The intuitive design — pull the shared model, fit it on local data, push it back
— does not accumulate, because scikit-learn has no incremental path:

```
Pipeline step               supports partial_fit?
  BandpassFilter                    False
  CSP                               False
  LinearDiscriminantAnalysis        False
```

`.fit()` resets the estimator, so everything pulled is discarded before the push.
Three simulated users taking turns:

```
after user 0 pushes: accuracy on [u0,u1,u2] = 47%, 53%, 60%
after user 1 pushes: accuracy on [u0,u1,u2] = 50%, 50%, 50%
after user 2 pushes: accuracy on [u0,u1,u2] = 50%, 50%, 47%
```

Last writer wins. Merging the weights instead would not help either: cosine
similarity between two users' first CSP filter is **0.101**, near-orthogonal. The
filters encode one person's electrode placement and motor cortex, so their
average describes nobody.

---

## 2 · Features are pushed instead

`ML/sync.py::maybe_sync()`:

```
1. EXPORT  profiles/<user>/*.npz → TangentSpaceMapper → features.npz
           F: n×36 float32, y: n int8
2. STAGE   copy to shared/<user>_features.npz
3. PUSH    git add / commit / push origin HEAD:shared-features
4. STAMP   last_sync → profiles/sharing.json
```

Features are data, so appending accumulates; the shared model is a fresh `.fit()`
over the combined set, so the partial-fit problem does not arise. Each user
writes a separate file, so simultaneous syncs cannot overwrite each other.

A feature vector is 36 numbers per 3 s trial — the upper triangle of a covariance
matrix in tangent space. The transform is not invertible, so no recording leaves
the machine. Tangent space is used rather than CSP output because it is defined
relative to a shared reference point and is therefore comparable across people.

| Item | Size |
|---|---|
| One session, raw EEG | 3000 KB |
| Same session, as features | 8.8 KB |
| `export_features()`, 192 trials | 25.9 KB |
| 100 users × 5 sessions, features | 4.3 MB |
| Raw EEG at that scale | 0.4 GB |
| Model on disk: csp_lda / riemann_svm | 1.9 KB / 11.4 KB |

GitHub is sufficient at these sizes. `git push` is the entire transport layer:
no API, no database, and the 30-day timer lives in `profiles/sharing.json` on the
client, so nothing runs server-side.

Note: `csp_lda` models could not be saved at all until `_Band` was moved out of
`make_csp_lda()` — Python cannot pickle a class defined inside a function. All
four pipelines now save and reload with identical predictions.

---

## 3 · Architecture

```
pip install bci-sdk
      │  ships starter_model.joblib (public PhysioNet data)
      ▼
FIRST RUN → 2 min calibration → personal model, saved locally
      │
      ├── inference 1.3–4.9 ms, offline, never contacts a server
      │
EVERY SESSION → profiles/<user>/calib_*.npz accumulates
      │
TRAIN MORE (36–150 ms) — refit over the whole history
      │
      └── opt in once → automatic every 30 days, background thread
            uploads tangent-space features only, never raw EEG
```

### Train more, not retrain

Sessions are appended, never replaced, and the model is refit over the entire
history:

```
1 session   20 trials → 76.7%
2 sessions  44 trials → 80.0%
3 sessions  71 trials → 81.7%
4 sessions  90 trials → 83.3%   (plateau)
```

Refit-over-all rather than incremental updates because: no pipeline supports
`partial_fit` (CSP solves a generalised eigenproblem over the full covariance,
which has no online form); a refit costs 36–150 ms, so approximating buys
nothing; and refitting is the exact optimum for the accumulated data, whereas
incremental updates drift and can forget when one session is unrepresentative.

### Usage

```bash
python ML/sync.py anson --enable-sharing --days 30   # once, explicit
```

```python
from ML.sync import train_more, maybe_sync_background

train_more("anson")             # fold in the latest session, report the delta
maybe_sync_background(user)     # at app start; daemon thread, no-op unless due
```

Consent is one-time and revocable (`--disable-sharing`). Until it is granted,
`maybe_sync()` returns `{"status": "not_enabled"}` and touches the network zero
times. Every failure path — offline, no credentials, rejected push, timeout — is
caught and returned as a status dict, then retried at the next interval; the
application works without ever syncing.

`bci_sdk/__init__.py` does not re-export these yet, so applications must import
from `ML.sync` directly.

---

## 4 · Limits

Pooling gives a weak shared model: 1 → 3 users moved accuracy 50.8% → 55.0% on
synthetic cross-subject data. Motor imagery is individual, so the pooled model is
a cold-start substitute — roughly 65% instead of 50% for a first-time user — not
a replacement for personal calibration.

All of the above assumes MI separates on this hardware for at least one person,
which is unmeasured. The go/no-go test is Cohen's d > 0.8 (Testing §2.2). Pooling
non-separable datasets yields a larger non-separable dataset.

---

## 5 · Status

Built (`ML/sync.py`):

| Function | Does |
|---|---|
| `train_more(user)` | refit over all sessions, report added trials and accuracy delta |
| `enable_sharing` / `disable_sharing` / `sharing_status` | one-time consent in `profiles/sharing.json` |
| `export_features(user)` | tangent-space features, n×36 float32 |
| `sync_due(user)` | has the interval elapsed |
| `maybe_sync(user, dry_run=)` | export, stage, commit, push, stamp |
| `maybe_sync_background(user)` | the above on a daemon thread |

CLI: `python ML/sync.py <user> --train | --enable-sharing --days 30 | --disable-sharing | --status | --sync [--dry-run]`

Outstanding:

| Step | Effort |
|---|---|
| Train `starter_model.joblib` on PhysioNet in Colab | 0.5 d |
| Bundle it (`get_model()` already falls back to it) | 0.2 d |
| Shared retrain script — read `shared/*.npz`, pool, fit, publish | 0.3 d |
| Re-export sync API from `bci_sdk/__init__.py` | 0.05 d |
| Consent prompt in UI rather than CLI-only | 0.3 d |

The last two follow hardware validation; `maybe_sync()` pushes per-user files
specifically to defer the pooling decision until then.
