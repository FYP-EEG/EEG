# Reusable motor-intent SDK

**Author:** Anson  
**Created on:** 19/10/2026  
**Purpose:** Developer and lab workflow for mapping reusable finger/body MI models to application actions.

## Core contract

The model predicts a physical intention:

```text
left_index, right_pinky, left_hand, both_feet, ...
```

The developer maps it to an application action:

```text
left_index -> rock
both_feet  -> jump
```

One universal recording is stored as subject-labelled training data:

```text
dataset/motor_intent/calib_<subject>_<timestamp>.npz
```

It is automatically filtered into independent personal classifiers:

```text
profiles/<user>/motor_intent/_categories/finger/intent_model.joblib
profiles/<user>/motor_intent/_categories/hand/intent_model.joblib
profiles/<user>/motor_intent/_categories/feet/intent_model.joblib
```

Games select models from their action-to-intent map. Rock–Paper–Scissors and Realistic UI load the finger model. Subway Runner loads both hand and feet models through a category router. An output not mapped by the active game is suppressed as no action. Exact task-set models remain only as an in-app-calibration fallback if category models have not yet been generated.

Each NPZ retains subject, session, montage, sampling-rate and label metadata. Personal category training filters to the requested subject, so another participant's universal recording is never mixed into the model. A public benchmark is never loaded directly as personal calibration. Dataset imports and optional EEGConformer/TCFormer pretraining are documented in `PUBLIC_MI_DATASETS.md`.

## Universal reusable body-intent recording

The recommended first recording collects a shared pool containing all ten fingers,
left hand, right hand, both feet, and idle. It uses local CC0 hand/foot outline
assets; the cued finger, whole hand, or both feet change colour while the real body
remains still.

```bash
python EEG/data_record.py --universal anson
```

This short command uses the current defaults: direct Cyton on COM8, ThinkPulse gain 8, `cyton8_motor`, eight trials per active intention, 15 idle trials, and automatic finger/hand/feet model training. Override individual options only when the hardware or protocol actually changes.

This records 13 active intentions x 8 trials plus 15 idle trials (119 total) into:

```text
dataset/motor_intent/calib_anson_*.npz
```

The universal recording is a reusable labelled data pool, not one forced 14-class runtime model. It is split into finger, hand and feet specialists. A game loads only its required specialist, or a router over multiple specialists when its actions cross categories.

Running the same command with `--dry-run` writes only under `dry_runs/` and prints a
warning not to train from synthetic data.

Before every real direct-Cyton MI recording, `data_record.py` calls the mandatory gate in `startup/check_headband.py` before loading the app or starting any cue:

1. state the selected electrode mode's physical wiring contract without asking for typed confirmation (passive SRB/reference plus BIAS, or powered ThinkPulse with one earlobe reference on SRB2 and the physical BIAS lead disconnected); software records that this wiring is not machine-verifiable;
2. transfer internal ground on all 8 channels;
3. transfer the known approximately 2 Hz square wave on all 8 channels;
4. restore Cyton defaults with `d`, then apply the selected all-channel Normal-input profile;
5. capture live electrodes and require all 8 channels to pass sample-rate, finite, flatness, signal-adaptive amplitude/jump and clipping checks;
6. if any stage fails, block recording/cues while keeping the recorder and hosted example alive; print each distinct root failure once and replace one in-place `Silent repeated failures: N` counter for identical polls; write one startup HTML/JSON only after the valid session reaches `READY`.

Run the same gate independently before non-recorder headset use:

```bash
python startup/check_headband.py --port COM8 --montage cyton8_motor \
  --out reports/startup/headband_pre_use
```

A failed startup check does not return from `data_record.py` or terminate an example hosted by it. The terminal prints a distinct root failure and corrective actions once; identical failures replace one in-place counter line rather than printing repeatedly. If Cyton is disconnected/no samples arrive, dependent square-wave and Normal-input stages are `SKIP`, not extra failures. For known ASCII Cyton commands, the shared command helper tolerates only BrainFlow's optional response-text `UnicodeDecodeError` (for example a raw standalone `A0`, which is not valid UTF-8) after native command success; actual success is then established from captured ground/square/Normal samples. Other command and hardware exceptions still block startup. BrainFlow 5.22.0 or newer remains required. `--startup-retry-delay` changes the polling interval, and `Ctrl+C` remains the deliberate operator-abort path. Failed polls write no reports.

After startup, `data_record.py` opens the actual recording backend and repeats a fresh still-baseline preflight. Connection or signal failure also leaves the recorder/example alive; identical failures update one in-place counter, while a new diagnostic block is printed only if the failure changes. HTML/JSON under `reports/recording_quality/<subject>/` is written only for the preflight that proceeds into a valid recording session (or an explicitly forced debugging session). `--force-record` cannot bypass a failed all-channel startup gate.

OpenBCI ThinkPulse active electrodes must use `--electrode-mode thinkpulse` (default gain 8, or explicitly `--cyton-gain 8`). In that mode both startup and the recording backend power all channels in Normal input, disable BIAS generation, keep SRB2 on/SRB1 off, and reapply the same profile after the normal-default reset. Connect one earlobe reference to SRB2. The physical BIAS lead—including a second earlobe lead formerly used as BIAS—must remain disconnected. Startup states this contract but does not ask for `YES` or claim that software inspected the wiring. Passive electrodes retain gain 24 and physical BIAS by default. ThinkPulse gains above 8 are rejected. If all eight channels show actual clipping/railing, startup automatically retries at 4, then 2 and 1, and passes the selected gain into the recording backend.

The `cyton8_motor` physical order is `F3, C3, Cz, P3, Pz, F4, C4, P4` for Cyton channels 1–8. This is the only motor montage used by the current universal/category-model workflow.

### Adaptive amplitude handling and apparent flatness

Startup and offline trial screening no longer assume a fixed default amplitude/jump cap. They derive robust per-channel scale and sample-step limits from the recording, compare each channel with its peers, and separately detect repeated extreme plateaus. Stable common amplification and stable DC offsets can pass; a gross local scale outlier, abrupt step, flat channel or clipped plateau fails. An explicit cap remains available only for a controlled diagnostic experiment.

An OpenBCI GUI trace that is outside its display range can appear flat in this code for two different reasons:

1. the value is a large but stable DC offset, and median-centring/high-pass filtering intentionally removes that offset while retaining its variation; or
2. the ADC is saturated, so many consecutive raw samples equal one extreme. Once centred or filtered, that plateau has almost no variation.

The report therefore shows the raw median, peak around the median, adaptive limit, standard deviation and repeated-extreme fraction. It does not “fix” railing by clipping or normalizing the data.

### OpenBCI GUI CSV offline input

Use a GUI export to check acquisition-consuming scripts without a board:

```bash
python EEG/data_record.py --universal anson \
  --eeg-csv "path/OpenBCI-RAW.csv" --no-auto-train
```

The parser recognizes `%` metadata, sample rate (or infers it from numeric timestamps), and named `EXG Channel 0` through `EXG Channel 7` columns even when accelerometer/analog columns move. Replay loops at wall-clock rate and undergoes the same adaptive preflight. The resulting NPZ is marked as an offline pipeline check and is excluded from automatic category-model training because replayed EEG is not related to the new cue times.

Other scripts can use:

```python
import bci_sdk
from bci_sdk import OpenBCICSVBackend

backend = OpenBCICSVBackend("path/OpenBCI-RAW.csv", loop=True)

# Or through the high-level SDK:
session = bci_sdk.BCISession(source="csv", eeg_csv="path/OpenBCI-RAW.csv")
```

A standalone CSV preflight is also available:

```bash
python EEG/signal_preflight.py --csv "path/OpenBCI-RAW.csv" \
  --out reports/signal_preflight/openbci
```

```bash
python EEG/data_record.py --universal anson
```

After the NPZ is saved, the inspection HTML/JSON contains a recording `ACCEPT`/`REJECT` indicator, ordered rejection-step indicators (amplitude, flatness, jump, within-class outlier and noncompliance), and a per-trial table naming both the rejecting rules and responsible channels. This makes a rejection traceable instead of reporting only one aggregate dropped-trial count.

## Start page and in-app Calibration page

All three demos open on a two-choice **GAME / CALIBRATION** landing page. Both choices
are mouse-clickable; use Left/Right and Enter from the keyboard. Choosing Calibration
opens the reusable calibration panel, which shows the action-to-body mapping on the
same two-hand/two-feet diagram.

The Calibration page provides the complete local flow:

1. Review each application action and assigned motor intention.
2. Use Up/Down to choose an action and Left/Right to change its intention.
3. Press D or click SOURCE to choose DRY RUN or CYTON.
4. Press Enter/START CALIBRATION.
5. Follow the in-app cues; no physical movement or clicking is required.
6. The Cyton recording is saved and training starts automatically.
7. The result page shows pipeline, held-out accuracy, session count and reliability.
8. The Calibration tab also shows recent `INTENDED → MODEL PREDICTION` rows. These are clearly labelled as a training-set preview; held-session accuracy remains the generalisation metric.

Launch a real calibration directly inside RPS:

```bash
python example/rockpaperscissors.py --user anson --calibration-source cyton
```

Use F2, then Enter. The tab stops a running BCI before opening the board and, after successful real training, can restart `intent-mi`. A dry run checks only timing/UI and deliberately saves no EEG model. One block trains a usable experimental model but cannot produce a trustworthy session-held-out score; record at least three separate blocks. During training, the loader combines the selected app-specific profile calibration with matching labels from the generalized `dataset/motor_intent/` pool. Legacy `_universal_13` profile recordings remain readable.

Default in-app block sizes provide screening headroom:

```text
RPS:          36 action + 12 idle (12 per action)
Subway:       36 action + 12 idle (12 per action)
Realistic UI: 60 action + 12 idle (12 per action)
```

The external universal workflow is the primary recording path: `python EEG/data_record.py --universal <subject>`.

### Raw versus filtered storage

`EEG/data_record.py` saves the epoched BrainFlow samples in microvolts without software band-pass filtering, normalization, covariance conversion or tangent-space transformation. The NPZ is therefore the reusable raw sensor-level epoch data (apart from the Cyton/ADC and BrainFlow's count-to-microvolt conversion). Trial screening filters temporary copies only. The Riemannian training pipeline applies its 8–30 Hz filter and feature transformations during `fit()`/`predict()`; those transformed arrays are not written back into the recording NPZ.

Before direct Cyton recording, the backend now sends the Cyton `d` command to restore default normal channel inputs. `tools/test_signal_check.py` also restores `d` after internal-ground or square-wave self-tests. This prevents persistent hardware test-signal settings from being mistaken for human EEG in the next calibration.

## Supported intentions

```text
left_thumb     right_thumb
left_index     right_index
left_middle    right_middle
left_ring      right_ring
left_pinky     right_pinky

left_hand      right_hand      both_hands
left_foot      right_foot      both_feet
```

The API accepts one to ten finger intentions. It emits warnings for adjacent fingers, large same-hand class sets and 6–10-finger research-only profiles. Software support is not an accuracy claim.

## Rock–Paper–Scissors

Default:

```text
left_index  -> rock
right_index -> paper
right_pinky -> scissors
```

This game selects the personal **finger** model produced by the universal recording.

```bash
python example/rockpaperscissors.py --bci intent-mi --user anson
```

Override the action mapping while retaining the same finger model:

```bash
python example/rockpaperscissors.py --bci intent-mi --user anson \
  --intent-map "rock=right_pinky,paper=left_index,scissors=right_index"
```

## Subway-style runner

Default:

```text
left_hand  -> move_left
right_hand -> move_right
both_feet  -> jump
```

This game loads the personal **hand** and **feet** models and routes the highest accepted category prediction.

```bash
python example/subway_run.py --bci intent-mi --user anson
```

Keyboard fallback: LEFT/A, RIGHT/D, UP/W. Space is reserved for pause/resume.

## Realistic UI

Default five-action mapping:

```text
left_thumb  -> inventory
left_pinky  -> message
right_index -> shoot
right_pinky -> reload
right_thumb -> map
```

The SDK marks this ADVANCED and warns about the adjacent right-thumb/right-index pair. This game selects the same personal **finger** model as Rock–Paper–Scissors.

```bash
python example/realistic_ui.py --bci intent-mi --user anson
```

## Gameplay feedback and separate performance monitor

On the GAME page of all three demos:

```text
Space        pause/resume the game and BCI dispatch
Left click   mark the latest classified output correct
Right click  mark the latest classified output incorrect
```

When BCI is active, mouse clicks are feedback controls rather than game-button clicks. Feedback is explicit human annotation only; it is logged but never used for automatic retraining. Because a physical click can add EMG/movement contamination, classification is suppressed and queued windows are discarded for 3.2 seconds after feedback or resume.

Gameplay sessions write:

```text
runtime_logs/<app>/<timestamp>/
    events.jsonl      # predictions, state changes and manual feedback
    status.json       # live summary and manual correctness metrics
    latest_eeg.npy    # latest raw 8-channel window
```

Open the separate dashboard in another terminal:

```bash
python example/performance_monitor.py
```

It visualizes the latest eight-channel EEG window, prediction state, confidence, dispatch/rejection counts and manually marked accuracy. Use `--app rockpaperscissors` or `--session <directory>` to select a specific stream.

## One-command layered debugger

Use one entry point to localize the first failing layer across live hardware, saved raw NPZ, 8–30 Hz filtering, within-session class separability, held-session SVM/logistic-regression comparison and optional gameplay logs:

```bash
python tools/debug_bci_pipeline.py \
  --port COM8 --montage cyton8_motor \
  --input "dataset/motor_intent/calib_brian_*.npz" \
  --out reports/brian_pipeline_debug
```

Offline-only use:

```bash
python tools/debug_bci_pipeline.py \
  --input "dataset/motor_intent/calib_brian_*.npz" \
  --out reports/brian_pipeline_debug
```

Compare the NPZ sessions with a continuous recording from another project in the same report:

```bash
python tools/debug_bci_pipeline.py \
  --input "dataset/motor_intent/calib_brian_*.npz" \
  --reference-csv "path/to/other_project_recording.csv" \
  --reference-columns "Fp1,Fp2,Fz,Cz,FC1,FC2,C3,C4" \
  --reference-time-column session_time --reference-unit uv \
  --out reports/brian_pipeline_debug
```

The CSV sample rate is inferred from `session_time`; pass `--reference-fs 250` if its timestamps are unsuitable. Set `--reference-time-unit ms` when timestamps are milliseconds, and set `--reference-unit mv` or `v` when the EEG columns are not already in µV. For raw device counts or another documented unit, pass the manufacturer's multiplier with `--reference-scale-to-uv`; this overrides `--reference-unit`, and the applied multiplier is recorded in report details. Do not guess that multiplier from one row. The tool applies the same gross amplitude/flat/jump limits to non-overlapping windows, draws every montage separately, overlays only identically named nodes (for these montages: C3, Cz and C4), and reports Welch band-power comparisons for those common nodes. It warns when common-node 1–40 Hz RMS is extremely low because ratios against a near-zero denominator can be misleading. It never silently treats Fp1 as FC3 or another unlike position. A reference `PASS` only means the CSV passes gross raw-signal checks; it does not prove MI usefulness or accuracy.

The command writes one HTML and one JSON report and prints `FIRST FAILURE`. Downstream layers are skipped after an upstream raw failure unless `--force-downstream` is deliberately supplied; forced downstream results are diagnostic only and never override the upstream failure. The older specialized scripts remain reusable implementation modules, but normal troubleshooting should begin with this unified command.

Generate a detailed offline HTML/JSON usefulness report for several recordings (including partial sessions):

```bash
python tools/inspect_training_recordings.py \
  --input "dataset/motor_intent/calib_brian_*.npz" \
  --out reports/training_data_brian.html --open
```

The report shows each file's trial count versus 119, missing universal labels, class balance, retained/dropped trials, rejection reasons, montage suitability and representative eight-channel traces. A per-channel table attributes amplitude, flat and sample-jump failures and reports median/p95/max values, making it possible to distinguish one dead electrode from a recording-wide reference/scaling problem. `CANDIDATE` means it passes software quality/count checks only; it is not proof of MI or cross-session accuracy.

The same live dashboard can inspect saved training data interactively:

```bash
python example/performance_monitor.py \
  --training "dataset/motor_intent/calib_brian_*.npz"
```

Add a model to show intended-versus-predicted training previews:

```bash
python example/performance_monitor.py \
  --training dataset/motor_intent \
  --user brian --intents left_hand,right_hand,both_feet
```

Training mode shows label counts, retained/dropped trials, aggregate screening reasons, the selected 8-channel epoch and optional model predictions. It reads `channel_labels` from each NPZ instead of assuming the motor montage. Left/Right changes trial and K toggles kept-only view. Model metrics in this view use training recordings and are not held-session or held-subject validation.

## Public Python API

```python
import bci_sdk

mapping = bci_sdk.MotorIntentMap({
    "rock": bci_sdk.MotorIntent.LEFT_INDEX,
    "paper": bci_sdk.MotorIntent.RIGHT_INDEX,
    "scissors": bci_sdk.MotorIntent.RIGHT_PINKY,
})

bci = bci_sdk.LiveBCI(
    buttons,
    source="intent-mi",
    user="anson",
    intent_map=mapping,  # COM8 + ThinkPulse are current defaults
    on_select=lambda button: button.click(),
).start()

while running:
    bci.poll()
```

## One-command applications

One finger is supported only when calibration includes idle/no-movement trials:

```text
right_index -> fire
idle        -> no action (internal class)
```

`data_record.py --universal <subject>` includes 15 idle trials by default. Increase `--universal-idle` when a larger no-action sample is needed.

## Difficulty levels

```text
1 intent    BASIC          (requires idle)
2 intents   PILOT
3 intents   EXPERIMENTAL
4–5         ADVANCED
6–10        RESEARCH_ONLY
```

Always validate using separate-session grouped cross-validation. Do not present an accepted mapping as reliable solely because the SDK can represent it.
