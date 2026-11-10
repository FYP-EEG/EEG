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

Runtime models and in-app personal calibration are stored by user and unordered task set:

```text
profiles/<user>/motor_intent/<task-set>/
    calib_*.npz                 # in-app personal calibration only
    intent_model.joblib
    intent_model_info.json
```

Standalone `EEG/data_record.py` sessions are generalized training data, not profiles:

```text
dataset/motor_intent/calib_<subject>_<timestamp>.npz
```

Each generalized NPZ still contains subject, session, montage, sampling-rate and label metadata so grouped evaluation remains possible.

Changing action names without changing the motor-intent set does not require retraining. Adding/removing a motor intention creates a different task set and requires calibration.

## Universal reusable body-intent recording

The recommended first recording collects a shared pool containing all ten fingers,
left hand, right hand, both feet, and idle. It uses local CC0 hand/foot outline
assets; the cued finger, whole hand, or both feet change colour while the real body
remains still.

```bash
python EEG/data_record.py --intent-mi --subject anson \
  --trials-per-intent 8 --universal-idle 15 --cue 4 --rest 2 \
  --board-id 0 --serial-port COM4
```

This records 13 active intentions x 8 trials plus 15 idle trials (119 total) into:

```text
dataset/motor_intent/calib_anson_*.npz
```

The universal recording is a reusable labelled data pool, not one forced 14-class
runtime model. Each application trains a smaller task-specific model by filtering the
intentions it needs. App-specific calibration blocks are combined with the reusable
pool automatically.

Running the same command with `--dry-run` writes only under `dry_runs/` and prints a
warning not to train from synthetic data.

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
python example/rockpaperscissors.py --user anson --port COM4 \
  --calibration-source cyton
```

Use F2, then Enter. The tab stops a running BCI before opening the board and, after successful real training, can restart `intent-mi`. A dry run checks only timing/UI and deliberately saves no EEG model. One block trains a usable experimental model but cannot produce a trustworthy session-held-out score; record at least three separate blocks. During training, the loader combines the selected app-specific profile calibration with matching labels from the generalized `dataset/motor_intent/` pool. Legacy `_universal_13` profile recordings remain readable.

Default in-app block sizes provide screening headroom:

```text
RPS:          36 action + 12 idle (12 per action)
Subway:       36 action + 12 idle (12 per action)
Realistic UI: 60 action + 12 idle (12 per action)
```

The external `EEG/data_record.py --intent-mi` workflow remains available for lab scripting and reproducible command-line sessions.

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

Record three separate sessions:

```bash
python EEG/data_record.py --file rockpaperscissors --intent-mi \
  --subject anson --trials 90 --idle-trials 15 --cue 4 --rest 2 \
  --board-id 0 --serial-port COM4
```

Train:

```bash
python ML/train_intent_mi.py anson \
  --intents left_index,right_index,right_pinky
```

Play:

```bash
python example/rockpaperscissors.py \
  --bci intent-mi --user anson --port COM4
```

Override actions while retaining the same trained task set:

```bash
python example/rockpaperscissors.py --bci intent-mi --user anson --port COM4 \
  --intent-map "rock=right_pinky,paper=left_index,scissors=right_index"
```

## Subway-style runner

Default:

```text
left_hand  -> move_left
right_hand -> move_right
both_feet  -> jump
```

Record:

```bash
python EEG/data_record.py --file subway_run --intent-mi \
  --subject anson --trials 90 --idle-trials 15 --cue 4 --rest 2 \
  --board-id 0 --serial-port COM4
```

Train:

```bash
python ML/train_intent_mi.py anson \
  --intents left_hand,right_hand,both_feet
```

Play:

```bash
python example/subway_run.py --bci intent-mi --user anson --port COM4
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

The SDK marks this ADVANCED and warns about the adjacent right-thumb/right-index pair.

```bash
python EEG/data_record.py --file realistic_ui --intent-mi \
  --subject anson --trials 150 --idle-trials 20 --cue 4 --rest 2 \
  --board-id 0 --serial-port COM4

python ML/train_intent_mi.py anson \
  --intents left_thumb,left_pinky,right_thumb,right_index,right_pinky

python example/realistic_ui.py --bci intent-mi --user anson --port COM4
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
    serial_port="COM4",
    intent_map=mapping,
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

`data_record.py --intent-mi` includes ten idle trials by default. Increase this for a reliable false-activation estimate.

## Difficulty levels

```text
1 intent    BASIC          (requires idle)
2 intents   PILOT
3 intents   EXPERIMENTAL
4–5         ADVANCED
6–10        RESEARCH_ONLY
```

Always validate using separate-session grouped cross-validation. Do not present an accepted mapping as reliable solely because the SDK can represent it.
