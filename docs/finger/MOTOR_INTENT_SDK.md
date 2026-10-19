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

Models are stored by user and unordered task set:

```text
profiles/<user>/motor_intent/<task-set>/
    calib_*.npz
    intent_model.joblib
    intent_model_info.json
```

Changing action names without changing the motor-intent set does not require retraining. Adding/removing a motor intention creates a different task set and requires calibration.

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

Keyboard fallback: LEFT/A, RIGHT/D, UP/W/SPACE.

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
