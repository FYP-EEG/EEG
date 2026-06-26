# 🧠 MindControl — Universal EEG Intent-to-Action Library

> A plug-and-play Python library that lets developers define custom "brain buttons" — mental actions mapped to EEG signals — so end users can control any application with their thoughts.

---

## 📌 Table of Contents

1. [Project Plan](#project-plan) 
1. [Vision & Overview](#vision--overview)
2. [How It Works](#how-it-works)
3. [Architecture](#architecture)
4. [Core Concepts](#core-concepts)
5. [Tech Stack](#tech-stack)
6. [Project Structure](#project-structure)
7. [Installation](#installation)
8. [Quick Start](#quick-start)
9. [Detailed Usage Examples](#detailed-usage-examples)
10. [EEG Signal Pipeline](#eeg-signal-pipeline)
11. [Supported EEG Devices](#supported-eeg-devices)
12. [Training & Calibration](#training--calibration)
13. [API Reference](#api-reference)
14. [Configuration](#configuration)
15. [Contributing](#contributing)
16. [Roadmap](#roadmap)
17. [FAQ](#faq)
18. [License](#license)

---

## 📅 Project Plan
See the full project Gantt chart and phase breakdown → [ROADMAP.md](./ROADMAP.md)

---

## 🌟 Vision & Overview

**MindControl** is an open-source Python library that bridges the gap between EEG (Electroencephalography) hardware and application logic. It provides a dead-simple API for developers to register "mental buttons" — named actions that users can trigger purely by **thinking** about them.

### The Problem

Every EEG-powered application today requires developers to:
- Understand neuroscience and signal processing
- Write custom ML pipelines from scratch
- Handle device-specific protocols
- Build calibration workflows manually

### The Solution

MindControl abstracts **all of that** into a single import:

```python
from mindcontrol import BrainInterface

brain = BrainInterface()
brain.add_button("rock")
brain.add_button("paper")
brain.add_button("scissors")

brain.calibrate()  # Guided training session

@brain.on("rock")
def on_rock():
    print("User chose ROCK! 🪨")

@brain.on("paper")
def on_paper():
    print("User chose PAPER! 📄")

@brain.on("scissors")
def on_scissors():
    print("User chose SCISSORS! ✂️")

brain.listen()
```

**That's it.** No signal processing. No ML code. No device drivers. Just buttons and callbacks.

---

## 🔬 How It Works

### High-Level Flow

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  EEG Device  │───▶│  Raw Signal  │───▶│  Feature     │───▶│  Classifier  │
│  (Hardware)  │    │  Acquisition │    │  Extraction  │    │  (ML Model)  │
└──────────────┘    └──────────────┘    └──────────────┘    └──────┬───────┘
                                                                    │
                    ┌──────────────┐    ┌──────────────┐           │
                    │  Developer's │◀───│  Action      │◀──────────┘
                    │  Callback    │    │  Dispatcher   │
                    └──────────────┘    └──────────────┘
```

### Step-by-Step

1. **Developer registers buttons** — Named mental actions (e.g., `"fire"`, `"jump"`, `"select"`)
2. **User calibrates** — A guided session where the user thinks about each action while the library records EEG patterns
3. **Model trains** — The library automatically trains a classifier to distinguish between the registered mental states
4. **Real-time inference** — During gameplay/usage, the library continuously reads EEG data, classifies the user's mental state, and fires the corresponding callback
5. **Developer receives events** — Simple event-driven callbacks, just like keyboard/mouse events

### The Neuroscience (Simplified)

The library leverages several well-established EEG paradigms:

| Paradigm | How It Works | Best For |
|----------|-------------|----------|
| **Motor Imagery (MI)** | User imagines moving left hand, right hand, feet, tongue | 2-5 distinct actions |
| **SSVEP** | User focuses on flickering visual stimuli at different frequencies | Menu selection with visual UI |
| **P300** | User focuses on a target among flashing options | Selection from many options |
| **Mental State** | User performs distinct mental tasks (math, singing, relaxing) | Mode switching |

MindControl **automatically selects** the best paradigm based on the number of buttons and configuration, or the developer can choose manually.

---

## 🏗️ Architecture

```
mindcontrol/
│
├── core/
│   ├── brain_interface.py      # Main API class — BrainInterface
│   ├── button.py               # Button registration & management
│   ├── event_dispatcher.py     # Event loop & callback management
│   └── session.py              # Session lifecycle management
│
├── acquisition/
│   ├── base_device.py          # Abstract base class for EEG devices
│   ├── muse_device.py          # Muse headband driver
│   ├── openbci_device.py       # OpenBCI driver
│   ├── emotiv_device.py        # Emotiv EPOC/Insight driver
│   ├── neurosity_device.py     # Neurosity Crown driver
│   ├── simulated_device.py     # Simulated device for testing/development
│   └── lsl_device.py           # Lab Streaming Layer (universal)
│
├── processing/
│   ├── filters.py              # Bandpass, notch, artifact rejection
│   ├── features.py             # Feature extraction (PSD, CSP, time-domain)
│   ├── artifacts.py            # Eye blink, jaw clench, motion detection
│   └── preprocessor.py         # Full preprocessing pipeline
│
├── classification/
│   ├── base_classifier.py      # Abstract classifier interface
│   ├── motor_imagery.py        # MI-based classifier (CSP + LDA/SVM)
│   ├── ssvep_classifier.py     # SSVEP frequency detection
│   ├── p300_classifier.py      # P300 oddball classifier
│   ├── mental_state.py         # Mental state classifier
│   ├── hybrid_classifier.py    # Combines multiple paradigms
│   └── transfer_learning.py    # Pre-trained models for faster calibration
│
├── calibration/
│   ├── calibration_manager.py  # Orchestrates calibration sessions
│   ├── guided_session.py       # Step-by-step user guidance
│   ├── adaptive_trainer.py     # Online/incremental learning
│   └── profiles.py             # Save/load user calibration profiles
│
├── paradigms/
│   ├── motor_imagery.py        # MI paradigm configuration
│   ├── ssvep.py                # SSVEP paradigm (generates visual stimuli)
│   ├── p300.py                 # P300 paradigm (generates oddball sequence)
│   └── auto_select.py          # Automatic paradigm selection
│
├── ui/
│   ├── calibration_ui.py       # Visual calibration interface
│   ├── feedback_overlay.py     # Real-time confidence overlay
│   └── debug_visualizer.py     # Live EEG signal viewer
│
├── utils/
│   ├── config.py               # Configuration management
│   ├── logging.py              # Structured logging
│   ├── metrics.py              # Accuracy, ITR calculation
│   └── data_recorder.py        # Record sessions for analysis
│
├── presets/
│   ├── gaming.py               # Gaming-optimized presets
│   ├── accessibility.py        # Accessibility presets (high accuracy)
│   └── research.py             # Research presets (raw data access)
│
└── __init__.py                 # Public API exports
```

---

## 🧩 Core Concepts

### Buttons

A **Button** is the fundamental unit of MindControl. It represents a named mental action that a user can "think" to trigger.

```python
# Buttons are just named actions
brain.add_button("fire")
brain.add_button("reload")
brain.add_button("grenade")

# Buttons with metadata
brain.add_button("fire", 
    description="Squeeze trigger / imagine right hand grip",
    icon="🔥",
    cooldown=0.5,          # Min seconds between triggers
    confidence_threshold=0.8  # Min confidence to trigger (0.0-1.0)
)
```

### Button Groups

**Button Groups** allow you to organize buttons into logical sets. Only one group is active at a time, improving classification accuracy.

```python
# Define groups for different game states
brain.add_group("combat", buttons=["fire", "reload", "grenade", "melee"])
brain.add_group("menu", buttons=["up", "down", "select", "back"])
brain.add_group("inventory", buttons=["use", "drop", "combine", "examine"])

# Switch active group
brain.set_active_group("combat")

# Context-aware switching
@brain.on("open_menu")
def open_menu():
    brain.set_active_group("menu")
```

### Paradigms

A **Paradigm** defines the underlying EEG technique used for classification.

```python
from mindcontrol.paradigms import MotorImagery, SSVEP, P300

# Auto-select (default — library chooses best paradigm)
brain = BrainInterface()  # Auto mode

# Explicit paradigm
brain = BrainInterface(paradigm=MotorImagery(classes=4))
brain = BrainInterface(paradigm=SSVEP(frequencies=[7, 10, 12, 15]))
brain = BrainInterface(paradigm=P300())
```

### Profiles

**Profiles** store a user's calibration data so they don't need to recalibrate every session.

```python
# Save after calibration
brain.calibrate()
brain.save_profile("user_john.mc")

# Load on next session
brain.load_profile("user_john.mc")
brain.listen()  # No calibration needed
```

---

## 🛠️ Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| **Language** | Python 3.10+ | Ecosystem, ML libraries, accessibility |
| **Signal Processing** | MNE-Python, SciPy | Industry-standard EEG processing |
| **Feature Extraction** | NumPy, MNE-Features | Fast numerical computation |
| **Classification** | scikit-learn, PyTorch (optional) | Lightweight default + deep learning option |
| **Real-time Streaming** | pylsl (Lab Streaming Layer) | Universal EEG device protocol |
| **Device Communication** | muselsl, brainflow | Multi-device support |
| **UI (Calibration)** | PsychoPy / PyQt6 | Visual stimulus presentation |
| **Async Runtime** | asyncio | Non-blocking real-time processing |
| **Data Storage** | HDF5 / JSON | Calibration profiles & session data |
| **Testing** | pytest, pytest-asyncio | Robust test suite |

---

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- A supported EEG device (or use simulated mode for development)

### Install from PyPI (planned)

```bash
pip install mindcontrol-eeg
```

### Install from source (current)

```bash
git clone https://github.com/yourusername/mindcontrol.git
cd mindcontrol
pip install -e ".[dev]"
```

### Install with specific device support

```bash
# Muse headband support
pip install mindcontrol-eeg[muse]

# OpenBCI support
pip install mindcontrol-eeg[openbci]

# All devices
pip install mindcontrol-eeg[all-devices]

# Development (includes simulated device, tests, docs)
pip install mindcontrol-eeg[dev]
```

### Verify Installation

```bash
python -c "import mindcontrol; print(mindcontrol.__version__)"
# Output: 0.1.0

# Run with simulated device (no hardware needed)
python -m mindcontrol.demo
```

---

## 🚀 Quick Start

### Example 1: Rock Paper Scissors (Simplest Possible)

```python
from mindcontrol import BrainInterface

# Initialize (auto-detects connected EEG device)
brain = BrainInterface()

# Register your "buttons"
brain.add_button("rock")
brain.add_button("paper")
brain.add_button("scissors")

# Calibrate — guides user through thinking about each action
# Takes ~3-5 minutes for 3 buttons
brain.calibrate()

# Register callbacks
@brain.on("rock")
def rock():
    print("🪨 ROCK selected!")

@brain.on("paper")
def paper():
    print("📄 PAPER selected!")

@brain.on("scissors")
def scissors():
    print("✂️ SCISSORS selected!")

# Optional: callback for any button
@brain.on_any
def any_selection(button_name, confidence):
    print(f"Detected: {button_name} (confidence: {confidence:.1%})")

# Start listening (blocking)
brain.listen()
```

### Example 2: Development Mode (No EEG Hardware)

```python
from mindcontrol import BrainInterface
from mindcontrol.acquisition import SimulatedDevice

# Use simulated device — keyboard keys simulate "thinking"
brain = BrainInterface(device=SimulatedDevice())

brain.add_button("rock")      # Press '1' to simulate
brain.add_button("paper")     # Press '2' to simulate  
brain.add_button("scissors")  # Press '3' to simulate

# Skip calibration in simulated mode
brain.calibrate(skip=True)

@brain.on("rock")
def rock():
    print("🪨 ROCK!")

brain.listen()
```

---

## 📚 Detailed Usage Examples

### Example 3: FPS Game Controller

```python
from mindcontrol import BrainInterface

brain = BrainInterface(
    mode="gaming",           # Optimized for low latency
    sample_rate=256,         # Hz
    inference_interval=0.25  # Classify every 250ms
)

# Combat buttons
brain.add_button("fire",     cooldown=0.3, confidence_threshold=0.85)
brain.add_button("reload",   cooldown=2.0, confidence_threshold=0.80)
brain.add_button("grenade",  cooldown=5.0, confidence_threshold=0.90)  # High threshold = fewer accidental grenades
brain.add_button("melee",    cooldown=1.0, confidence_threshold=0.80)
brain.add_button("idle",     is_rest=True)  # "thinking nothing" = idle state

# Calibrate with profile support
if brain.profile_exists("fps_player1"):
    brain.load_profile("fps_player1")
else:
    brain.calibrate(
        trials_per_button=20,     # 20 trials each
        trial_duration=4.0,       # 4 seconds per trial
        rest_duration=2.0,        # 2 seconds rest between trials
        show_feedback=True        # Real-time feedback during calibration
    )
    brain.save_profile("fps_player1")

# Print calibration accuracy
print(f"Calibration accuracy: {brain.accuracy:.1%}")
print(f"Information Transfer Rate: {brain.itr:.1f} bits/min")

# Game integration
import pyautogui  # or pydirectinput for games

@brain.on("fire")
def fire():
    pyautogui.click()  # Left mouse click

@brain.on("reload")
def reload():
    pyautogui.press('r')

@brain.on("grenade")
def grenade():
    pyautogui.press('g')

@brain.on("melee")
def melee():
    pyautogui.press('v')

# Start with stats display
brain.listen(show_stats=True)
```

### Example 4: Menu Navigation System

```python
from mindcontrol import BrainInterface
from mindcontrol.paradigms import SSVEP

# SSVEP is ideal for menu selection — user looks at flickering buttons
brain = BrainInterface(paradigm=SSVEP(
    frequencies=[7.5, 10.0, 12.0, 15.0],  # Hz — each button flickers at different freq
    harmonics=2                              # Also detect harmonic frequencies
))

brain.add_button("menu",      stimulus_freq=7.5)
brain.add_button("inventory", stimulus_freq=10.0)
brain.add_button("map",       stimulus_freq=12.0)
brain.add_button("settings",  stimulus_freq=15.0)

brain.calibrate(method="ssvep", duration=30)  # 30-second SSVEP calibration

@brain.on("menu")
def open_menu():
    print("📋 Opening Menu")
    brain.set_active_group("menu_items")

@brain.on("inventory")
def open_inventory():
    print("🎒 Opening Inventory")

@brain.on("map")
def open_map():
    print("🗺️ Opening Map")

brain.listen()
```

### Example 5: Async / Integration with Game Loops

```python
import asyncio
from mindcontrol import BrainInterface

brain = BrainInterface()
brain.add_button("attack")
brain.add_button("defend")
brain.add_button("special")

brain.load_profile("rpg_player")

async def game_loop():
    """Your existing game loop."""
    while True:
        # Poll for brain events (non-blocking)
        event = brain.poll()
        
        if event:
            print(f"Action: {event.button} | Confidence: {event.confidence:.1%}")
            
            if event.button == "attack":
                await perform_attack()
            elif event.button == "defend":
                await perform_defend()
            elif event.button == "special":
                await perform_special()
        
        # Rest of game logic
        await update_game_state()
        await render_frame()
        
        await asyncio.sleep(1/60)  # 60 FPS

async def main():
    await brain.start_async()  # Start EEG processing in background
    await game_loop()

asyncio.run(main())
```

### Example 6: Accessibility Application

```python
from mindcontrol import BrainInterface

brain = BrainInterface(
    mode="accessibility",  # Prioritizes accuracy over speed
    confirmation_mode=True  # Requires sustained thought to trigger (prevents accidental triggers)
)

# Communication board
brain.add_button("yes",     confidence_threshold=0.70)
brain.add_button("no",      confidence_threshold=0.70)
brain.add_button("help",    confidence_threshold=0.60)  # Lower threshold for emergency
brain.add_button("neutral", is_rest=True)

brain.calibrate(
    trials_per_button=30,     # More trials for higher accuracy
    trial_duration=5.0,
    adaptive=True             # Continues until accuracy target is met
)

@brain.on("yes")
def yes():
    speak("Yes")              # Text-to-speech

@brain.on("no")
def no():
    speak("No")

@brain.on("help")
def help():
    speak("I need help!")
    send_alert_to_caregiver()

brain.listen()
```

### Example 7: Pygame Integration

```python
import pygame
from mindcontrol import BrainInterface

pygame.init()
screen = pygame.display.set_mode((800, 600))

brain = BrainInterface(device="simulated")  # or real device
brain.add_button("left")
brain.add_button("right")
brain.add_button("jump")
brain.load_profile("platformer_player")

player_x, player_y = 400, 500
vel_y = 0

# Register callbacks that modify game state
@brain.on("left")
def move_left():
    global player_x
    player_x -= 5

@brain.on("right")
def move_right():
    global player_x
    player_x += 5

@brain.on("jump")
def jump():
    global vel_y
    if player_y >= 500:  # On ground
        vel_y = -15

# Start brain listener in background thread
brain.listen_in_background()

# Standard Pygame loop
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Physics
    vel_y += 0.8  # Gravity
    player_y += vel_y
    if player_y > 500:
        player_y = 500
        vel_y = 0
    
    # Render
    screen.fill((0, 0, 0))
    pygame.draw.rect(screen, (0, 255, 0), (player_x, player_y, 40, 40))
    
    # Show brain status
    status = brain.get_status()
    font = pygame.font.Font(None, 24)
    text = font.render(
        f"Last: {status.last_action} | Confidence: {status.confidence:.0%} | Signal: {'✅' if status.signal_quality > 0.7 else '⚠️'}",
        True, (255, 255, 255)
    )
    screen.blit(text, (10, 10))
    
    pygame.display.flip()
    clock.tick(60)

brain.stop()
pygame.quit()
```

---

## 🔌 EEG Signal Pipeline

### Detailed Processing Chain

```
RAW EEG (μV) ──────────────────────────────────────────────────────────────
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. ACQUISITION (250-1000 Hz sampling)                                   │
│    • Connect to device via Bluetooth/USB/WiFi/LSL                       │
│    • Buffer incoming samples into windows                               │
│    • Synchronize multi-channel data                                     │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. PREPROCESSING                                                        │
│    • Bandpass filter: 1-45 Hz (remove DC drift & high-freq noise)       │
│    • Notch filter: 50/60 Hz (remove power line interference)            │
│    • Re-referencing: Common Average Reference (CAR)                     │
│    • Bad channel interpolation                                          │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 3. ARTIFACT REJECTION                                                   │
│    • Eye blink detection (frontal channels, >100μV threshold)           │
│    • Jaw clench detection (temporal channels, EMG pattern)              │
│    • Motion artifact detection (all channels, sudden amplitude change)  │
│    • Reject or repair contaminated windows                              │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 4. FEATURE EXTRACTION                                                   │
│    ┌─────────────────┬─────────────────┬──────────────────────────┐     │
│    │ Frequency Domain │  Spatial Domain  │   Time Domain            │     │
│    │ • Band powers    │  • CSP filters   │   • Hjorth parameters    │     │
│    │   (α, β, θ, γ)  │  • Laplacian     │   • Zero-crossing rate   │     │
│    │ • PSD estimates  │  • xDAWN         │   • Statistical moments  │     │
│    │ • Spectral edge  │                  │   • Waveform length      │     │
│    └─────────────────┴─────────────────┴──────────────────────────┘     │
│    • Feature vector: [f1, f2, ..., fn]                                  │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 5. CLASSIFICATION                                                       │
│    • Input: Feature vector                                              │
│    • Model: LDA / SVM / Random Forest / CNN (configurable)             │
│    • Output: {button_name: probability} for each registered button     │
│    • Confidence thresholding                                            │
│    • Temporal smoothing (majority vote over N consecutive predictions)  │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 6. ACTION DISPATCH                                                      │
│    • Check confidence > threshold                                       │
│    • Check cooldown timer                                               │
│    • Fire registered callback                                           │
│    • Emit event to event queue (for polling mode)                       │
│    • Log action + confidence + latency                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Timing Budget (Target: <300ms end-to-end)

| Stage | Time Budget | Notes |
|-------|------------|-------|
| Acquisition window | 500-1000ms | Sliding window with 250ms step |
| Preprocessing | ~10ms | Optimized with SciPy |
| Artifact check | ~5ms | Simple threshold checks |
| Feature extraction | ~15ms | Vectorized NumPy operations |
| Classification | ~5ms | Lightweight sklearn models |
| Dispatch | <1ms | Direct callback |
| **Total latency** | **~300ms** | From thought to action |

---

## 🎧 Supported EEG Devices

| Device | Channels | Price Range | Connection | Status |
|--------|----------|-------------|------------|--------|
| **Muse 2 / Muse S** | 4 (+1 aux) | $250-$380 | Bluetooth | ✅ Primary Support |
| **OpenBCI Cyton** | 8-16 | $500-$1,000 | Bluetooth/Dongle | ✅ Primary Support |
| **OpenBCI Ganglion** | 4 | $250 | Bluetooth | ✅ Supported |
| **Emotiv EPOC X** | 14 | $849 | USB Dongle | 🔄 In Progress |
| **Emotiv Insight** | 5 | $499 | Bluetooth | 🔄 In Progress |
| **Neurosity Crown** | 8 | $899 | WiFi | 📋 Planned |
| **BrainBit** | 4 | $499 | Bluetooth | 📋 Planned |
| **Any LSL-compatible** | Varies | Varies | LSL Protocol | ✅ Supported |
| **Simulated Device** | 8 (virtual) | Free | Keyboard input | ✅ Built-in |

### Adding Custom Device Support

```python
from mindcontrol.acquisition import BaseDevice

class MyCustomDevice(BaseDevice):
    def connect(self):
        # Your device connection logic
        pass
    
    def start_stream(self):
        # Start data streaming
        pass
    
    def read_sample(self) -> np.ndarray:
        # Return shape: (n_channels,)
        pass
    
    def get_channel_names(self) -> list[str]:
        return ["Fp1", "Fp2", "C3", "C4", "P3", "P4", "O1", "O2"]
    
    def get_sample_rate(self) -> int:
        return 256

# Use it
brain = BrainInterface(device=MyCustomDevice())
```

---

## 🎯 Training & Calibration

### How Calibration Works

```
┌────────────────────────────────────────────────────────────────┐
│                    CALIBRATION SESSION                         │
│                                                                │
│  Trial 1/20:  "Think about: ROCK 🪨"                         │
│  ████████████████████████░░░░░░  4.0s                         │
│  [Recording EEG...]                                            │
│                                                                │
│  Rest period...                                                │
│  ██████████░░░░░░░░░░░░░░░░░░░  2.0s                         │
│                                                                │
│  Trial 2/20:  "Think about: SCISSORS ✂️"                      │
│  ██████████████░░░░░░░░░░░░░░░  4.0s                         │
│  [Recording EEG...]                                            │
│                                                                │
│  Progress: ████░░░░░░░░░░░░░░░  Trial 2/60 (3.3%)           │
│  Current accuracy estimate: Calculating...                     │
│  Signal quality: ████████░░ Good (82%)                        │
└────────────────────────────────────────────────────────────────┘
```

### Calibration Tips for Users

The library provides built-in guidance, but here are the mental strategies that work best:

| Button Action | Suggested Mental Strategy |
|---------------|--------------------------|
| Action 1 | Imagine squeezing your **left hand** |
| Action 2 | Imagine squeezing your **right hand** |
| Action 3 | Imagine wiggling your **toes/feet** |
| Action 4 | Imagine moving your **tongue** |
| Rest/Idle | Relax, think of nothing specific |

### Adaptive Calibration

```python
# Minimum viable calibration (fastest)
brain.calibrate(
    trials_per_button=10,
    trial_duration=3.0,
    target_accuracy=0.70    # Stop when 70% accuracy reached
)

# High-accuracy calibration
brain.calibrate(
    trials_per_button=40,
    trial_duration=5.0,
    target_accuracy=0.90,
    adaptive=True           # Add more trials if needed
)

# Incremental calibration (improve existing profile)
brain.load_profile("player1")
brain.calibrate(mode="incremental", additional_trials=10)
brain.save_profile("player1")  # Updated profile
```

---

## 📖 API Reference

### `BrainInterface`

The main class. Everything starts here.

```python
class BrainInterface:
    def __init__(
        self,
        device: str | BaseDevice = "auto",        # EEG device or "auto"
        paradigm: str | BaseParadigm = "auto",     # Classification paradigm
        mode: str = "balanced",                     # "gaming" | "accessibility" | "balanced" | "research"
        sample_rate: int = 256,                     # Target sample rate (Hz)
        inference_interval: float = 0.5,            # Seconds between classifications
        window_size: float = 1.0,                   # EEG window size in seconds
        buffer_size: float = 5.0,                   # Ring buffer size in seconds
    ): ...
    
    # --- Button Management ---
    def add_button(
        self,
        name: str,
        description: str = "",
        icon: str = "",
        cooldown: float = 0.5,
        confidence_threshold: float = 0.75,
        is_rest: bool = False,
        stimulus_freq: float | None = None,         # For SSVEP paradigm
    ) -> Button: ...
    
    def remove_button(self, name: str) -> None: ...
    def list_buttons(self) -> list[Button]: ...
    
    # --- Button Groups ---
    def add_group(self, name: str, buttons: list[str]) -> None: ...
    def set_active_group(self, name: str) -> None: ...
    
    # --- Calibration ---
    def calibrate(
        self,
        trials_per_button: int = 20,
        trial_duration: float = 4.0,
        rest_duration: float = 2.0,
        target_accuracy: float = 0.75,
        adaptive: bool = False,
        show_feedback: bool = True,
        mode: str = "full",                         # "full" | "incremental" | "transfer"
        skip: bool = False,                          # Skip (for simulated device)
    ) -> CalibrationResult: ...
    
    # --- Profiles ---
    def save_profile(self, path: str) -> None: ...
    def load_profile(self, path: str) -> None: ...
    def profile_exists(self, path: str) -> bool: ...
    
    # --- Event Handling ---
    def on(self, button_name: str) -> Callable: ...             # Decorator
    def on_any(self, callback: Callable) -> None: ...           # Any button
    def on_error(self, callback: Callable) -> None: ...         # Error handler
    def on_signal_quality(self, callback: Callable) -> None: ... # Signal quality updates
    
    # --- Listening ---
    def listen(self, show_stats: bool = False) -> None: ...           # Blocking
    def listen_in_background(self) -> threading.Thread: ...           # Non-blocking
    async def start_async(self) -> None: ...                          # Async
    def poll(self) -> BrainEvent | None: ...                          # Non-blocking poll
    def stop(self) -> None: ...                                       # Stop listening
    
    # --- Status ---
    def get_status(self) -> BrainStatus: ...
    
    @property
    def accuracy(self) -> float: ...                # Last calibration accuracy
    @property
    def itr(self) -> float: ...                     # Information Transfer Rate
    @property
    def is_connected(self) -> bool: ...
    @property
    def signal_quality(self) -> float: ...          # 0.0 - 1.0
```

### `BrainEvent`

Returned by `brain.poll()` and passed to callbacks.

```python
@dataclass
class BrainEvent:
    button: str              # Button name that was triggered
    confidence: float        # 0.0 - 1.0
    timestamp: float         # Unix timestamp
    all_probabilities: dict  # {"rock": 0.85, "paper": 0.10, "scissors": 0.05}
    latency_ms: float        # Processing latency in milliseconds
```

### `BrainStatus`

```python
@dataclass
class BrainStatus:
    is_connected: bool
    device_name: str
    signal_quality: float       # 0.0 - 1.0
    battery_level: float | None # 0.0 - 1.0 (if supported)
    last_action: str | None
    confidence: float
    actions_per_minute: float
    active_group: str | None
    uptime_seconds: float
```

---

## ⚙️ Configuration

### Configuration File (`mindcontrol.yaml`)

```yaml
# mindcontrol.yaml — place in project root or ~/.mindcontrol/config.yaml

device:
  type: "auto"                  # auto | muse | openbci | emotiv | simulated | lsl
  serial_port: null             # For wired devices
  bluetooth_address: null       # Specific BT address

processing:
  sample_rate: 256
  bandpass: [1, 45]             # Hz
  notch_filter: 60              # 50 (Europe) or 60 (US) Hz
  window_size: 1.0              # seconds
  overlap: 0.75                 # 75% overlap between windows
  artifact_rejection: true
  artifact_threshold: 100       # μV

classification:
  model: "lda"                  # lda | svm | random_forest | cnn
  paradigm: "auto"              # auto | motor_imagery | ssvep | p300
  confidence_threshold: 0.75
  temporal_smoothing: 3         # Majority vote over N predictions
  inference_interval: 0.5       # seconds

calibration:
  trials_per_button: 20
  trial_duration: 4.0
  rest_duration: 2.0
  target_accuracy: 0.75
  profiles_directory: "~/.mindcontrol/profiles/"

logging:
  level: "INFO"                 # DEBUG | INFO | WARNING | ERROR
  file: null                    # Log to file
  record_sessions: false        # Save raw EEG data

ui:
  show_calibration_ui: true
  show_feedback_overlay: false
  theme: "dark"
```

### Environment Variables

```bash
export MINDCONTROL_DEVICE=muse
export MINDCONTROL_LOG_LEVEL=DEBUG
export MINDCONTROL_PROFILES_DIR=/path/to/profiles
```

---

## 🤝 Contributing

We welcome contributions! MindControl is a complex project spanning neuroscience, signal processing, machine learning, and software engineering.

### How to Contribute

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/my-feature`
3. **Commit** your changes: `git commit -m 'Add my feature'`
4. **Push** to the branch: `git push origin feature/my-feature`
5. **Open** a Pull Request

### Development Setup

```bash
git clone https://github.com/yourusername/mindcontrol.git
cd mindcontrol

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install in development mode with all extras
pip install -e ".[dev,all-devices]"

# Run tests
pytest tests/ -v

# Run tests with simulated EEG
pytest tests/ -v --simulated

# Run linter
ruff check .

# Run type checker
mypy mindcontrol/

# Build docs
cd docs && make html
```

### Areas We Need Help With

| Area | Skills Needed | Priority |
|------|--------------|----------|
| Device drivers | Bluetooth, hardware protocols | 🔴 High |
| Signal processing | DSP, MNE-Python | 🔴 High |
| ML models | scikit-learn, PyTorch, BCI | 🔴 High |
| Transfer learning | Deep learning, domain adaptation | 🟡 Medium |
| UI/Calibration | PyQt/PsychoPy, UX design | 🟡 Medium |
| Documentation | Technical writing | 🟡 Medium |
| Game integrations | Pygame, Unity bridge, Godot | 🟢 Nice to have |
| Mobile support | Android/iOS EEG apps | 🟢 Nice to have |

---

## 🗺️ Roadmap

### Phase 1: Foundation (v0.1.0) — Current
- [x] Project architecture & API design
- [ ] Core `BrainInterface` class
- [ ] Simulated device for development
- [ ] Basic Motor Imagery pipeline (CSP + LDA)
- [ ] Simple calibration workflow
- [ ] Event system (callbacks + polling)
- [ ] Profile save/load
- [ ] Basic tests

### Phase 2: Real Hardware (v0.2.0)
- [ ] Muse device support (via muselsl/brainflow)
- [ ] OpenBCI device support
- [ ] LSL universal protocol support
- [ ] Signal quality indicators
- [ ] Artifact rejection pipeline
- [ ] Calibration UI (PsychoPy)

### Phase 3: Advanced Classification (v0.3.0)
- [ ] SSVEP paradigm support
- [ ] P300 paradigm support
- [ ] Hybrid paradigm (auto-select)
- [ ] Transfer learning (pre-trained models)
- [ ] Adaptive/online learning
- [ ] Button groups & context switching

### Phase 4: Polish & Ecosystem (v0.4.0)
- [ ] Pygame integration module
- [ ] Unity bridge (Python ↔ C#)
- [ ] Godot integration
- [ ] Web dashboard for monitoring
- [ ] Community model marketplace
- [ ] Performance benchmarks

### Phase 5: Production (v1.0.0)
- [ ] 90%+ accuracy for 4-class MI
- [ ] <300ms end-to-end latency
- [ ] Support for 5+ EEG devices
- [ ] Comprehensive documentation
- [ ] Accessibility certification
- [ ] PyPI stable release

---

## ❓ FAQ

### General

**Q: Do I need an EEG device to develop with MindControl?**
A: No! Use `SimulatedDevice()` for development. It maps keyboard keys to buttons so you can build and test your application without hardware.

**Q: How many buttons can I register?**
A: Practically, Motor Imagery supports 2-5 buttons reliably. SSVEP can support 6-10+. P300 can support 20+ (like a keyboard). More buttons = longer calibration + lower accuracy per button.

**Q: How accurate is it?**
A: With proper calibration:
- 2 buttons: 85-95% accuracy
- 3 buttons: 75-90% accuracy
- 4 buttons: 65-85% accuracy
- Results vary significantly between users

**Q: Can this read my thoughts / mind?**
A: **Absolutely not.** EEG measures general brain activity patterns, not specific thoughts. The library detects *which mental task* you're performing from a small set you trained it on. It cannot read words, images, or private thoughts.

### Technical

**Q: What's the latency?**
A: Target is <300ms from mental state change to callback firing. Actual latency depends on window size, device sampling rate, and processing power.

**Q: Can I use this with Unity / Unreal / Godot?**
A: Direct integration modules are planned. For now, you can bridge via WebSocket, OSC, or named pipes.

**Q: Does it work with consumer EEG headsets?**
A: Yes, primarily designed for consumer devices like Muse and OpenBCI. Research-grade devices also work via LSL.

**Q: Can multiple users use it simultaneously?**
A: Create separate `BrainInterface` instances with different devices. Each user needs their own EEG headset and calibration profile.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [MNE-Python](https://mne.tools/) — EEG processing foundation
- [BrainFlow](https://brainflow.org/) — Universal EEG device library
- [MOABB](https://moabb.neurotechx.com/) — BCI benchmarking
- [Lab Streaming Layer](https://labstreaminglayer.org/) — Real-time streaming protocol
- [OpenBCI](https://openbci.com/) — Open-source EEG hardware
- The BCI research community

---

<div align="center">

**🧠 MindControl — Think it. Do it.**

[Documentation](https://mindcontrol.readthedocs.io) · [Discord](https://discord.gg/mindcontrol) · [Report Bug](https://github.com/yourusername/mindcontrol/issues) · [Request Feature](https://github.com/yourusername/mindcontrol/issues)

</div>
