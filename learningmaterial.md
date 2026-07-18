# MindControl — Developer Guide (No Code, Just Direction)

---

## Your Learning Path First

```
Before touching any file, you need to understand these 5 domains:

Domain 1 → Python fundamentals (you probably know this)
Domain 2 → Signal Processing (how EEG data looks and gets cleaned)
Domain 3 → Machine Learning (how to classify brain states)
Domain 4 → Software Architecture (how to structure a library)
Domain 5 → EEG/BCI Fundamentals (what brain signals actually are)

You don't need to master all 5 before starting.
Learn as you build each step.
```

---

## Domain 1 — Python You Must Know

### Topics to study
```
- Classes and OOP (__init__, self, inheritance)
- Decorators (@something above a function)
- Threading and asyncio (running things in background)
- Abstract base classes (ABC)
- Context managers (with statements)
- Type hints (def func(x: int) -> str)
- dataclasses
- Exception handling patterns
```

### Resources
```
📘 Books:
   - "Fluent Python" by Luciano Ramalho
     → Chapters 1, 7 (decorators), 17 (concurrency)
   - "Python Cookbook" by David Beazley
     → Chapter 8 (classes), Chapter 12 (concurrency)

🌐 Free Online:
   - realpython.com/python-decorators
   - realpython.com/python-concurrency
   - realpython.com/python-classes
   - docs.python.org/3/library/threading.html
   - docs.python.org/3/library/asyncio.html

🎥 YouTube:
   - "Python OOP Tutorial" by Corey Schafer (6-part series)
   - "Python Threading" by Corey Schafer
   - "Python Decorators" by Corey Schafer
```

### What to focus on
```
CRITICAL (learn before writing any code):
   ✅ Classes, inheritance, ABC
   ✅ Decorators
   ✅ Threading basics

LEARN WHEN YOU GET THERE:
   ⏳ asyncio (Step 5 of building)
   ⏳ dataclasses (Step 2)
   ⏳ Type hints (throughout)
```

---

## Domain 2 — Signal Processing

### What you need to understand
```
EEG is just a time series of voltage readings.
Like audio, but from your brain.

You need to understand:
   - What sampling rate means (256 Hz = 256 readings per second)
   - What frequency bands mean (alpha, beta, theta, gamma waves)
   - Why we filter signals (noise removal)
   - What a bandpass filter does
   - What FFT is (convert time → frequency domain)
   - What an epoch is (a time window of EEG)
   - What artifacts are (eye blinks, jaw clenches polluting data)
```

### Resources
```
📘 Books:
   - "Analyzing Neural Time Series Data" by Mike X Cohen
     → THE best book for EEG signal processing
     → Read chapters 1-12 before touching processing code
   - "Digital Signal Processing" by Proakis & Manolakis
     → More math-heavy, read if you want deep understanding
   
🌐 Free Online:
   - mikexcohen.com (free videos by the book author above)
   - mne.tools/stable/documentation.html (MNE library docs)
   - scipy.org/doc (scipy.signal module specifically)
   - numpy.org/doc (numpy.fft module)

🎥 YouTube:
   - "Signal Processing" playlist by Mike X Cohen
     → Search "Mike X Cohen EEG" on YouTube
     → Watch the entire playlist, it's free
   - "Fourier Transform" by 3Blue1Brown
     → Essential to understand frequency analysis

📄 Papers:
   - "A review of classification algorithms for EEG-based
     brain-computer interfaces" (Lotte et al., 2007)
     → Free on Google Scholar
```

### Concepts to understand in order
```
Week 1:
   → What is a signal? What is sampling rate?
   → Time domain vs frequency domain
   → What is Fourier Transform (FFT)?

Week 2:
   → What are EEG frequency bands?
      Delta  (0.5-4 Hz)  → deep sleep
      Theta  (4-8 Hz)    → drowsiness
      Alpha  (8-13 Hz)   → relaxed, eyes closed
      Beta   (13-30 Hz)  → active thinking, focus
      Gamma  (30+ Hz)    → high cognitive load
   → Why do we bandpass filter? (1-45 Hz)
   → What is a notch filter? (remove 50/60 Hz powerline noise)

Week 3:
   → What is an epoch? (slicing continuous EEG into windows)
   → What are artifacts? (noise from eyes, muscles, movement)
   → How to detect and remove artifacts
```

---

## Domain 3 — Machine Learning for EEG

### What you need to understand
```
After cleaning the EEG signal, you need to:
   1. Extract features (turn raw signal into numbers a model understands)
   2. Train a classifier (teach model to recognize each mental state)
   3. Predict in real time (classify new incoming EEG)
```

### Resources
```
📘 Books:
   - "Brain-Computer Interfaces: Principles and Practice"
     by Wolpaw & Wolpaw
     → Chapters 1, 3, 5, 9 are most relevant
   - "Hands-On Machine Learning" by Aurélien Géron
     → Chapters 1-6 for scikit-learn basics
     → Chapter 11+ if you go deep learning route

🌐 Free Online:
   - scikit-learn.org/stable/user_guide.html
     → Linear Discriminant Analysis (LDA) section
     → SVM section
     → Cross-validation section
   - mne.tools/stable/auto_tutorials (decoding/classification section)
   - moabb.neurotechx.com/docs (EEG benchmarking, lots of examples)

📄 Papers (all free on Google Scholar or arxiv):
   - "EEGNet: A Compact CNN for EEG-Based BCIs" (Lawhern et al., 2018)
     → The standard deep learning baseline for EEG
   - "Common Spatial Patterns" (CSP) — search this term
     → The standard feature extraction for Motor Imagery
   - "A review of feature extraction for EEG-based BCI" (Zander, 2011)

🎥 YouTube:
   - "StatQuest" by Josh Starmer
     → Watch: LDA, SVM, Cross-validation, Neural Networks playlists
   - "Neurotechx EEG ML tutorials" on YouTube
     → Search "NeuroTechX EEG machine learning"
```

### ML concepts to learn in order
```
Week 1: Scikit-learn basics
   → What is a classifier?
   → Train/test split
   → Cross-validation
   → Accuracy, F1 score, confusion matrix

Week 2: EEG-specific features
   → Power Spectral Density (PSD)
   → Common Spatial Patterns (CSP)
     ← THIS is the core feature extractor for Motor Imagery EEG
   → What a feature vector is

Week 3: Classifiers
   → Linear Discriminant Analysis (LDA) ← simplest, works great for EEG
   → Support Vector Machine (SVM)
   → When to use which

Week 4 (optional, later):
   → EEGNet (CNN for EEG)
   → Transfer learning
   → Domain adaptation
```

---

## Domain 4 — Software Architecture (Library Design)

### What you need to understand
```
You're building a LIBRARY not an app.
Other developers will import your code.
This changes how you design everything.

Key concepts:
   - How to design a clean public API
   - Abstract Base Classes (for device abstraction)
   - Strategy pattern (swap classifiers/devices)
   - Observer/Event pattern (callbacks)
   - Dependency injection
   - How PyPI packages are structured
```

### Resources
```
📘 Books:
   - "Architecture Patterns with Python" by Percival & Gregory
     → Free online: cosmicpython.com
     → Read chapters 1-3 first
   - "Clean Code" by Robert Martin
     → Language-agnostic but essential thinking

🌐 Free Online:
   - realpython.com/python-application-layouts
     → How to structure a Python project
   - packaging.python.org/tutorials/packaging-projects
     → How PyPI packages work
   - refactoring.guru/design-patterns/python
     → Strategy pattern, Observer pattern — both used in this project

🎥 YouTube:
   - "Python Design Patterns" by ArjanCodes on YouTube
     → Watch: Strategy, Observer, Abstract Factory episodes
   - "How to write a Python library" by ArjanCodes
```

### Study these design patterns specifically
```
1. Strategy Pattern
   → Used for: swapping devices (Muse vs OpenBCI vs Simulated)
   → Used for: swapping classifiers (LDA vs SVM vs EEGNet)

2. Observer / Event Pattern
   → Used for: @brain.on("rock") decorator system
   → Used for: callback registration

3. Abstract Base Class
   → Used for: BaseDevice (all devices implement same interface)
   → Used for: BaseClassifier

4. Factory Pattern
   → Used for: auto-detecting which device is connected
```

---

## Domain 5 — EEG & BCI Fundamentals

### What you need to understand
```
What IS an EEG signal?
What are the different BCI paradigms?
How does a BCI system actually work end to end?
```

### Resources
```
📘 Books:
   - "Brain-Computer Interfaces: An Introduction" by Rao
     → Best intro book, very readable
     → Read chapters 1-4 before writing any EEG code
   - "EEG Signal Processing" by Sanei & Chambers
     → More technical reference

🌐 Free Online:
   - openbci.com/community (tutorials section)
   - bnci-horizon-2020.eu (free research papers)
   - neurotechx.com/resources (community resources)
   - bcidataset.org (datasets with documentation)

📄 Papers:
   - "Review of the BCI Competition Datasets" (Tangermann et al.)
   - "Motor Imagery BCI: A Review" — search on Google Scholar
   - "SSVEP-Based BCI: A Review" — search on Google Scholar

🎥 YouTube:
   - "What is a BCI?" by NeuroTechX
   - "EEG for Beginners" by OpenBCI official channel
   - "Motor Imagery BCI tutorial" — search on YouTube
```

### Paradigms to understand
```
1. Motor Imagery (MI)
   → User IMAGINES moving left/right hand, feet, tongue
   → No actual movement needed
   → This is your PRIMARY paradigm for gaming buttons
   → Study: CSP algorithm specifically for MI

2. SSVEP (Steady State Visual Evoked Potentials)
   → User LOOKS at flickering buttons at different frequencies
   → Each button flickers at different Hz (7Hz, 10Hz, 12Hz...)
   → Brain responds at same frequency → detected by FFT
   → Good for menu selection

3. P300
   → User focuses on target while random items flash
   → Brain produces P300 wave ~300ms after seeing target
   → Good for spelling/selection systems

4. Mental State
   → Math vs singing vs relaxing → different EEG patterns
   → Less reliable but no external stimulus needed
```

---

## File Structure (What Each File Does)

```
D:\EEG\
│
├── mindcontrol/                    ← your library package
│   │
│   ├── __init__.py                 ← exposes public API
│   │                                  Study: Python package __init__
│   │
│   ├── core.py                     ← BrainInterface main class
│   │                                  Study: Python classes, decorators
│   │
│   ├── button.py                   ← Button dataclass
│   │                                  Study: Python dataclasses
│   │
│   ├── events.py                   ← BrainEvent, event dispatcher
│   │                                  Study: Observer pattern
│   │
│   ├── acquisition/                ← EEG device drivers
│   │   ├── __init__.py
│   │   ├── base_device.py          ← Abstract base class
│   │   │                              Study: ABC, Strategy pattern
│   │   ├── simulated.py            ← Keyboard fake device
│   │   │                              Study: keyboard library, threading
│   │   ├── muse.py                 ← Muse headband
│   │   │                              Study: muselsl library, pylsl
│   │   ├── openbci.py              ← OpenBCI board
│   │   │                              Study: brainflow library
│   │   └── lsl_device.py           ← Universal LSL stream
│   │                                  Study: pylsl documentation
│   │
│   ├── processing/                 ← Signal processing pipeline
│   │   ├── __init__.py
│   │   ├── filters.py              ← Bandpass, notch filters
│   │   │                              Study: scipy.signal, MNE filtering
│   │   ├── epochs.py               ← Cut continuous EEG into windows
│   │   │                              Study: MNE epochs documentation
│   │   ├── artifacts.py            ← Eye blink/jaw clench removal
│   │   │                              Study: MNE artifact rejection
│   │   └── features.py             ← Extract CSP, PSD features
│   │                                  Study: MNE CSP, scipy.signal PSD
│   │
│   ├── classification/             ← ML models
│   │   ├── __init__.py
│   │   ├── base_classifier.py      ← Abstract classifier
│   │   │                              Study: ABC pattern
│   │   ├── lda_classifier.py       ← Linear Discriminant Analysis
│   │   │                              Study: sklearn LDA docs
│   │   ├── svm_classifier.py       ← Support Vector Machine
│   │   │                              Study: sklearn SVM docs
│   │   ├── eegnet.py               ← CNN for EEG (later phase)
│   │   │                              Study: EEGNet paper (Lawhern 2018)
│   │   └── pipeline.py             ← Chains features → classifier
│   │                                  Study: sklearn Pipeline docs
│   │
│   ├── calibration/                ← Training session management
│   │   ├── __init__.py
│   │   ├── calibrator.py           ← Orchestrates calibration
│   │   │                              Study: MNE epochs, sklearn fit()
│   │   ├── session.py              ← Single trial recording
│   │   │                              Study: threading, timing
│   │   └── profiles.py             ← Save/load trained models
│   │                                  Study: pickle, joblib, json
│   │
│   └── utils/                      ← Shared helpers
│       ├── __init__.py
│       ├── logger.py               ← Logging setup
│       │                              Study: Python logging module
│       ├── config.py               ← Load yaml config files
│       │                              Study: PyYAML library
│       └── metrics.py              ← Accuracy, ITR calculation
│                                      Study: sklearn metrics docs
│
├── tests/                          ← All your tests
│   ├── __init__.py
│   ├── test_button.py              ← Test Button class
│   ├── test_events.py              ← Test event system
│   ├── test_filters.py             ← Test signal filters
│   ├── test_classifier.py          ← Test ML pipeline
│   └── test_calibration.py         ← Test calibration flow
│                                      Study: pytest documentation
│
├── examples/                       ← Demo scripts
│   ├── rock_paper_scissors.py      ← Simplest demo
│   ├── fps_controller.py           ← Gaming demo
│   └── menu_navigation.py          ← SSVEP menu demo
│
├── data/                           ← Downloaded public datasets
│   ├── bci_competition_iv/
│   ├── physionet_eegmmidb/
│   └── README.md                   ← How to download datasets
│
├── notebooks/                      ← Jupyter exploration notebooks
│   ├── 01_explore_raw_eeg.ipynb    ← Look at raw EEG data
│   ├── 02_preprocessing.ipynb      ← Test filters visually
│   ├── 03_feature_extraction.ipynb ← CSP, PSD exploration
│   ├── 04_train_classifier.ipynb   ← Train and evaluate models
│   └── 05_realtime_test.ipynb      ← Simulate real-time pipeline
│                                      Study: Jupyter, matplotlib
│
├── requirements.txt                ← pip dependencies
├── setup.py                        ← Package installation config
├── README.md                       ← Library documentation
├── ROADMAP.md                      ← Your Gantt chart
└── mindcontrol.yaml                ← Default config file
```

---

## What To Study For Each File

```
FILE                    STUDY THESE TOPICS
────────────────────────────────────────────────────────────────

button.py           →   Python dataclasses
                        Python type hints

core.py             →   Python classes and OOP
                        Decorator pattern
                        Python threading basics

events.py           →   Observer design pattern
                        Python callable objects
                        Queue module

base_device.py      →   Python ABC (abstract base class)
                        Strategy design pattern

simulated.py        →   keyboard library docs
                        Python threading
                        time module

muse.py             →   muselsl GitHub readme
                        pylsl documentation
                        Bluetooth basics

openbci.py          →   brainflow.org documentation
                        BrainFlow Python examples

filters.py          →   scipy.signal.butter (bandpass)
                        scipy.signal.iirnotch (notch)
                        mne.filter documentation
                        "Analyzing Neural Time Series" Ch 12-14

epochs.py           →   MNE epochs tutorial
                        numpy array slicing
                        Sliding window concept

artifacts.py        →   MNE artifact rejection tutorial
                        ICA decomposition basics
                        Threshold-based rejection

features.py         →   MNE CSP documentation
                        scipy.signal.welch (PSD)
                        numpy operations

lda_classifier.py   →   sklearn LinearDiscriminantAnalysis docs
                        sklearn Pipeline docs
                        Cross-validation in sklearn

eegnet.py           →   EEGNet paper (Lawhern et al. 2018)
                        PyTorch basics
                        CNN fundamentals

calibrator.py       →   MNE epochs creation
                        sklearn model fitting
                        Python threading for timed trials

profiles.py         →   joblib.dump / joblib.load
                        JSON for metadata
                        pathlib module

logger.py           →   Python logging module docs

config.py           →   PyYAML documentation
                        Python pathlib

test_*.py files     →   pytest documentation
                        pytest fixtures
                        Test-driven development basics
```

---

## Build Order With What To Study First

```
PHASE A — Learn & Scaffold (Week 1-2)
─────────────────────────────────────
Study first:
   → Python OOP (Corey Schafer OOP series)
   → Python decorators (realpython.com)
   → Design patterns: Observer, Strategy, ABC (ArjanCodes)

Then create:
   → Folder structure
   → button.py
   → events.py
   → core.py (skeleton only)
   → __init__.py files

────────────────────────────────────────────────

PHASE B — Fake Device + Working Demo (Week 3-4)
────────────────────────────────────────────────
Study first:
   → Python threading (realpython.com/intro-to-python-threading)
   → keyboard library (github.com/boppreh/keyboard)

Then create:
   → acquisition/base_device.py
   → acquisition/simulated.py
   → Wire it into core.py
   → examples/rock_paper_scissors.py → RUNS ✅

────────────────────────────────────────────────

PHASE C — Signal Processing (Week 5-7)
───────────────────────────────────────
Study first:
   → "Analyzing Neural Time Series" Ch 1-14 (Mike X Cohen)
   → MNE getting started tutorial (mne.tools)
   → scipy.signal documentation
   → Watch ALL Mike X Cohen YouTube videos

Then create:
   → processing/filters.py
   → processing/epochs.py
   → processing/artifacts.py
   → notebooks/01_explore_raw_eeg.ipynb
   → notebooks/02_preprocessing.ipynb

────────────────────────────────────────────────

PHASE D — Feature Extraction (Week 8-9)
────────────────────────────────────────
Study first:
   → CSP algorithm — read the original paper
   → MNE CSP tutorial (mne.tools/stable/auto_examples)
   → scipy.signal.welch documentation

Then create:
   → processing/features.py
   → notebooks/03_feature_extraction.ipynb

────────────────────────────────────────────────

PHASE E — Classifier (Week 10-12)
──────────────────────────────────
Study first:
   → StatQuest: LDA video on YouTube
   → sklearn LDA documentation
   → sklearn Pipeline documentation
   → sklearn cross_val_score documentation

Then create:
   → classification/base_classifier.py
   → classification/lda_classifier.py
   → classification/pipeline.py
   → notebooks/04_train_classifier.ipynb

────────────────────────────────────────────────

PHASE F — Real EEG Device (Week 13-15)
───────────────────────────────────────
Study first:
   → brainflow.org getting started
   → muselsl GitHub (github.com/alexandrebarachant/muse-lsl)
   → pylsl documentation

Then create:
   → acquisition/muse.py OR acquisition/openbci.py
   → acquisition/lsl_device.py

────────────────────────────────────────────────

PHASE G — Calibration System (Week 16-18)
──────────────────────────────────────────
Study first:
   → MNE epochs documentation
   → sklearn model persistence (joblib)
   → Python threading for timed events

Then create:
   → calibration/session.py
   → calibration/calibrator.py
   → calibration/profiles.py

────────────────────────────────────────────────

PHASE H — Tests & Polish (Week 19-20)
──────────────────────────────────────
Study first:
   → pytest getting started (docs.pytest.org)
   → pytest fixtures

Then create:
   → All test files
   → requirements.txt finalized
   → setup.py
```

---

## Absolute Starting Point Right Now

```
TODAY:
   1. Watch Corey Schafer "Python OOP" playlist (6 videos, ~2 hours)
      → youtube.com/c/Coreyms
   
   2. Read realpython.com/primer-on-python-decorators
      → Takes ~45 minutes
   
   3. Watch ArjanCodes "Observer Pattern in Python"
      → ~20 minutes

TOMORROW:
   1. Create the folder structure (empty files only)
   2. Open "Analyzing Neural Time Series" Ch 1 and start reading
   
THIS WEEK:
   1. Finish OOP + decorators + threading study
   2. Write button.py from scratch yourself
   3. Write events.py from scratch yourself
   
NEXT WEEK:
   1. Write simulated device
   2. Wire up core.py
   3. Run rock_paper_scissors.py successfully
```

---

## One Rule

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   Study the concept FIRST.                          │
│   Then write the file.                              │
│   Never write a file you don't understand yet.      │
│                                                     │
│   It's better to spend 3 days studying CSP          │
│   and write features.py correctly once              │
│   than to copy code and be stuck debugging          │
│   something you don't understand.                   │
│                                                     │
└─────────────────────────────────────────────────────┘
```