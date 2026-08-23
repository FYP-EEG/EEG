"""
bci_sdk -- a developer library for brain-controlled user interfaces.

    import bci_sdk

    session = bci_sdk.BCISession(source="sim")
    session.on_command(lambda cmd: print("brain said:", cmd))
    session.start()

WHAT THIS PACKAGE IS
--------------------
The project's engine, acquisition and UI layers were developed as separate modules
(ML/, EEG/, pygame_lib/). This package is the *stable public surface* over them:
a third-party developer imports `bci_sdk` and never touches the internals, so we can
refactor underneath without breaking their code.

Three layers, use whichever you need:

  1. HIGH LEVEL   bci_sdk.BCISession
                  One object that wires backend -> stream bridge -> engine and
                  delivers decoded commands to your callbacks. ~5 lines to a
                  working brain-controlled app.

  2. MID LEVEL    bci_sdk.BrainButton, bci_sdk.ButtonGroup, bci_sdk.Scanner
                  Event-driven UI widgets. `button.on_brain_select(fn)` fires when
                  the user selects that button with their brain.

  3. LOW LEVEL    bci_sdk.BCIEngine, bci_sdk.StreamBridge, bci_sdk.HybridSSVEPClassifier
                  The components themselves, unchanged.

DESIGN NOTES FOR INTEGRATORS
----------------------------
* Everything is non-blocking. `session.poll()` is designed to be called once per
  render frame and returns immediately.
* The engine can answer "no command" (`NO_ACTION`). Do not treat silence as an error
  -- a user who is reading or looking away SHOULD produce no events.
* pygame is an optional dependency. Importing `bci_sdk` without it still gives you
  the engine, bridge and classifiers; only the UI widgets are withheld.
"""

import sys as _sys
from pathlib import Path as _Path

# The sibling packages are not installed on sys.path in a source checkout, so make
# them importable. A packaged install (pyproject.toml) maps them properly instead.
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT / "pygame_lib"), str(_ROOT / "EEG")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

__version__ = "0.9.0"

# ----------------------------------------------------------------- core (always)
from ML.bci_engine import BCIEngine                                    # noqa: E402
from ML.hybrid_classifier import (HybridSSVEPClassifier,               # noqa: E402
                                  MotorImageryClassifier, NO_ACTION)
from ML.trca import TRCAClassifier, HybridTRCAClassifier               # noqa: E402
from ML.text_buffer import TextBuffer                                  # noqa: E402
from ML import montage                                                 # noqa: E402
from ML import datasets                                                # noqa: E402
from ML.datasets import (load_benchmark_mat, load_benchmark_dir,       # noqa: E402
                         load_physionet_edf, load_calibration_npz,
                         pick_channels, resample_to, crop_or_pad, combine)

from stream_bridge import (StreamBridge, RingBuffer, SyntheticBackend, # noqa: E402
                           ReplayBackend, BrainFlowBackend)

from .session import BCISession, Command                               # noqa: E402
from .profiles import UserProfile, list_users                          # noqa: E402
from .errors import (BCIError, HardwareError, CalibrationError,        # noqa: E402
                     NotCalibratedError)

__all__ = [
    "__version__",
    # high level
    "BCISession", "Command", "UserProfile", "list_users",
    # engine / ML
    "BCIEngine", "HybridSSVEPClassifier", "MotorImageryClassifier",
    "TRCAClassifier", "HybridTRCAClassifier", "NO_ACTION",
    "TextBuffer", "montage", "datasets",
    # dataset loaders (public + your own)
    "load_benchmark_mat", "load_benchmark_dir", "load_physionet_edf",
    "load_calibration_npz", "pick_channels", "resample_to", "crop_or_pad",
    "combine",
    # acquisition
    "StreamBridge", "RingBuffer", "SyntheticBackend", "ReplayBackend",
    "BrainFlowBackend",
    # errors
    "BCIError", "HardwareError", "CalibrationError", "NotCalibratedError",
]

# ------------------------------------------------------------- UI (needs pygame)
try:
    import pygame as _pygame  # noqa: F401
    _HAS_PYGAME = True
except Exception:                                    # pragma: no cover
    _HAS_PYGAME = False

if _HAS_PYGAME:
    from .widgets import BrainButton, ButtonGroup, Scanner               # noqa: E402
    from stimulus import (StimulusEngine, FlickerPlan, open_display,     # noqa: E402
                          suggest_frequencies, detect_refresh_rate)
    from TileButton import TileButton                                    # noqa: E402
    __all__ += ["BrainButton", "ButtonGroup", "Scanner", "TileButton",
                "StimulusEngine", "FlickerPlan", "open_display",
                "suggest_frequencies", "detect_refresh_rate"]
else:                                                # pragma: no cover
    def _missing(name):
        def _raise(*a, **k):
            raise ImportError(
                f"bci_sdk.{name} requires pygame. Install it with: pip install pygame")
        return _raise
    for _n in ("BrainButton", "ButtonGroup", "Scanner", "TileButton",
               "StimulusEngine", "open_display"):
        globals()[_n] = _missing(_n)


def has_ui():
    """True if the UI widgets are available (pygame installed)."""
    return _HAS_PYGAME


def info():
    """Human-readable summary of the installed SDK and its optional deps."""
    try:
        import brainflow  # noqa: F401
        hw = True
    except Exception:
        hw = False
    return {
        "version": __version__,
        "ui_available": _HAS_PYGAME,
        "hardware_available": hw,
        "montages": sorted(montage.MONTAGES),
    }
