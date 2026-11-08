"""
bci_sdk -- a developer library for brain-controlled user interfaces.

    import bci_sdk

    session = bci_sdk.BCISession(source="sim")
    session.on_command(lambda cmd: print("brain said:", cmd))
    session.start()

The package exposes high-level sessions, UI widgets, classifiers, acquisition
backends, reusable motor-intent models, and calibration helpers through one
stable public namespace.

Edited by Anson
Date: 19/10/2026
Purpose: make the public namespace lazy. Importing one lightweight component
         such as bci_sdk.body_map must not eagerly load SciPy, pandas, sklearn,
         every classifier, and the hardware stack before a pygame window opens.
"""

import sys as _sys
from importlib import import_module as _import_module
from importlib.util import find_spec as _find_spec
from pathlib import Path as _Path

# Source-checkout compatibility. A clean installed wheel remains a separate
# packaging concern, but these paths preserve the project's established layout.
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT / "pygame_lib"), str(_ROOT / "EEG")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

__version__ = "0.12.0"
_HAS_PYGAME = _find_spec("pygame") is not None

_CORE_EXPORTS = [
    "BCISession", "Command", "UserProfile", "list_users",
    "BCIEngine", "HybridSSVEPClassifier", "MotorImageryClassifier",
    "TRCAClassifier", "HybridTRCAClassifier", "NO_ACTION",
    "TextBuffer", "montage", "datasets",
    "load_benchmark_mat", "load_benchmark_dir", "load_physionet_edf",
    "load_calibration_npz", "pick_channels", "resample_to", "crop_or_pad",
    "combine",
    "StreamBridge", "RingBuffer", "SyntheticBackend", "ReplayBackend",
    "BrainFlowBackend",
    "LocalMIModel", "ButtonMIModel", "MotorIntentModel", "MotorIntent",
    "MotorIntentMap", "FINGER_INTENTS", "BODY_INTENTS", "UNIVERSAL_INTENTS",
    "assess_intent_set", "parse_intent_map", "SelectionTree",
    "DirectButtonSelector", "IntentActionSelector", "RealtimeDecider",
    "ArtifactDetector", "build_runtime", "LEFT", "RIGHT",
    "LiveBCI", "ScanSelector", "SineProbe", "SineProbeModel",
    "IntentCalibrationPanel", "draw_body_intent_map", "GameCalibrationMenu",
    "BCIError", "HardwareError", "CalibrationError", "NotCalibratedError",
]
_UI_EXPORTS = [
    "BrainButton", "ButtonGroup", "Scanner", "TileButton",
    "StimulusEngine", "FlickerPlan", "open_display",
    "suggest_frequencies", "detect_refresh_rate",
]
__all__ = ["__version__", *_CORE_EXPORTS]
if _HAS_PYGAME:
    __all__ += _UI_EXPORTS

# Public name -> (module, attribute). Values are imported only on first access.
_LAZY_EXPORTS = {
    "BCISession": ("bci_sdk.session", "BCISession"),
    "Command": ("bci_sdk.session", "Command"),
    "UserProfile": ("bci_sdk.profiles", "UserProfile"),
    "list_users": ("bci_sdk.profiles", "list_users"),
    "BCIEngine": ("ML.bci_engine", "BCIEngine"),
    "HybridSSVEPClassifier": ("ML.hybrid_classifier", "HybridSSVEPClassifier"),
    "MotorImageryClassifier": ("ML.hybrid_classifier", "MotorImageryClassifier"),
    "NO_ACTION": ("ML.hybrid_classifier", "NO_ACTION"),
    "TRCAClassifier": ("ML.trca", "TRCAClassifier"),
    "HybridTRCAClassifier": ("ML.trca", "HybridTRCAClassifier"),
    "TextBuffer": ("ML.text_buffer", "TextBuffer"),
    "load_benchmark_mat": ("ML.datasets", "load_benchmark_mat"),
    "load_benchmark_dir": ("ML.datasets", "load_benchmark_dir"),
    "load_physionet_edf": ("ML.datasets", "load_physionet_edf"),
    "load_calibration_npz": ("ML.datasets", "load_calibration_npz"),
    "pick_channels": ("ML.datasets", "pick_channels"),
    "resample_to": ("ML.datasets", "resample_to"),
    "crop_or_pad": ("ML.datasets", "crop_or_pad"),
    "combine": ("ML.datasets", "combine"),
    "StreamBridge": ("stream_bridge", "StreamBridge"),
    "RingBuffer": ("stream_bridge", "RingBuffer"),
    "SyntheticBackend": ("stream_bridge", "SyntheticBackend"),
    "ReplayBackend": ("stream_bridge", "ReplayBackend"),
    "BrainFlowBackend": ("stream_bridge", "BrainFlowBackend"),
    "LocalMIModel": ("ML.local_model", "LocalMIModel"),
    "ButtonMIModel": ("ML.button_mi", "ButtonMIModel"),
    "MotorIntentModel": ("ML.motor_intent", "MotorIntentModel"),
    "MotorIntent": ("ML.motor_intent", "MotorIntent"),
    "MotorIntentMap": ("ML.motor_intent", "MotorIntentMap"),
    "FINGER_INTENTS": ("ML.motor_intent", "FINGER_INTENTS"),
    "BODY_INTENTS": ("ML.motor_intent", "BODY_INTENTS"),
    "UNIVERSAL_INTENTS": ("ML.motor_intent", "UNIVERSAL_INTENTS"),
    "assess_intent_set": ("ML.motor_intent", "assess_intent_set"),
    "parse_intent_map": ("ML.motor_intent", "parse_intent_map"),
    "SelectionTree": ("bci_sdk.selection", "SelectionTree"),
    "LEFT": ("bci_sdk.selection", "LEFT"),
    "RIGHT": ("bci_sdk.selection", "RIGHT"),
    "DirectButtonSelector": ("bci_sdk.direct", "DirectButtonSelector"),
    "IntentActionSelector": ("bci_sdk.direct", "IntentActionSelector"),
    "RealtimeDecider": ("bci_sdk.runtime", "RealtimeDecider"),
    "ArtifactDetector": ("bci_sdk.runtime", "ArtifactDetector"),
    "build_runtime": ("bci_sdk.runtime", "build_runtime"),
    "LiveBCI": ("bci_sdk.live", "LiveBCI"),
    "ScanSelector": ("bci_sdk.live", "ScanSelector"),
    "SineProbe": ("bci_sdk.live", "SineProbe"),
    "SineProbeModel": ("bci_sdk.live", "SineProbeModel"),
    "IntentCalibrationPanel": ("bci_sdk.intent_calibration", "IntentCalibrationPanel"),
    "draw_body_intent_map": ("bci_sdk.body_map", "draw_body_intent_map"),
    "GameCalibrationMenu": ("bci_sdk.start_menu", "GameCalibrationMenu"),
    "BCIError": ("bci_sdk.errors", "BCIError"),
    "HardwareError": ("bci_sdk.errors", "HardwareError"),
    "CalibrationError": ("bci_sdk.errors", "CalibrationError"),
    "NotCalibratedError": ("bci_sdk.errors", "NotCalibratedError"),
    "BrainButton": ("bci_sdk.widgets", "BrainButton"),
    "ButtonGroup": ("bci_sdk.widgets", "ButtonGroup"),
    "Scanner": ("bci_sdk.widgets", "Scanner"),
    "TileButton": ("TileButton", "TileButton"),
    "StimulusEngine": ("stimulus", "StimulusEngine"),
    "FlickerPlan": ("stimulus", "FlickerPlan"),
    "open_display": ("stimulus", "open_display"),
    "suggest_frequencies": ("stimulus", "suggest_frequencies"),
    "detect_refresh_rate": ("stimulus", "detect_refresh_rate"),
}
_LAZY_MODULES = {
    "montage": "ML.montage",
    "datasets": "ML.datasets",
}


def __getattr__(name):
    module_name = _LAZY_MODULES.get(name)
    if module_name is not None:
        value = _import_module(module_name)
        globals()[name] = value
        return value

    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name in _UI_EXPORTS and not _HAS_PYGAME:
        raise ImportError(
            f"bci_sdk.{name} requires pygame-ce. Install it with: "
            "python -m pip install pygame-ce")
    module_name, attribute = target
    value = getattr(_import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))


def has_ui():
    """True when pygame-ce/pygame can be imported by UI components."""
    return _HAS_PYGAME


def info():
    """Human-readable summary of the SDK and optional dependencies."""
    montage_module = _import_module("ML.montage")
    return {
        "version": __version__,
        "ui_available": _HAS_PYGAME,
        "hardware_available": _find_spec("brainflow") is not None,
        "montages": sorted(montage_module.MONTAGES),
    }
