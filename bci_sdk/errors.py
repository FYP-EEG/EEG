"""
Typed exceptions for bci_sdk.

Integrators need to distinguish "the headset fell off" from "you passed a bad
argument" from "this model needs calibration first". Bare RuntimeError does not
let them do that, so the SDK raises these instead.
"""


class BCIError(Exception):
    """Base class for every error raised by bci_sdk."""


class HardwareError(BCIError):
    """Acquisition hardware is missing, disconnected, or misconfigured.

    Typically recoverable by falling back to a synthetic or replay backend.
    """


class CalibrationError(BCIError):
    """A calibration recording or profile is missing, malformed, or unusable."""


class NotCalibratedError(CalibrationError):
    """A subject-specific model was used before being fitted.

    Raised by TRCA-backed paths when no calibration data has been supplied.
    """
