"""Mandatory headset startup checks used before real EEG recording or use."""

from .check_headband import (
    evaluate_hardware_stages,
    run_headband_check,
    startup_failure_hints,
    wait_for_headband_ready,
)

__all__ = [
    "evaluate_hardware_stages",
    "run_headband_check",
    "startup_failure_hints",
    "wait_for_headband_ready",
]
