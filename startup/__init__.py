"""Mandatory headset startup checks used before real EEG recording or use."""

from .check_headband import evaluate_hardware_stages, run_headband_check

__all__ = ["evaluate_hardware_stages", "run_headband_check"]
