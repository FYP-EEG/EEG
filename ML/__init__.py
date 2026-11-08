"""
Author: Brian
Created on: 03/08/2026
Purpose: moduling the ML package to include the hybrid classifier

Edited by Anson
Date: 19/10/2026
Purpose: defer scientific classifier imports until they are actually requested.
         Lightweight recorder modules such as ML.motor_intent must not import
         SciPy/sklearn merely to read the motor-intent vocabulary.
"""

from importlib import import_module

__all__ = [
    "HybridSSVEPClassifier", "TRCAClassifier", "HybridTRCAClassifier",
]

_LAZY_EXPORTS = {
    "HybridSSVEPClassifier": ("ML.hybrid_classifier", "HybridSSVEPClassifier"),
    "TRCAClassifier": ("ML.trca", "TRCAClassifier"),
    "HybridTRCAClassifier": ("ML.trca", "HybridTRCAClassifier"),
}


def __getattr__(name):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
