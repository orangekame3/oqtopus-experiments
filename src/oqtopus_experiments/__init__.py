#!/usr/bin/env python3
"""
OQTOPUS Experiments - Quantum computing experiment library
"""

# Core components
# Backend implementations
from .backends import LocalBackend, OqtopusBackend

# Circuit utilities
from .core import BaseExperiment, ExperimentDataManager

# Device information
from .devices import DeviceInfo

# Experiments
from .experiments import (
    CHSH,
    T1,
    CHSHPhaseScan,
    ParityOscillation,
    Rabi,
    Ramsey,
    T2Echo,
)

__version__ = "0.1.0"
__author__ = "orangekame3"
__all__ = [
    "BaseExperiment",
    "CHSH",
    "CHSHPhaseScan",
    "Rabi",
    "Ramsey",
    "T1",
    "T2Echo",
    "ParityOscillation",
    "LocalBackend",
    "OqtopusBackend",
    "DeviceInfo",
    "ExperimentDataManager",
]
