#!/usr/bin/env python3
"""
OQTOPUS Experiments - Quantum computing experiment library
"""

# Core components
# Backend implementations
from .backends import LocalBackend, OqtopusBackend

# Circuit utilities
from .circuit import create_chsh_circuit
from .core import BaseExperiment, ExperimentDataManager

# Device information
from .devices import DeviceInfo

# Experiments
# Legacy imports
from .experiments import (
    CHSH,
    T1,
    CHSHExperiment,
    ParityOscillation,
    ParityOscillationExperiment,
    Rabi,
    RabiExperiment,
    Ramsey,
    RamseyExperiment,
    T1Experiment,
    T2Echo,
    T2EchoExperiment,
)

__version__ = "0.1.0"
__author__ = "quantumlib"
__all__ = [
    "BaseExperiment",
    "CHSH",
    "Rabi",
    "Ramsey",
    "T1",
    "T2Echo",
    "ParityOscillation",
    "LocalBackend",
    "OqtopusBackend",
    "DeviceInfo",
    "create_chsh_circuit",
    "ExperimentDataManager",
    # Legacy names
    "CHSHExperiment",
    "RabiExperiment",
    "RamseyExperiment",
    "T1Experiment",
    "T2EchoExperiment",
    "ParityOscillationExperiment",
]
