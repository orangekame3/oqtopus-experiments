#!/usr/bin/env python3
"""
Experiments module - Concrete experiment implementations
"""

from .chsh import CHSH

# Legacy imports for backward compatibility
from .chsh import CHSH as CHSHExperiment
from .chsh_phase_scan import CHSHPhaseScan
from .hadamard_test import HadamardTest
from .parity_oscillation import ParityOscillation
from .parity_oscillation import ParityOscillation as ParityOscillationExperiment
from .rabi import Rabi
from .rabi import Rabi as RabiExperiment
from .ramsey import Ramsey
from .ramsey import Ramsey as RamseyExperiment
from .t1 import T1
from .t1 import T1 as T1Experiment
from .t2_echo import T2Echo
from .t2_echo import T2Echo as T2EchoExperiment

__all__ = [
    "CHSH",
    "CHSHPhaseScan",
    "HadamardTest",
    "ParityOscillation",
    "Rabi",
    "Ramsey",
    "T1",
    "T2Echo",
    # Legacy names
    "CHSHExperiment",
    "ParityOscillationExperiment",
    "RabiExperiment",
    "RamseyExperiment",
    "T1Experiment",
    "T2EchoExperiment",
]
