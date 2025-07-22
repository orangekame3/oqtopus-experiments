#!/usr/bin/env python3
"""
Core module - Base classes and core functionality
"""

from .base_experiment import BaseExperiment
from .data_manager import ExperimentDataManager

__all__ = ["BaseExperiment", "ExperimentDataManager"]
