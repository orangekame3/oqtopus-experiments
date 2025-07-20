#!/usr/bin/env python3
"""
Experiment result wrapper for usage.py compatibility
"""

from typing import Any


class ExperimentResult:
    """
    Wrapper for experiment results that provides analyze() method
    """

    def __init__(
        self,
        raw_results: dict[str, Any],
        experiment_instance: Any,
        experiment_type: str = "generic",
        **kwargs,
    ):
        """
        Initialize experiment result

        Args:
            raw_results: Raw measurement results
            experiment_instance: The experiment instance that generated this result
            experiment_type: Type of experiment (rabi, t1, ramsey, etc.)
            **kwargs: Additional parameters for analysis
        """
        self.raw_results = raw_results
        self.experiment = experiment_instance
        self.experiment_type = experiment_type
        self.analysis_params = kwargs
        self._analyzed_results = None

    def analyze(self, save: bool = True, **kwargs) -> dict[str, Any]:
        """
        Analyze the experiment results

        Args:
            save: Whether to save results
            **kwargs: Additional parameters passed to experiment's analyze method

        Returns:
            Analysis results dictionary
        """
        if self._analyzed_results is None:
            # Perform analysis using the experiment's analyze method
            analysis_params = {**self.analysis_params, **kwargs}
            self._analyzed_results = self.experiment.analyze(
                self.raw_results, **analysis_params
            )

        # Save results if requested
        if save:
            try:
                self.experiment.save_experiment_data(
                    self._analyzed_results,
                    metadata={
                        "experiment_type": self.experiment_type,
                        **self.analysis_params,
                    },
                )
            except Exception as e:
                print(f"Warning: Could not save results: {e}")

        return self._analyzed_results


    @property
    def results(self) -> dict[str, Any]:
        """Get analyzed results (lazy evaluation)"""
        if self._analyzed_results is None:
            self.analyze(save=False)
        return self._analyzed_results

    def __repr__(self) -> str:
        return f"ExperimentResult(type={self.experiment_type}, analyzed={self._analyzed_results is not None})"
