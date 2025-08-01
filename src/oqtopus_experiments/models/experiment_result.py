#!/usr/bin/env python3
"""
Experiment result wrapper for usage.py compatibility
"""

from typing import Any

import pandas as pd


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

    def analyze(
        self, plot: bool = True, save_data: bool = True, save_image: bool = True
    ) -> pd.DataFrame:
        """
        Analyze the experiment results

        Args:
            plot: Whether to generate plots
            save_data: Whether to save analysis results data
            save_image: Whether to save generated images

        Returns:
            DataFrame with analysis results
        """
        if self._analyzed_results is None:
            try:
                # Perform analysis using the experiment's analyze method
                raw_analysis = self.experiment.analyze(
                    self.raw_results,
                    plot=plot,
                    save_data=save_data,
                    save_image=save_image,
                )

                # Convert to DataFrame format
                self._analyzed_results = self._extract_dataframe(raw_analysis)

            except Exception as e:
                print(f"⚠️  Analysis failed: {e}")
                self._analyzed_results = pd.DataFrame()

        # Save results if requested (DataFrame is easily serializable)
        if save_data:
            try:
                if (
                    hasattr(self.experiment, "save_experiment_data")
                    and self._analyzed_results is not None
                    and hasattr(self._analyzed_results, "to_dict")
                ):
                    # DataFrame to dict conversion for clean JSON storage
                    saved_path = self.experiment.save_experiment_data(
                        self._analyzed_results.to_dict(orient="records"),
                        metadata={
                            "experiment_type": self.experiment_type,
                            **self.analysis_params,
                        },
                    )
                    print(f"💾 Results saved to: {saved_path}")
                else:
                    print("⚠️  Save method not available for this experiment")
            except Exception as e:
                print(f"⚠️  Warning: Could not save results: {e}")

        return self._analyzed_results

    def _extract_dataframe(self, analysis_result: Any) -> pd.DataFrame:
        """Extract DataFrame from analysis result of various types"""
        # If already a DataFrame, return as-is
        if isinstance(analysis_result, pd.DataFrame):
            return analysis_result

        # If it's a dict, look for DataFrames within it
        if isinstance(analysis_result, dict):
            for _, value in analysis_result.items():
                if isinstance(value, pd.DataFrame):
                    return value  # Return the first DataFrame found

        # If no DataFrame found, return empty DataFrame
        return pd.DataFrame()

    @property
    def results(self) -> pd.DataFrame:
        """Get analyzed results (lazy evaluation)"""
        if self._analyzed_results is None:
            self._analyzed_results = self.analyze(save_data=False)
        return self._analyzed_results

    def __repr__(self) -> str:
        return f"ExperimentResult(type={self.experiment_type}, analyzed={self._analyzed_results is not None})"
