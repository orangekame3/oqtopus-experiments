#!/usr/bin/env python3
"""
T2 Echo (Hahn Echo) Experiment Class
"""

from typing import Any

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from scipy.optimize import curve_fit

from ..core.base_experiment import BaseExperiment
from ..models.t2_echo_models import (
    T2EchoAnalysisResult,
    T2EchoCircuitParams,
    T2EchoData,
    T2EchoFittingResult,
    T2EchoParameters,
)


class T2Echo(BaseExperiment):
    """T2 Echo (Hahn Echo) experiment for T2 measurement"""

    def __init__(
        self,
        experiment_name: str | None = None,
        physical_qubit: int | None = None,
        delay_points: int = 20,
        max_delay: float = 30000.0,
    ):
        """Initialize T2 Echo experiment with explicit parameters"""
        # Track if physical_qubit was explicitly specified
        self._physical_qubit_specified = physical_qubit is not None
        actual_physical_qubit = physical_qubit if physical_qubit is not None else 0

        self.params = T2EchoParameters(
            experiment_name=experiment_name,
            physical_qubit=actual_physical_qubit,
            delay_points=delay_points,
            max_delay=max_delay,
        )
        super().__init__(self.params.experiment_name or "t2_echo_experiment")

        self.physical_qubit = self.params.physical_qubit
        self.delay_points = self.params.delay_points
        self.max_delay = self.params.max_delay

    def analyze(
        self,
        results: dict[str, list[dict[str, Any]]],
        plot: bool = True,
        save_data: bool = True,
        save_image: bool = True,
    ) -> pd.DataFrame:
        """Analyze T2 Echo results with simplified single-result processing"""

        if not results:
            return pd.DataFrame()

        # Flatten all results into single list (no device separation)
        all_results = []
        for device_data in results.values():
            all_results.extend(device_data)

        if not all_results:
            return pd.DataFrame()

        # Fit the data
        fitting_result = self._fit_t2_echo_data(all_results)
        if not fitting_result:
            return pd.DataFrame()

        # Get device name from results
        device_name = "unknown"
        if all_results:
            # Get device name from first result's backend field
            device_name = all_results[0].get("backend", "unknown")

        # Create DataFrame
        df = self._create_dataframe(fitting_result, device_name)

        # Create T2EchoData for new analysis system
        t2_echo_data = T2EchoData(
            delay_times=fitting_result.delay_times,
            probabilities=fitting_result.probabilities,
            probability_errors=[0.02]
            * len(fitting_result.probabilities),  # Default error
            shots_per_point=1000,  # Default shots
            fitting_result=fitting_result,
        )

        # Create analysis result with new pattern
        analysis_result = T2EchoAnalysisResult(
            data=t2_echo_data,
            raw_results=results,
            experiment_instance=self,
        )

        # Analysis handled by T2EchoAnalysisResult class
        df = analysis_result.analyze(
            plot=plot, save_data=save_data, save_image=save_image
        )

        return df

    def circuits(self, **kwargs: Any) -> list[Any]:
        """Generate T2 Echo circuits with automatic transpilation"""
        delay_times = np.logspace(
            np.log10(1.0),
            np.log10(self.max_delay),
            self.delay_points,
        )
        circuits = []

        for delay in delay_times:
            qc = QuantumCircuit(1, 1)
            qc.ry(np.pi / 2, 0)  # First π/2 pulse (creates superposition)

            # First half of the delay
            if delay > 0:
                qc.delay(delay / 2, 0, unit="ns")  # τ/2 delay

            qc.x(0)  # π pulse (echo pulse)

            # Second half of the delay
            if delay > 0:
                qc.delay(delay / 2, 0, unit="ns")  # τ/2 delay

            qc.ry(np.pi / 2, 0)  # Second π/2 pulse (analysis pulse)
            qc.measure(0, 0)  # Measure final state
            circuits.append(qc)

        # Store parameters for analysis and OQTOPUS
        self.experiment_params = {
            "delay_times": delay_times,
            "logical_qubit": 0,
            "physical_qubit": self.physical_qubit,
        }

        # Auto-transpile if physical qubit explicitly specified using base class method
        if (
            hasattr(self, "_physical_qubit_specified")
            and self._physical_qubit_specified
        ):
            circuits = self._transpile_circuits_with_tranqu(
                circuits, 0, self.physical_qubit
            )

        return circuits  # type: ignore

    def _fit_t2_echo_data(
        self, all_results: list[dict[str, Any]]
    ) -> T2EchoFittingResult | None:
        """Fit T2 Echo decay for all data combined"""
        try:
            # Extract data points
            delay_times, probabilities = self._extract_data_points(all_results)
            if len(delay_times) < 4:
                return None

            # Perform fitting
            popt = self._perform_t2_echo_fit(delay_times, probabilities)
            if popt is None:
                return None

            fitted_amplitude, fitted_t2, fitted_offset = popt
            r_squared = self._calculate_r_squared(delay_times, probabilities, popt)

            return T2EchoFittingResult(
                t2_time=fitted_t2,
                amplitude=fitted_amplitude,
                offset=fitted_offset,
                r_squared=r_squared,
                delay_times=delay_times.tolist(),
                probabilities=probabilities.tolist(),
            )

        except Exception as e:
            return T2EchoFittingResult(
                t2_time=5000.0,
                amplitude=0.5,
                offset=0.5,
                r_squared=0.0,
                delay_times=[],
                probabilities=[],
                error_info=str(e),
            )

    def _extract_data_points(
        self, all_results: list[dict[str, Any]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract delay time and probability data points"""
        delay_times: list[float] = []
        probabilities: list[float] = []

        for result in all_results:
            # Get delay from embedded params
            delay = None
            if "params" in result and "delay_time" in result["params"]:
                delay = result["params"]["delay_time"]
            elif (
                hasattr(self, "experiment_params")
                and "delay_times" in self.experiment_params
            ):
                # Fallback to experiment params
                circuit_idx = result.get("params", {}).get(
                    "circuit_index", len(delay_times)
                )
                delay_times_data = self.experiment_params["delay_times"]
                if (
                    hasattr(delay_times_data, "__len__")
                    and hasattr(delay_times_data, "__getitem__")
                    and circuit_idx < len(delay_times_data)
                ):
                    delay = delay_times_data[circuit_idx]

            if delay is None:
                continue

            # Extract probability of |1⟩
            counts = result.get("counts", {})
            total = sum(counts.values())
            if total > 0:
                prob_1 = counts.get("1", counts.get(1, 0)) / total
                delay_times.append(delay)
                probabilities.append(prob_1)

        if not delay_times:
            return np.array([]), np.array([])

        delay_times_array = np.array(delay_times)
        probabilities_array = np.array(probabilities)

        # Sort by delay time
        sort_indices = np.argsort(delay_times_array)
        return delay_times_array[sort_indices], probabilities_array[sort_indices]

    def _perform_t2_echo_fit(
        self, delay_times: np.ndarray, probabilities: np.ndarray
    ) -> np.ndarray | None:
        """Perform T2 Echo exponential decay fitting"""

        def t2_echo_func(t, amplitude, t2, offset):
            # T2 Echo signal: A * exp(-t/T2) + C
            return amplitude * np.exp(-t / t2) + offset

        # Estimate initial parameters
        amplitude_guess = np.max(probabilities) - np.min(probabilities)
        offset_guess = np.min(probabilities)
        t2_guess = self.max_delay / 3  # Initial estimate

        try:
            popt, _ = curve_fit(
                t2_echo_func,
                delay_times,
                probabilities,
                p0=[amplitude_guess, t2_guess, offset_guess],
                bounds=([0, 10, -0.1], [2, self.max_delay * 10, 1.1]),
                maxfev=2000,
            )
            return popt  # type: ignore
        except Exception:
            return None

    def _calculate_r_squared(
        self, delay_times: np.ndarray, probabilities: np.ndarray, popt: np.ndarray
    ) -> float:
        """Calculate R-squared for fit quality"""

        def t2_echo_func(t, amplitude, t2, offset):
            return amplitude * np.exp(-t / t2) + offset

        y_pred = t2_echo_func(delay_times, *popt)
        ss_res = np.sum((probabilities - y_pred) ** 2)
        ss_tot = np.sum((probabilities - np.mean(probabilities)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    def _create_dataframe(
        self, fitting_result: T2EchoFittingResult, device_name: str = "unknown"
    ) -> pd.DataFrame:
        """Create DataFrame from fitting results"""
        df_data = []
        for delay, prob in zip(
            fitting_result.delay_times, fitting_result.probabilities, strict=True
        ):
            df_data.append(
                {
                    "device": device_name,
                    "delay_time": delay,
                    "probability": prob,
                    "t2_time": fitting_result.t2_time,
                    "amplitude": fitting_result.amplitude,
                    "r_squared": fitting_result.r_squared,
                }
            )
        return pd.DataFrame(df_data) if df_data else pd.DataFrame()

    def _get_circuit_params(self) -> list[dict[str, Any]] | None:
        """Get circuit parameters for OQTOPUS"""
        if not hasattr(self, "experiment_params"):
            return None

        delay_times = self.experiment_params["delay_times"]
        logical_qubit = self.experiment_params.get("logical_qubit", 0)
        physical_qubit = self.experiment_params.get("physical_qubit", logical_qubit)

        circuit_params = []
        if hasattr(delay_times, "__iter__"):
            for delay in delay_times:
                param_model = T2EchoCircuitParams(
                    delay_time=float(delay),
                    logical_qubit=(
                        int(logical_qubit)
                        if isinstance(logical_qubit, (int | float))
                        else 0
                    ),
                    physical_qubit=(
                        int(physical_qubit)
                        if isinstance(physical_qubit, (int | float))
                        else 0
                    ),
                )
                circuit_params.append(param_model.model_dump())

        return circuit_params
