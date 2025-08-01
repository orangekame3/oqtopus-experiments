#!/usr/bin/env python3
"""
Base Experiment Class - Base class for experiments
Base class for all quantum experiment classes
"""

import time
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from ..models.experiment_result import ExperimentResult
from .data_manager import ExperimentDataManager


class BaseExperiment(ABC):
    """
    Base class for quantum experiments

    All concrete experiment classes (CHSH, Rabi, T1, etc.) inherit from this
    Common functionality: OQTOPUS connection, parallel execution, data management
    """

    def __init__(
        self,
        experiment_name: str | None = None,
    ):
        """
        Initialize base experiment

        Args:
            experiment_name: Experiment name
        """
        self.experiment_name = (
            experiment_name or f"{self.__class__.__name__.lower()}_{int(time.time())}"
        )
        self.data_manager = ExperimentDataManager(self.experiment_name)

        # Initialize optional attributes used by save methods
        self.transpiler_options: dict[str, Any] = {}
        self.mitigation_options: dict[str, Any] = {}
        self.anemone_basis_gates: list[str] = []
        self.oqtopus_available: bool = False
        self.experiment_params: dict[str, Any] = {}

        print(f"{self.__class__.__name__}: {self.experiment_name}")

    # Abstract methods: implemented in each experiment class
    @abstractmethod
    def circuits(self, **kwargs) -> list[Any]:
        """Experiment-specific circuit creation (implemented in each experiment class)"""
        pass

    @abstractmethod
    def analyze(
        self,
        results: dict[str, list[dict[str, Any]]],
        plot: bool = True,
        save_data: bool = True,
        save_image: bool = True,
    ) -> pd.DataFrame:
        """
        Experiment-specific result analysis (implemented in each experiment class)

        Args:
            results: Raw experiment results from backend execution
            plot: Whether to generate plots
            save_data: Whether to save analysis results (handled by experiment classes)
            save_image: Whether to save generated plots

        Returns:
            DataFrame with analysis results

        Note:
            All concrete implementations should return pd.DataFrame for consistency.
            Data saving is handled by experiment classes via data manager.
        """
        pass

    def save_experiment_data(
        self,
        results: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        experiment_type: str | None = None,
    ) -> str:
        """Save experiment data using data manager"""
        exp_type = experiment_type or self.__class__.__name__.lower().replace(
            "experiment", ""
        )
        return self.data_manager.save_results(
            results=results, metadata=metadata or {}, experiment_type=exp_type
        )

    # Common save methods
    def save_job_ids(
        self,
        job_ids: dict[str, list[str]],
        metadata: dict[str, Any] | None = None,
        filename: str = "job_ids",
    ) -> str:
        """Save job IDs"""
        save_data = {
            "job_ids": job_ids,
            "submitted_at": time.time(),
            "experiment_type": self.__class__.__name__,
            "oqtopus_config": {
                "transpiler_options": self.transpiler_options,
                "mitigation_options": self.mitigation_options,
                "basis_gates": self.anemone_basis_gates,
            },
            "metadata": metadata or {},
        }
        return self.data_manager.save_data(save_data, filename)

    def save_raw_results(
        self,
        results: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        filename: str = "raw_results",
    ) -> str:
        """Save raw results"""
        save_data = {
            "results": results,
            "saved_at": time.time(),
            "experiment_type": self.__class__.__name__,
            "oqtopus_available": self.oqtopus_available,
            "metadata": metadata or {},
        }
        return self.data_manager.save_data(save_data, filename)

    def save_experiment_summary(self) -> str:
        """Save experiment summary"""
        return self.data_manager.summary()

    def save_plot_figure(
        self,
        figure,
        experiment_name: str,
        save_formats: list[str] | None = None,
    ) -> str | None:
        """
        Save plot figure using data manager with unified .results/ organization

        Args:
            figure: Plotly figure object to save
            experiment_name: Name of the experiment for filename
            save_formats: List of formats to save (default: ["png"])

        Returns:
            Path to saved plot file (primary format) or None if saving failed

        Note:
            Saves plots to .results/experiment_TIMESTAMP/plots/ directory
            via data manager for consistent file organization.
        """
        if figure is None:
            return None

        try:
            from ..utils.visualization import save_plotly_figure

            # Use data manager to get plots directory
            plots_dir = self.data_manager.get_plots_directory()

            # Default to PNG format
            if save_formats is None:
                save_formats = ["png"]

            # Save figure using visualization utility
            saved_path = save_plotly_figure(
                figure, name=experiment_name, images_dir=plots_dir, formats=save_formats
            )

            return saved_path

        except Exception as e:
            print(f"Warning: Failed to save plot figure: {e}")
            return None

    # Template method: overall experiment flow
    def run(
        self,
        backend,
        shots: int = 1024,
        mitigation_info: dict[str, Any] | None = None,
    ) -> ExperimentResult:
        """
        Run experiment sequentially using backend

        Args:
            backend: Backend instance (required)
            shots: Number of shots per circuit
            mitigation_info: Optional mitigation settings passed to backend
        """
        if backend is None:
            raise ValueError("Backend is required for execution")

        print(f"Running {self.__class__.__name__}")

        # Get circuits
        if hasattr(self, "_circuits") and self._circuits is not None:
            circuits = self._circuits
            print(f"Using pre-created circuits: {len(circuits)} circuits")
        else:
            circuits = self.circuits()
            print(f"Created default circuits: {len(circuits)} circuits")

        # Store backend info for device name resolution
        if hasattr(backend, "device_name"):
            self._last_backend_device = backend.device_name
        elif hasattr(backend, "backend_type"):
            self._last_backend_device = backend.backend_type
        else:
            self._last_backend_device = "unknown"

        # Auto-transpile if physical qubit specified
        circuits, was_transpiled = self._auto_transpile_if_needed(circuits, backend)

        # Get circuit parameters if available from experiment
        circuit_params = None
        if hasattr(self, "_get_circuit_params"):
            circuit_params = self._get_circuit_params()

        # Sequential execution
        raw_results = {}
        for i, circuit in enumerate(circuits):
            try:
                # Prepare parameters for this specific circuit
                single_circuit_params = None
                if circuit_params and i < len(circuit_params):
                    single_circuit_params = circuit_params[i]

                # Get experiment name for OqtopusBackend
                experiment_name = getattr(self, "experiment_name", None)

                # Pass experiment_name to OqtopusBackend if available
                if (
                    hasattr(backend, "backend_type")
                    and backend.backend_type == "oqtopus"
                    and experiment_name
                ):
                    result = backend.run(
                        circuit,
                        shots,
                        circuit_params=single_circuit_params,
                        mitigation_info=mitigation_info,
                        experiment_name=experiment_name,
                    )
                else:
                    result = backend.run(
                        circuit,
                        shots,
                        circuit_params=single_circuit_params,
                        mitigation_info=mitigation_info,
                    )
                raw_results[f"circuit_{i}"] = [result]
            except Exception as e:
                print(f"Backend execution failed: {e}")
                raw_results[f"circuit_{i}"] = []

        print(f"{self.__class__.__name__} completed")

        return ExperimentResult(
            raw_results=raw_results,
            experiment_instance=self,
            experiment_type=self.__class__.__name__.lower().replace("experiment", ""),
        )

    def _auto_transpile_if_needed(self, circuits, backend):
        """
        Auto-transpile circuits if physical qubit is specified and differs from logical

        Args:
            circuits: Circuit collection or list
            backend: Backend instance

        Returns:
            Tuple of (circuits, was_transpiled)
            - circuits: Transpiled circuits if needed, otherwise original circuits
            - was_transpiled: Boolean indicating if transpilation occurred
        """
        # Check if physical qubits were explicitly specified
        if (
            hasattr(self, "_physical_qubits_specified")
            and self._physical_qubits_specified
        ):
            # Two-qubit case (e.g., CHSH experiments)
            physical_qubit_0 = self.experiment_params.get("physical_qubit_0")
            physical_qubit_1 = self.experiment_params.get("physical_qubit_1")
            logical_qubit_0 = self.experiment_params.get("logical_qubit_0", 0)
            logical_qubit_1 = self.experiment_params.get("logical_qubit_1", 1)

            if physical_qubit_0 is not None and physical_qubit_1 is not None:
                if hasattr(backend, "transpile"):
                    print(
                        f"Auto-transpiling circuits: logical qubits [{logical_qubit_0}, {logical_qubit_1}] → physical qubits [{physical_qubit_0}, {physical_qubit_1}]"
                    )
                    try:
                        transpiled = backend.transpile(
                            circuits,
                            physical_qubits=[physical_qubit_0, physical_qubit_1],
                        )
                        print("Transpilation successful")
                        return transpiled, True
                    except Exception as e:
                        print(f"Transpilation failed: {e}, using original circuits")
                        return circuits, False
                else:
                    print(
                        "Backend does not support transpilation, using original circuits"
                    )
                    return circuits, False
        elif (
            hasattr(self, "_physical_qubit_specified")
            and self._physical_qubit_specified
        ):
            # Single-qubit case (legacy support)
            physical_qubit = self.experiment_params.get("physical_qubit")
            logical_qubit = self.experiment_params.get("logical_qubit", 0)

            if physical_qubit is not None:
                if hasattr(backend, "transpile"):
                    print(
                        f"Auto-transpiling circuits: logical qubit {logical_qubit} → physical qubit {physical_qubit}"
                    )
                    try:
                        transpiled = backend.transpile(
                            circuits, physical_qubits=[physical_qubit]
                        )
                        print("Transpilation successful")
                        return transpiled, True
                    except Exception as e:
                        print(f"Transpilation failed: {e}, using original circuits")
                        return circuits, False
                else:
                    print(
                        "Backend does not support transpilation, using original circuits"
                    )
                    return circuits, False

        # No explicit physical qubit - let OQTOPUS handle transpilation
        return circuits, False

    def _transpile_circuits_with_tranqu(
        self, circuits, logical_qubit=0, physical_qubit=None
    ):
        """
        Transpile circuits using Tranqu (backend-independent)

        Args:
            circuits: Circuit collection or list
            logical_qubit: Logical qubit index (default: 0)
            physical_qubit: Target physical qubit

        Returns:
            Transpiled circuits or original circuits if transpilation fails
        """
        if physical_qubit is None:
            return circuits

        try:
            from tranqu import Tranqu

            from ..devices import DeviceInfo

            device_info = DeviceInfo("anemone")
            if not device_info.available:
                print("Device info not available, using original circuits")
                return circuits

            tranqu = Tranqu()
            transpiled_circuits = []

            for i, circuit in enumerate(circuits):
                try:
                    initial_layout = {circuit.qubits[logical_qubit]: physical_qubit}
                    result = tranqu.transpile(
                        program=circuit,
                        transpiler_lib="qiskit",
                        program_lib="qiskit",
                        transpiler_options={
                            "basis_gates": ["sx", "x", "rz", "cx"],
                            "optimization_level": 1,
                            "initial_layout": initial_layout,
                        },
                        device=device_info.device_info,
                        device_lib="oqtopus",
                    )
                    transpiled_circuits.append(result.transpiled_program)
                except Exception as e:
                    print(f"Circuit {i + 1} transpilation failed: {e}")
                    transpiled_circuits.append(circuit)

            print(
                f"Transpiled {len(transpiled_circuits)} circuits to physical qubit {physical_qubit}"
            )
            return transpiled_circuits

        except ImportError:
            print("Tranqu not available, using original circuits")
            return circuits
        except Exception as e:
            print(f"Transpilation failed: {e}, using original circuits")
            return circuits

    def run_parallel(
        self,
        backend,
        shots: int = 1024,
        workers: int = 4,
        mitigation_info: dict[str, Any] | None = None,
    ) -> ExperimentResult:
        """
        Run experiment in parallel using backend

        Args:
            backend: Backend instance (required)
            shots: Number of shots per circuit
            workers: Number of parallel workers
            mitigation_info: Optional mitigation settings passed to backend
        """
        if backend is None:
            raise ValueError("Backend is required for parallel execution")

        print(f"Running {self.__class__.__name__} in parallel")

        # Get circuits
        if hasattr(self, "_circuits") and self._circuits is not None:
            circuits = self._circuits
            print(f"Using pre-created circuits: {len(circuits)} circuits")
        else:
            circuits = self.circuits()
            print(f"Created default circuits: {len(circuits)} circuits")

        # Auto-transpile if physical qubit specified
        circuits, was_transpiled = self._auto_transpile_if_needed(circuits, backend)

        # Debug: show transpiled circuit if requested
        if was_transpiled and len(circuits) > 1:
            print("Transpiled circuit sample:")
            print(circuits[1].draw())

        # Execute in parallel
        if hasattr(backend, "submit_parallel") and hasattr(backend, "collect_parallel"):
            # Backend with parallel support
            print(
                f"Submitting {len(circuits)} circuits in parallel to {backend.device_name}"
            )

            # Get circuit parameters if available from experiment
            circuit_params = None
            if hasattr(self, "_get_circuit_params"):
                circuit_params = self._get_circuit_params()

            # Check if physical qubit mapping is needed (disable OQTOPUS transpilation)
            disable_transpilation = self._should_disable_transpilation()

            # Get experiment name for OqtopusBackend
            experiment_name = getattr(self, "experiment_name", None)

            # Pass experiment_name to OqtopusBackend if available
            if (
                hasattr(backend, "backend_type")
                and backend.backend_type == "oqtopus"
                and experiment_name
            ):
                job_ids = backend.submit_parallel(
                    circuits,
                    shots,
                    circuit_params,
                    disable_transpilation,
                    mitigation_info,
                    experiment_name,
                )
            else:
                job_ids = backend.submit_parallel(
                    circuits,
                    shots,
                    circuit_params,
                    disable_transpilation,
                    mitigation_info,
                )
            print(f"Collecting {len(job_ids)} results")
            results = backend.collect_parallel(job_ids)

            # Format results
            raw_results = {}
            for i, result in enumerate(results):
                raw_results[f"circuit_{i}"] = [result] if result else []
        else:
            # Sequential execution for backends without parallel support
            print("Backend does not support parallel execution, running sequentially")

            # Get circuit parameters if available from experiment
            circuit_params = None
            if hasattr(self, "_get_circuit_params"):
                circuit_params = self._get_circuit_params()

            raw_results = {}
            for i, circuit in enumerate(circuits):
                try:
                    # Prepare parameters for this specific circuit
                    single_circuit_params = None
                    if circuit_params and i < len(circuit_params):
                        single_circuit_params = circuit_params[i]

                    # Get experiment name for OqtopusBackend (in run_parallel sequential fallback)
                    experiment_name = getattr(self, "experiment_name", None)

                    # Pass experiment_name to OqtopusBackend if available
                    if (
                        hasattr(backend, "backend_type")
                        and backend.backend_type == "oqtopus"
                        and experiment_name
                    ):
                        result = backend.run(
                            circuit,
                            shots,
                            circuit_params=single_circuit_params,
                            mitigation_info=mitigation_info,
                            experiment_name=experiment_name,
                        )
                    else:
                        result = backend.run(
                            circuit,
                            shots,
                            circuit_params=single_circuit_params,
                            mitigation_info=mitigation_info,
                        )
                    raw_results[f"circuit_{i}"] = [result]
                except Exception as e:
                    print(f"Backend execution failed: {e}")
                    raw_results[f"circuit_{i}"] = []

        print(f"{self.__class__.__name__} completed")

        return ExperimentResult(
            raw_results=raw_results,
            experiment_instance=self,
            experiment_type=self.__class__.__name__.lower().replace("experiment", ""),
        )

    def _should_disable_transpilation(self) -> bool:
        """
        Check if OQTOPUS transpilation should be disabled

        Returns True if physical qubit was explicitly specified by user
        """
        # Check for two-qubit experiment with explicit physical qubits (e.g., CHSH)
        if hasattr(self, "_physical_qubits_specified"):
            return bool(self._physical_qubits_specified)

        # Check for single-qubit experiment with explicit physical qubit
        if hasattr(self, "_physical_qubit_specified"):
            return bool(self._physical_qubit_specified)

        # Fallback: check experiment params (legacy compatibility)
        if not hasattr(self, "experiment_params") or not self.experiment_params:
            return False

        # Check for two-qubit case
        physical_qubit_0 = self.experiment_params.get("physical_qubit_0")
        physical_qubit_1 = self.experiment_params.get("physical_qubit_1")
        if physical_qubit_0 is not None and physical_qubit_1 is not None:
            return True

        # Check for single-qubit case
        physical_qubit = self.experiment_params.get("physical_qubit")
        return physical_qubit is not None
