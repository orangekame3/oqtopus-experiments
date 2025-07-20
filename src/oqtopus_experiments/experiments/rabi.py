#!/usr/bin/env python3
"""
Rabi Experiment Class - Simplified Rabi oscillation experiment
"""

from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from scipy.optimize import curve_fit

from ..core.base_experiment import BaseExperiment
from ..models.circuit_collection import CircuitCollection


class Rabi(BaseExperiment):
    """
    Rabi oscillation experiment

    Creates circuits with varying rotation amplitudes and analyzes oscillation patterns.
    """

    def __init__(
        self,
        experiment_name: str = None,
        physical_qubit: int = 0,
        amplitude_points: int = 10,
        max_amplitude: float = 2.0,
    ):
        super().__init__(experiment_name or "rabi_experiment")
        self.physical_qubit = physical_qubit
        self.amplitude_points = amplitude_points
        self.max_amplitude = max_amplitude

    def analyze(self, results: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        """
        Analyze Rabi experiment results using embedded parameters

        Args:
            results: Raw measurement results with embedded parameters

        Returns:
            Analysis results with π-pulse amplitude estimates
        """
        if not results:
            return {"error": "No results to analyze"}

        analysis_results = {}

        # Handle both device-based and circuit-based result structures
        if any(key.startswith("circuit_") for key in results.keys()):
            # Circuit-based structure from run_parallel: {circuit_0: [result], circuit_1: [result]}
            device_results = []
            for circuit_key in sorted(results.keys()):
                if results[circuit_key]:
                    device_results.extend(results[circuit_key])

            # Process all results for a single "device" (backend)
            device_name = "backend"
            result_groups = {device_name: device_results}
        else:
            # Device-based structure: {device_name: [results]}
            result_groups = results

        for device, device_results in result_groups.items():
            if not device_results:
                continue

            # Extract amplitudes and probabilities from results with embedded params
            amplitudes = []
            probabilities = []

            for result in device_results:
                # Get amplitude from embedded params (preferred) or fallback to experiment params
                amplitude = None
                if "params" in result and "amplitude" in result["params"]:
                    amplitude = result["params"]["amplitude"]
                elif (
                    hasattr(self, "experiment_params")
                    and "amplitudes" in self.experiment_params
                ):
                    # Fallback: use index-based lookup from experiment params
                    circuit_idx = result.get("params", {}).get(
                        "circuit_index", len(amplitudes)
                    )
                    if circuit_idx < len(self.experiment_params["amplitudes"]):
                        amplitude = self.experiment_params["amplitudes"][circuit_idx]

                if amplitude is None:
                    print(
                        f"⚠️  Warning: Could not determine amplitude for result, skipping"
                    )
                    continue

                amplitudes.append(amplitude)

                # Extract probability of measuring |1⟩
                counts = result.get("counts", {})
                total = sum(counts.values())
                if total > 0:
                    prob_1 = counts.get("1", counts.get(1, 0)) / total
                    probabilities.append(prob_1)
                else:
                    probabilities.append(0.0)

            if not amplitudes:
                analysis_results[device] = {
                    "error": "No valid amplitude parameters found"
                }
                continue

            amplitudes = np.array(amplitudes)
            probabilities = np.array(probabilities)

            # Sort by amplitude for proper fitting
            sort_indices = np.argsort(amplitudes)
            amplitudes = amplitudes[sort_indices]
            probabilities = probabilities[sort_indices]

            # Simple Rabi fitting: P = A * sin²(π * amp * freq) + offset
            try:

                def rabi_func(amp, amplitude, frequency, offset):
                    return amplitude * np.sin(np.pi * amp * frequency) ** 2 + offset

                initial_guess = [0.5, 1.0, 0.5]  # amplitude, frequency, offset
                popt, _ = curve_fit(
                    rabi_func, amplitudes, probabilities, p0=initial_guess
                )

                fitted_amplitude, fitted_freq, fitted_offset = popt
                pi_amplitude = 0.5 / fitted_freq  # First π pulse

                # Calculate R² for fit quality
                y_pred = rabi_func(amplitudes, *popt)
                ss_res = np.sum((probabilities - y_pred) ** 2)
                ss_tot = np.sum((probabilities - np.mean(probabilities)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                analysis_results[device] = {
                    "pi_amplitude": pi_amplitude,
                    "frequency": fitted_freq,
                    "fit_amplitude": fitted_amplitude,
                    "offset": fitted_offset,
                    "r_squared": r_squared,
                    "amplitudes": amplitudes.tolist(),
                    "probabilities": probabilities.tolist(),
                }

                print(
                    f"📊 {device}: π-pulse amp = {pi_amplitude:.3f}, freq = {fitted_freq:.3f}, R² = {r_squared:.3f} (using embedded params)"
                )

            except Exception as e:
                print(f"⚠️  Fitting failed for {device}: {e}")
                analysis_results[device] = {
                    "pi_amplitude": 0.5,
                    "frequency": 1.0,
                    "error": str(e),
                    "amplitudes": amplitudes.tolist(),
                    "probabilities": probabilities.tolist(),
                }

        # Return overall results
        all_amplitudes = []
        for device_result in analysis_results.values():
            if "amplitudes" in device_result:
                all_amplitudes.extend(device_result["amplitudes"])

        return {
            "amplitudes": sorted(list(set(all_amplitudes))) if all_amplitudes else [],
            "devices": analysis_results,
        }

    def circuits(
        self,
        amplitude_points: int = None,
        max_amplitude: float = None,
        qubit: int = None,
    ) -> list[Any]:
        """Create Rabi oscillation circuits"""
        # Use constructor defaults if not provided
        amplitude_points = amplitude_points or self.amplitude_points
        max_amplitude = max_amplitude or self.max_amplitude
        qubit = qubit if qubit is not None else self.physical_qubit

        amplitudes = np.linspace(0, max_amplitude, amplitude_points)

        circuits = []
        # Always create circuits with logical qubits first (starting from 0)
        logical_qubit = 0

        for amplitude in amplitudes:
            qc = QuantumCircuit(1, 1)
            if amplitude > 0:
                qc.rx(amplitude * np.pi, logical_qubit)
            qc.measure(logical_qubit, 0)
            circuits.append(qc)

        # Store parameters for analysis
        self.experiment_params = {
            "amplitudes": amplitudes,
            "logical_qubit": logical_qubit,
            "physical_qubit": qubit,  # Target physical qubit
        }

        print(
            f"Created {len(circuits)} Rabi circuits (amplitude range: {amplitudes[0]:.3f} - {amplitudes[-1]:.3f})"
        )

        # If physical qubit specified, perform transpilation
        if qubit is not None:
            print(f"Physical qubit {qubit} specified, performing transpilation...")
            circuits = self._transpile_circuits_if_needed(circuits, logical_qubit, qubit)
        else:
            print("Rabi circuit structure: |0⟩ → RX(amp·π) → measure (expected: oscillation with amp)")

        circuit_collection = CircuitCollection(circuits)
        self._circuits = circuit_collection
        return circuit_collection

    def _get_circuit_params(self) -> list[dict]:
        """Get parameters for each circuit for OQTOPUS description field"""
        if not hasattr(self, "experiment_params"):
            return None

        amplitudes = self.experiment_params["amplitudes"]
        logical_qubit = self.experiment_params.get("logical_qubit", 0)
        physical_qubit = self.experiment_params.get("physical_qubit", logical_qubit)

        return [
            {
                "experiment": "rabi",
                "amplitude": float(amp),
                "logical_qubit": logical_qubit,
                "physical_qubit": physical_qubit,
                "rotation_angle": float(amp * np.pi),
            }
            for amp in amplitudes
        ]

    def _transpile_circuits_if_needed(self, circuits, logical_qubit, physical_qubit):
        """
        Transpile circuits using available backend for physical qubit mapping
        
        Args:
            circuits: List of circuits to transpile
            logical_qubit: Source logical qubit index
            physical_qubit: Target physical qubit index
            
        Returns:
            Transpiled circuits or original if transpilation fails
        """
        try:
            from tranqu import Tranqu
            from ..devices import DeviceInfo
            
            # Get device info for the physical qubit
            device_info = DeviceInfo("anemone")  # Default to anemone for now
            if not device_info.available:
                print("⚠️  Device info not available, using original circuits")
                return circuits
                
            tranqu = Tranqu()
            transpiled_circuits = []
            
            for i, circuit in enumerate(circuits):
                try:
                    # Create initial layout mapping logical to physical qubit
                    initial_layout = {circuit.qubits[logical_qubit]: physical_qubit}
                    
                    # Transpile with physical qubit mapping
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
                    print(f"⚠️  Circuit {i+1} transpilation failed: {e}, using original")
                    transpiled_circuits.append(circuit)
            
            print(f"✅ Transpiled {len(transpiled_circuits)} circuits to physical qubit {physical_qubit}")
            return transpiled_circuits
            
        except ImportError:
            print("⚠️  Tranqu not available, using original circuits")
            return circuits
        except Exception as e:
            print(f"⚠️  Transpilation failed: {e}, using original circuits")
            return circuits

    def save_experiment_data(
        self, results: dict[str, Any], metadata: dict[str, Any] = None
    ) -> str:
        """Save Rabi experiment data"""
        return self.data_manager.save_results(results, metadata or {}, "rabi")
