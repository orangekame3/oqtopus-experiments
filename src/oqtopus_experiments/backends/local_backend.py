#!/usr/bin/env python3
"""
Local Backend - Qiskit Aer simulator backend for local execution
"""

import uuid
from typing import Any

# Qiskit imports
try:
    import numpy as np
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import (
        NoiseModel,
        amplitude_damping_error,
        depolarizing_error,
        phase_damping_error,
        thermal_relaxation_error,
    )

    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


class LocalBackend:
    """
    Local backend using Qiskit Aer simulator

    Provides a simple interface for running circuits locally with optional noise models.
    Compatible with usage.py style API.
    """

    def __init__(self, device: str = "noisy", t1: float = 25.0, t2: float = 50.0):
        """
        Initialize local backend

        Args:
            device: Device type ("noisy" or "ideal")
            t1: T1 relaxation time in microseconds
            t2: T2 dephasing time in microseconds
        """
        self.backend_type = "local"
        self.device_name = device
        self.noise_enabled = device == "noisy"
        self.t1_us = t1
        self.t2_us = t2

        # Try to initialize Qiskit Aer
        if QISKIT_AVAILABLE:
            self.simulator = AerSimulator()
            if self.noise_enabled:
                self.noise_model = self._create_noise_model()
            else:
                self.noise_model = None
            self.available = True
            noise_status = (
                f"with noise (T1={t1}μs, T2={t2}μs)"
                if self.noise_enabled
                else "noiseless"
            )
            print(f"Local backend initialized ({device}, {noise_status})")
        else:
            self.simulator = None
            self.noise_model = None
            self.available = False
            print("Local backend not available (missing qiskit-aer)")

    def _create_noise_model(self) -> "NoiseModel":
        """
        Create a realistic noise model based on IBM quantum devices

        Returns:
            NoiseModel with depolarizing and thermal relaxation errors
        """
        noise_model = NoiseModel()

        # Gate times (typical values in nanoseconds)
        gate_times = {
            "sx": 35,  # √X gate
            "x": 35,  # X gate
            "rz": 0,  # Virtual Z rotation (no time)
            "cx": 500,  # CNOT gate
        }

        # Single qubit depolarizing error (typical: 0.1% per gate)
        single_qubit_error = 0.001
        two_qubit_error = 0.06  # CNOT error ~2%

        # Add errors to gates
        for gate_name, gate_time in gate_times.items():
            if gate_name in ["sx", "x"]:
                # Single qubit gates: depolarizing + thermal relaxation
                depol_error = depolarizing_error(single_qubit_error, 1)
                if gate_time > 0:
                    # Convert microseconds to nanoseconds for gate time
                    thermal_error = thermal_relaxation_error(
                        self.t1_us * 1000, self.t2_us * 1000, gate_time, 1
                    )
                    combined_error = depol_error.compose(thermal_error)
                else:
                    combined_error = depol_error
                noise_model.add_all_qubit_quantum_error(combined_error, gate_name)

            elif gate_name == "cx":
                # Two qubit gates: depolarizing error
                depol_error = depolarizing_error(two_qubit_error, 2)
                noise_model.add_all_qubit_quantum_error(depol_error, gate_name)

        return noise_model

    def _has_delay_instructions(self, circuit: Any) -> bool:
        """Check if circuit contains delay instructions"""
        if not QISKIT_AVAILABLE:
            return False
        try:
            for instruction in circuit.data:
                if instruction.operation.name == "delay":
                    return True
            return False
        except Exception:
            return False

    def _get_delay_time_ns(self, circuit: Any) -> float:
        """Extract total delay time in nanoseconds from circuit"""
        if not QISKIT_AVAILABLE:
            return 0.0
        try:
            total_delay = 0.0
            for instruction in circuit.data:
                if instruction.operation.name == "delay":
                    delay_op = instruction.operation
                    # Handle different units
                    if hasattr(delay_op, "duration") and hasattr(delay_op, "unit"):
                        duration = delay_op.duration
                        unit = delay_op.unit
                        if unit == "ns":
                            total_delay += float(duration)
                        elif unit == "s":
                            total_delay += float(duration) * 1e9
                        elif unit == "us":
                            total_delay += float(duration) * 1e3
                        elif unit == "ms":
                            total_delay += float(duration) * 1e6
                        elif unit == "dt":
                            # Qiskit default time unit - assume 1 dt = 1 ns for simulation
                            total_delay += float(duration)
                    elif hasattr(delay_op, "params") and len(delay_op.params) > 0:
                        # Fallback: assume nanoseconds
                        total_delay += float(delay_op.params[0])
            return total_delay
        except Exception:
            return 0.0

    def _create_coherence_noise_model(self, delay_time_ns: float) -> "NoiseModel":
        """Create noise model for coherence experiments (T1, T2 Echo, etc.) with given delay time"""
        noise_model = NoiseModel()

        if delay_time_ns <= 0:
            return noise_model

        # Convert delay time to seconds
        delay_sec = delay_time_ns * 1e-9
        t1_sec = self.t1_us * 1e-6
        t2_sec = self.t2_us * 1e-6

        # Calculate amplitude damping probability for T1 decay
        p_amp = 1 - np.exp(-delay_sec / t1_sec)

        # Calculate phase damping probability for T2 dephasing
        # 1/T2_phi = 1/T2 - 1/(2*T1)
        if t2_sec > 0 and (1 / t2_sec) > (1 / (2 * t1_sec)):
            t2_phi_sec = 1 / (1 / t2_sec - 1 / (2 * t1_sec))
            p_phase = 1 - np.exp(-delay_sec / t2_phi_sec)
        else:
            p_phase = 0.0

        # Create damping errors
        if p_amp > 0:
            amp_error = amplitude_damping_error(p_amp)
            if p_phase > 0:
                phase_error = phase_damping_error(p_phase)
                combined_error = amp_error.compose(phase_error)
            else:
                combined_error = amp_error

            # Add error to delay instruction
            noise_model.add_all_qubit_quantum_error(combined_error, "delay")

        # Also add standard gate errors for other operations
        single_qubit_error = 0.001
        depol_error = depolarizing_error(single_qubit_error, 1)
        gate_time_ns = 35  # X gate time
        thermal_error = thermal_relaxation_error(
            self.t1_us * 1000, self.t2_us * 1000, gate_time_ns, 1
        )
        x_error = depol_error.compose(thermal_error)
        noise_model.add_all_qubit_quantum_error(x_error, "x")

        return noise_model

    def run(
        self, circuit: Any, shots: int = 1024, circuit_params: dict[str, Any] = None
    ) -> dict[str, Any]:
        """
        Run circuit on local simulator

        Args:
            circuit: Quantum circuit to run
            shots: Number of shots
            circuit_params: Optional parameters (ignored for local backend, for compatibility)

        Returns:
            Result dictionary with counts
        """
        if not self.available:
            print("Local simulator not available, using fake results")
            return {"counts": {"0": shots // 2, "1": shots // 2}}

        try:
            # Check if this is a coherence experiment (circuit with delay instructions)
            has_delays = self._has_delay_instructions(circuit)

            if has_delays and self.noise_enabled:
                # Special handling for coherence experiments (T1, T2 Echo, etc.)
                delay_time_ns = self._get_delay_time_ns(circuit)

                # Use density matrix simulation for proper relaxation handling
                coherence_simulator = AerSimulator(method="density_matrix")
                compiled_circuit = transpile(circuit, coherence_simulator)

                # Create coherence-specific noise model for this delay time
                coherence_noise_model = self._create_coherence_noise_model(
                    delay_time_ns
                )

                job = coherence_simulator.run(
                    compiled_circuit, shots=shots, noise_model=coherence_noise_model
                )
            else:
                # Standard handling for other experiments
                compiled_circuit = transpile(circuit, self.simulator)

                if self.noise_model:
                    job = self.simulator.run(
                        compiled_circuit, shots=shots, noise_model=self.noise_model
                    )
                else:
                    job = self.simulator.run(compiled_circuit, shots=shots)

            result = job.result()
            counts = result.get_counts()

            # Generate unique job ID
            job_id = str(uuid.uuid4())[:8]

            return {
                "job_id": job_id,
                "counts": dict(counts),
                "shots": shots,
                "success": True,
                "backend": self.device_name,
                "noise_enabled": self.noise_enabled,
                "params": circuit_params or {},
            }

        except Exception as e:
            print(f"Local simulation failed: {e}")
            # Return fake results on error
            return {
                "job_id": "error",
                "counts": {"0": shots // 2, "1": shots // 2},
                "success": False,
                "error": str(e),
                "params": circuit_params or {},
            }

    def submit(self, circuit: Any, shots: int = 1024) -> str:
        """
        Submit circuit (synchronous for local backend)

        Args:
            circuit: Quantum circuit to submit
            shots: Number of shots

        Returns:
            Job ID string
        """
        # For local backend, we run immediately and cache the result
        result = self.run(circuit, shots)
        job_id = result["job_id"]

        # Cache result for get_result method
        if not hasattr(self, "_cached_results"):
            self._cached_results = {}
        self._cached_results[job_id] = result

        return str(job_id)

    def get_result(self, job_id: str) -> dict[str, Any]:
        """
        Get result from job ID

        Args:
            job_id: Job ID from submit()

        Returns:
            Result dictionary with counts
        """
        if hasattr(self, "_cached_results") and job_id in self._cached_results:
            return self._cached_results[job_id]
        else:
            # Return default if job not found
            return {"counts": {"0": 500, "1": 500}, "job_id": job_id, "success": False}

    def get_noise_info(self) -> dict[str, Any]:
        """
        Get information about the noise model

        Returns:
            Dictionary with noise model information
        """
        return {
            "noise_enabled": self.noise_enabled,
            "t1_us": self.t1_us,
            "t2_us": self.t2_us,
            "backend_type": self.backend_type,
            "available": self.available,
        }

    def set_noise_parameters(self, t1: float, t2: float):
        """
        Update noise model parameters

        Args:
            t1: T1 relaxation time in microseconds
            t2: T2 dephasing time in microseconds
        """
        self.t1_us = t1
        self.t2_us = t2
        if self.noise_enabled and QISKIT_AVAILABLE:
            self.noise_model = self._create_noise_model()
            print(f"Noise model updated (T1={t1}μs, T2={t2}μs)")

    def disable_noise(self):
        """Switch to ideal simulation"""
        self.noise_enabled = False
        self.noise_model = None
        self.device_name = "ideal"
        print("Switched to ideal simulation")

    def enable_noise(self):
        """Switch to noisy simulation"""
        self.noise_enabled = True
        self.device_name = "noisy"
        if QISKIT_AVAILABLE:
            self.noise_model = self._create_noise_model()
            print(
                f"Switched to noisy simulation (T1={self.t1_us}μs, T2={self.t2_us}μs)"
            )
