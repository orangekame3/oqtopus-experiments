#!/usr/bin/env python3
"""
Local Backend - Qiskit Aer simulator backend for local execution
"""

import uuid
from typing import Any

# Qiskit imports
try:
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import (
        NoiseModel,
        depolarizing_error,
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

    def __init__(self, noise: bool = True, t1: float = 50.0, t2: float = 100.0):
        """
        Initialize local backend

        Args:
            noise: Whether to include noise model (IBM-like parameters)
            t1: T1 relaxation time in microseconds
            t2: T2 dephasing time in microseconds
        """
        self.backend_type = "local"
        self.noise_enabled = noise
        self.t1_us = t1
        self.t2_us = t2

        # Try to initialize Qiskit Aer
        if QISKIT_AVAILABLE:
            self.simulator = AerSimulator()
            if noise:
                self.noise_model = self._create_noise_model()
            else:
                self.noise_model = None
            self.available = True
            noise_status = (
                f"with noise (T1={t1}μs, T2={t2}μs)" if noise else "noiseless"
            )
            print(f"✅ Local backend initialized ({noise_status})")
        else:
            self.simulator = None
            self.noise_model = None
            self.available = False
            print("❌ Local backend not available (missing qiskit-aer)")

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
        two_qubit_error = 0.02  # CNOT error ~2%

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

    def run(self, circuit: Any, shots: int = 1024) -> dict[str, Any]:
        """
        Run circuit on local simulator

        Args:
            circuit: Quantum circuit to run
            shots: Number of shots

        Returns:
            Result dictionary with counts
        """
        if not self.available:
            print("⚠️  Local simulator not available, using fake results")
            return {"counts": {"0": shots // 2, "1": shots // 2}}

        try:
            # Transpile circuit for the simulator
            compiled_circuit = transpile(circuit, self.simulator)

            # Run with or without noise
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
                "backend": "local_aer",
                "noise_enabled": self.noise_enabled,
            }

        except Exception as e:
            print(f"❌ Local simulation failed: {e}")
            # Return fake results on error
            return {
                "job_id": "error",
                "counts": {"0": shots // 2, "1": shots // 2},
                "success": False,
                "error": str(e),
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
            print(f"✅ Noise model updated (T1={t1}μs, T2={t2}μs)")

    def disable_noise(self):
        """Disable noise model"""
        self.noise_enabled = False
        self.noise_model = None
        print("✅ Noise model disabled")

    def enable_noise(self):
        """Enable noise model"""
        self.noise_enabled = True
        if QISKIT_AVAILABLE:
            self.noise_model = self._create_noise_model()
            print(f"✅ Noise model enabled (T1={self.t1_us}μs, T2={self.t2_us}μs)")
