#!/usr/bin/env python3
"""
Parity Oscillation Experiment with Local Simulator
Demonstrates GHZ state decoherence and parity oscillations
"""

from oqtopus_experiments.backends import LocalBackend
from oqtopus_experiments.experiments import ParityOscillation


def main():
    print("=== Parity Oscillation: GHZ State Decoherence Study ===")

    # Local backend for ideal simulation
    backend = LocalBackend(
        device="noisy"
    )  # Use noisy simulator for realistic decoherence

    # Create Parity Oscillation experiment with simplified parameters for testing
    for num_qubits in [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
    ]:
        exp = ParityOscillation(
            experiment_name="parity_oscillation_test",
            num_qubits=num_qubits,  # Fixed parameter name
            delay_us=20.0,  # Fixed parameter name
            shots=1000,
        )

        print(f"Testing with N={exp.num_qubits} qubits")
        print(f"Delay: {exp.delay_us} μs")
        print(f"Phase points: {exp.phase_points}")

        # Run experiment with unified interface
        print("\nRunning Parity Oscillation experiment...")
        result = exp.run(backend=backend, shots=1000)

        # Analyze results
        print("Analyzing parity oscillations...")
        df = result.analyze(plot=True, save_data=True, save_image=True)
        print(df.head())


if __name__ == "__main__":
    main()
