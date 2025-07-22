#!/usr/bin/env python3
"""
Parity Oscillation Experiment with Local Simulator
Demonstrates GHZ state decoherence and parity oscillations
"""

from oqtopus_experiments.backends import LocalBackend
from oqtopus_experiments.experiments import ParityOscillation


def main():

    # Local backend for ideal simulation
    backend = LocalBackend(device="noisy")

    # Create Parity Oscillation experiment
    for num_qubits in [
        1,
        2,
        3,
        4,
    ]:
        exp = ParityOscillation(
            experiment_name="parity_oscillation_test",
            num_qubits=num_qubits,
            delay_us=0.0,
        )

        print(f"Testing with N={exp.num_qubits} qubits")
        print(f"Delay: {exp.delay_us} μs")
        print(f"Phase points: {exp.phase_points}")

        # Run the experimen
        result = exp.run(backend=backend, shots=1000)

        # Analyze results
        df = result.analyze(plot=True, save_data=True, save_image=True)
        print(df.head())


if __name__ == "__main__":
    main()
