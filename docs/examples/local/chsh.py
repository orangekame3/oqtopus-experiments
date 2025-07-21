#!/usr/bin/env python3
"""
CHSH Bell inequality experiment with Local Simulator
"""


from oqtopus_experiments.backends import LocalBackend
from oqtopus_experiments.experiments import CHSH


def main():
    print("=== CHSH Bell Inequality Test ===")

    # Local backend for Qiskit Aer simulator
    backend = LocalBackend(
        device="ideal"
    )  # Use ideal simulator for perfect entanglement

    # Create CHSH experiment with optimal theta angle for Bell violation
    exp = CHSH(
        experiment_name="chsh_bell_test",
        shots_per_circuit=2000,  # More shots for better statistics
    )

    # Run CHSH experiment with optimal theta = π/4 (45°) for maximum CHSH violation
    import math

    result = exp.run(backend=backend, shots=2000, theta=math.pi / 4)

    # Analyze results and check for Bell inequality violation
    df = result.analyze(plot=True, save_data=True, save_image=True)
    print(df.head())


if __name__ == "__main__":
    main()
