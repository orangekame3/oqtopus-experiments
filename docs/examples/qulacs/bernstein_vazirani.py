#!/usr/bin/env python3
"""
Bernstein-Vazirani Algorithm Example (Qulacs Backend)

This example demonstrates how to use the BernsteinVazirani experiment
to find a hidden binary string with a single quantum query using the
OQTOPUS Qulacs backend.

The Qulacs backend provides fast noiseless simulation for testing
quantum algorithms before running on real hardware.
"""

from oqtopus_experiments.backends import OqtopusBackend
from oqtopus_experiments.experiments import BernsteinVazirani


def main():
    """Run Bernstein-Vazirani algorithm with Qulacs backend"""

    # OQTOPUS backend with Qulacs simulator
    backend = OqtopusBackend(device="qulacs")

    # Create BernsteinVazirani experiment with a longer secret string
    experiment = BernsteinVazirani(
        experiment_name="bernstein_vazirani_qulacs",
        secret_string="110101",  # 6-bit secret string
    )

    # Run experiment with backend
    result = experiment.run(backend=backend, shots=1000)

    # Analyze results
    df = result.analyze()
    print("Measurement results:")
    print(df.head())

    # Print summary
    if not df.empty:
        top_result = df.iloc[0]
        print(f"\nSecret string: {top_result['secret_string']}")
        print(f"Measured string: {top_result['measured_string']}")
        print(f"Success probability: {top_result['success_probability']:.3f}")
        print(f"Correct: {top_result['is_correct']}")


if __name__ == "__main__":
    main()
