#!/usr/bin/env python3
"""
Bernstein-Vazirani Algorithm Example (Local Backend)

This example demonstrates the Bernstein-Vazirani algorithm using a noisy
simulation to show the statistical nature of quantum measurements.
"""

from oqtopus_experiments.backends import LocalBackend
from oqtopus_experiments.experiments import BernsteinVazirani


def main():
    """Run Bernstein-Vazirani algorithm with noisy backend"""

    # Local backend with noise model
    backend = LocalBackend(device="noisy")

    # Create BernsteinVazirani experiment with a secret string
    experiment = BernsteinVazirani(
        experiment_name="bernstein_vazirani_local",
        secret_string="011",  # 3-bit secret for testing
    )

    # Run experiment with more shots to see statistical distribution
    result = experiment.run(backend=backend, shots=2000)
    print(experiment.circuits()[0].draw())

    # Analyze results
    df = result.analyze()
    print("Measurement results:")
    print(df)

    # Print summary
    if not df.empty:
        top_result = df.iloc[0]
        print(f"\nSecret string: {top_result['secret_string']}")
        print(f"Most frequent outcome: {top_result['measured_string']}")
        print(f"Success rate: {top_result['success_probability']:.1%}")
        print(f"Correct: {top_result['is_correct']}")

        # Show noise effects
        if len(df) > 1:
            print(f"\nNoise effects: {len(df)} different outcomes measured")
            print("All measurement outcomes:")
            for _, row in df.iterrows():
                print(
                    f"  {row['outcome']}: {row['counts']} shots ({row['probability']:.1%})"
                )


if __name__ == "__main__":
    main()
