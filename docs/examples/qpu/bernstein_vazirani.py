#!/usr/bin/env python3
"""
Bernstein-Vazirani Algorithm Example (QPU Backend)

This example demonstrates how to use the BernsteinVazirani experiment
to find a hidden binary string on real quantum hardware using the
OQTOPUS QPU backend.

Note: Running on real hardware may show effects of quantum noise,
resulting in lower success probabilities compared to simulators.
"""

from oqtopus_experiments.backends import OqtopusBackend
from oqtopus_experiments.experiments import BernsteinVazirani


def main():
    """Run Bernstein-Vazirani algorithm on real quantum hardware"""

    # OQTOPUS backend with real quantum processor
    backend = OqtopusBackend(device="urchin")

    # Create BernsteinVazirani experiment with a short secret string
    # (suitable for current hardware limitations)
    experiment = BernsteinVazirani(
        experiment_name="bernstein_vazirani_qpu",
        secret_string="111",  # 3-bit secret string for hardware demo
    )

    # Run experiment with backend
    result = experiment.run(backend=backend, shots=1000, mitigation_info=None)
    print(f"{experiment.circuits()[0].draw()}")
    # Analyze results
    df = result.analyze()
    print("Measurement results:")
    print(df.head(10))  # Show top 10 outcomes

    # Print summary
    if not df.empty:
        top_result = df.iloc[0]
        print(f"\nSecret string: {top_result['secret_string']}")
        print(f"Measured string: {top_result['measured_string']}")
        print(f"Success probability: {top_result['success_probability']:.3f}")
        print(f"Correct: {top_result['is_correct']}")

        # Show noise effects if present
        if len(df) > 1:
            print(f"\nNoise effects detected: {len(df)} different outcomes measured")
            print("Top 3 measurement outcomes:")
            for i in range(min(3, len(df))):
                row = df.iloc[i]
                print(f"  {row['outcome']}: {row['probability']:.3f}")


if __name__ == "__main__":
    main()
