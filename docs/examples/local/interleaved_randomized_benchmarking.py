#!/usr/bin/env python3
"""
Interleaved Randomized Benchmarking Example - Using Dedicated Class
Demonstrates the new InterleavedRandomizedBenchmarking class for gate-specific error characterization
"""

from oqtopus_experiments.backends import LocalBackend
from oqtopus_experiments.experiments import InterleavedRandomizedBenchmarking


def main():
    # Initialize local backend for noisy simulation
    backend = LocalBackend(device="noisy")
    print(f"Backend: {backend.device_name} (with enhanced noise for visible decay)")

    # Create interleaved RB experiment using the dedicated class
    interleaved_rb = InterleavedRandomizedBenchmarking(
        experiment_name="x_gate_characterization",
        physical_qubit=0,
        max_sequence_length=128,  # Longer sequences to see decay
        num_lengths=8,  # More data points
        num_samples=30,  # Better statistics
        interleaved_gate="x",  # Target X gate for characterization
    )

    print("\nInterleaved Randomized Benchmarking with Dedicated Class")
    print("=" * 65)
    print("This example uses the InterleavedRandomizedBenchmarking class to")
    print("automatically run both standard and interleaved RB experiments,")
    print("compare results, and calculate gate-specific fidelity.")
    print("Using LocalBackend with ~2% depolarizing error per gate\n")

    # Run both experiments
    print("Running both Standard and Interleaved RB experiments...")
    interleaved_result = interleaved_rb.run(backend=backend, shots=1000)

    # Analyze and create combined visualization
    interleaved_df = interleaved_result.analyze(
        plot=True, save_data=True, save_image=True
    )

    print("\n" + "=" * 65)
    print("SUMMARY:")
    print(f"Gate: {interleaved_rb.interleaved_gate.upper()}")
    print("Interleaved RB completed successfully!")

    # Demonstrate easy access to individual experiment data
    print("\nData Access:")
    print(
        f"Interleaved RB final survival: {interleaved_df.iloc[-1]['mean_survival_probability']:.4f}"
    )
    if interleaved_rb.standard_df is not None:
        print(
            f"Standard RB final survival: {interleaved_rb.standard_df.iloc[-1]['mean_survival_probability']:.4f}"
        )
    print(f"Sequence lengths tested: {interleaved_rb.sequence_lengths}")

    print("\n✅ Interleaved RB experiment completed successfully!")
    print("📊 Combined plot and analysis data saved automatically")


if __name__ == "__main__":
    main()
