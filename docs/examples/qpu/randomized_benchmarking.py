#!/usr/bin/env python3
"""
Randomized Benchmarking experiment with QPU Backend (Real Hardware)
Standard RB experiment on real quantum hardware
"""

from oqtopus_experiments.backends import OqtopusBackend
from oqtopus_experiments.experiments import RandomizedBenchmarking


def main():
    # Initialize OQTOPUS backend for real hardware
    backend = OqtopusBackend(device="anemone")

    # Create Randomized Benchmarking experiment
    rb = RandomizedBenchmarking(
        experiment_name="rb_qpu_demo",
        physical_qubit=0,
        max_sequence_length=100,  # Maximum number of Clifford gates
        num_lengths=10,  # Number of different sequence lengths
        num_samples=50,  # Number of random sequences per length
        rb_type="standard",  # Standard RB (not interleaved)
    )

    print(f"Experiment: {rb.experiment_name}")
    print(f"Target qubit: {rb.physical_qubit}")
    print(f"Sequence lengths: {rb.sequence_lengths}")
    print(f"Total circuits: {len(rb.sequence_lengths) * rb.num_samples}")

    # Run the experiment on real hardware
    print("\nRunning Randomized Benchmarking on QPU...")
    result = rb.run(backend=backend, shots=1024)

    # Analyze results and create plots
    print("\nAnalyzing results...")
    df = result.analyze(plot=True, save_data=True, save_image=True)

    print("\nResults summary:")
    print(df)

    # Print key metrics
    print(f"\nRandomized Benchmarking Results for Qubit {rb.physical_qubit}:")
    print("=" * 50)

    if "fitted_survival_probability" in df.columns:
        print("✅ Exponential decay fitting completed successfully!")
        print("Detailed results with fitting parameters have been saved.")
        print(f"Sequence lengths tested: {sorted(df['sequence_length'].unique())}")
        print(
            f"Mean survival probabilities: {df['mean_survival_probability'].tolist()}"
        )
        print("Check the generated plot for visual analysis of the decay curve.")
    else:
        print("⚠️  Fitting failed - showing raw data only")
        if "mean_survival_probability" in df.columns:
            for _, row in df.iterrows():
                print(
                    f"  Length {row['sequence_length']:2d}: {row['mean_survival_probability']:.6f}"
                )

    print("\nData saved to analysis files")
    print("Plot saved as randomized_benchmarking_decay.png")


if __name__ == "__main__":
    main()
