#!/usr/bin/env python3
"""
Randomized Benchmarking experiment with Qulacs Backend (Noiseless Simulator)
Standard RB experiment using noiseless simulation
"""

from oqtopus_experiments.backends import OqtopusBackend
from oqtopus_experiments.experiments import RandomizedBenchmarking


def main():
    # Initialize OQTOPUS backend with Qulacs simulator
    backend = OqtopusBackend(device="qulacs")

    # Create Randomized Benchmarking experiment
    rb = RandomizedBenchmarking(
        experiment_name="rb_qulacs_demo",
        physical_qubit=0,
        max_sequence_length=200,  # Can use longer sequences for simulation
        num_lengths=12,           # More data points for smooth curve
        num_samples=30,           # Fewer samples needed for noiseless simulation
        rb_type="standard",       # Standard RB (not interleaved)
    )

    print(f"Experiment: {rb.experiment_name}")
    print(f"Sequence lengths: {rb.sequence_lengths}")
    print(f"Total circuits: {len(rb.sequence_lengths) * rb.num_samples}")

    # Run the experiment on noiseless simulator
    print("\nRunning Randomized Benchmarking on Qulacs simulator...")
    result = rb.run(backend=backend, shots=1000)

    # Analyze results and create plots
    print("\nAnalyzing results...")
    df = result.analyze(plot=True, save_data=True, save_image=True)

    print("\nResults summary:")
    print(df)

    # Print interpretation for noiseless simulation
    print(f"\nRandomized Benchmarking Results (Noiseless Simulation):")
    print("=" * 55)
    print("Note: In a perfect noiseless simulator, the survival probability")
    print("should remain close to 1.0 for all sequence lengths.")
    print("Any decay observed indicates implementation imperfections.")

    if 'fitted_survival_probability' in df.columns:
        print("\n✅ Exponential decay fitting completed successfully!")
        print("Results have been saved with detailed fitting parameters.")
        print(f"Sequence lengths tested: {sorted(df['sequence_length'].unique())}")
        print(f"Mean survival probabilities: {df['mean_survival_probability'].tolist()}")
        print("Check the saved metadata file for detailed error analysis.")
    else:
        print("\n⚠️  Fitting failed - showing raw data only")
        if 'mean_survival_probability' in df.columns:
            for _, row in df.iterrows():
                print(f"  Length {row['sequence_length']:2d}: {row['mean_survival_probability']:.6f}")

    print(f"\nThis demonstrates the RB protocol working correctly.")
    print(f"Compare with noisy simulations or real hardware to see the difference!")


if __name__ == "__main__":
    main()