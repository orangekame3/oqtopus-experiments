#!/usr/bin/env python3
"""
Grover's Quantum Search Algorithm experiment with Qulacs Backend (Noiseless Simulator)
Demonstration of ideal quantum search performance
"""

from oqtopus_experiments.backends import OqtopusBackend
from oqtopus_experiments.experiments import Grover


def main():
    # Initialize OQTOPUS backend with Qulacs simulator
    backend = OqtopusBackend(device="qulacs")
    print("Backend: Qulacs (Noiseless quantum simulator)")
    print("Note: Perfect noiseless simulation for ideal Grover performance")

    # Create Grover's algorithm experiment
    grover = Grover(
        experiment_name="grover_qulacs_demo",
        n_qubits=4,  # 16-state search space for more interesting demo
        marked_states=[3, 7, 12],  # Search for three specific states
        num_iterations=None,  # Use optimal number of iterations
    )

    print(f"Experiment: {grover.experiment_name}")
    print(f"Search space: {grover.search_space_size} states ({grover.n_qubits} qubits)")
    print(
        f"Marked states: {grover.marked_states} ({len(grover.marked_states)} targets)"
    )
    print(f"Optimal iterations: {grover.optimal_iterations}")
    print(f"Actual iterations: {grover.num_iterations}")

    # Show binary representations of marked states
    print("\nMarked states (binary):")
    for state in grover.marked_states:
        binary = format(state, f"0{grover.n_qubits}b")
        print(f"  State {state:2d} = |{binary}⟩")

    # Theoretical analysis
    import math

    num_marked = len(grover.marked_states)
    theta = math.asin(math.sqrt(num_marked / grover.search_space_size))
    theoretical_success = math.sin((2 * grover.num_iterations + 1) * theta) ** 2

    print("\n📐 Theoretical Prediction:")
    print(
        f"   Success probability: {theoretical_success:.6f} ({theoretical_success * 100:.4f}%)"
    )
    print(
        f"   Classical random: {num_marked / grover.search_space_size:.6f} ({num_marked / grover.search_space_size * 100:.4f}%)"
    )
    print(
        f"   Theoretical speedup: {theoretical_success / (num_marked / grover.search_space_size):.2f}x"
    )

    # Run the experiment on noiseless simulator
    print("\nRunning Grover's algorithm on Qulacs simulator...")
    result = grover.run(backend=backend, shots=4000)

    # Analyze results with plotting
    print("\nAnalyzing results...")
    df = result.analyze(plot=True, save_data=True, save_image=True)

    print("\nResults summary:")
    print(df)

    # Print detailed analysis
    if not df.empty and "probability" in df.columns:
        print("\n✅ Grover's Algorithm Analysis Results (Noiseless):")
        print("=" * 60)

        # Show measurement probabilities for all states
        print("Measurement Distribution:")
        marked_states_data = df[df["is_marked"]].sort_values("state")
        unmarked_states_data = df[~df["is_marked"]].sort_values("state")

        print("\n🎯 Marked States (should have high probability):")
        for _, row in marked_states_data.iterrows():
            state = int(row["state"])
            binary = row["state_binary"]
            prob = row["probability"]
            theoretical = row.get("theoretical_probability", 0)
            print(
                f"   State {state:2d} |{binary}⟩: {prob:.6f} (theory: {theoretical:.6f})"
            )

        print("\n   Unmarked States (should have low probability):")
        # Show only first few and last few unmarked states to avoid clutter
        if len(unmarked_states_data) > 8:
            for _, row in unmarked_states_data.head(3).iterrows():
                state = int(row["state"])
                binary = row["state_binary"]
                prob = row["probability"]
                print(f"   State {state:2d} |{binary}⟩: {prob:.6f}")
            print("   ...")
            for _, row in unmarked_states_data.tail(3).iterrows():
                state = int(row["state"])
                binary = row["state_binary"]
                prob = row["probability"]
                print(f"   State {state:2d} |{binary}⟩: {prob:.6f}")
        else:
            for _, row in unmarked_states_data.iterrows():
                state = int(row["state"])
                binary = row["state_binary"]
                prob = row["probability"]
                print(f"   State {state:2d} |{binary}⟩: {prob:.6f}")

        # Calculate performance metrics
        measured_success = df[df["is_marked"]]["probability"].sum()
        classical_probability = num_marked / grover.search_space_size

        print("\n📊 Performance Metrics:")
        print(
            f"   Measured success: {measured_success:.6f} ({measured_success * 100:.4f}%)"
        )
        print(
            f"   Theoretical success: {theoretical_success:.6f} ({theoretical_success * 100:.4f}%)"
        )
        print(
            f"   Error (should be ~0): {abs(theoretical_success - measured_success):.8f}"
        )

        # Quantum advantage analysis
        print("\n🚀 Quantum Advantage (Noiseless):")
        print(
            f"   Classical random search: {classical_probability:.6f} ({classical_probability * 100:.4f}%)"
        )
        print(
            f"   Grover's algorithm: {measured_success:.6f} ({measured_success * 100:.4f}%)"
        )
        speedup = measured_success / classical_probability
        print(f"   Speedup factor: {speedup:.2f}x")

        if abs(theoretical_success - measured_success) < 0.01:
            print("   ✅ Perfect quantum performance achieved!")
        else:
            print("   ⚠️  Deviation from theory detected")

        # Statistical analysis
        total_shots = df["count"].sum()
        marked_counts = df[df["is_marked"]]["count"].sum()
        statistical_error = math.sqrt(marked_counts) / total_shots

        print("\n📈 Statistical Analysis:")
        print(f"   Total shots: {total_shots}")
        print(f"   Marked hits: {marked_counts}")
        print(f"   Statistical error: ±{statistical_error:.6f}")

        if abs(theoretical_success - measured_success) <= 3 * statistical_error:
            print("   ✅ Result within statistical uncertainty")
        else:
            print("   ⚠️  Result outside statistical uncertainty")

        print("\nCheck the generated plot for visual analysis.")
        print("In noiseless simulation, the distribution should closely match theory.")
    else:
        print("\n⚠️  Analysis failed or returned empty results")

    print("\nExperiment completed. Total shots: 4000")
    print("✅ Grover's algorithm implementation successfully demonstrated!")

    # Additional insights for noiseless simulation
    print("\n💡 Noiseless Simulation Insights:")
    print("   • Perfect coherence allows exact theoretical predictions")
    print("   • Try different num_iterations to see amplitude oscillations")
    print("   • Compare with noisy LocalBackend to see decoherence effects")
    print("   • Larger n_qubits showcase greater quantum advantage")


if __name__ == "__main__":
    main()
