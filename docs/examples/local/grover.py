#!/usr/bin/env python3
"""
Grover's Quantum Search Algorithm experiment with Local Backend (Noisy Simulator)
Demonstration of quantum search with quadratic speedup
"""

from oqtopus_experiments.backends import LocalBackend
from oqtopus_experiments.experiments import Grover


def main():
    # Initialize local backend for noisy simulation
    backend = LocalBackend(device="noisy")
    print(f"Backend: {backend.device_name} (Local simulator with built-in noise model)")
    print("Note: Noise may reduce the success probability of Grover's algorithm")

    # Create Grover's algorithm experiment
    grover = Grover(
        experiment_name="grover_local_demo",
        n_qubits=3,  # 8-state search space
        marked_states=[2, 5],  # Search for states |010⟩ and |101⟩
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
        print(f"  State {state} = |{binary}⟩")

    # Run the experiment
    print("\nRunning Grover's algorithm experiment...")
    print("LocalBackend provides realistic noise for quantum search demonstration")
    result = grover.run(backend=backend, shots=2000)

    # Analyze results with plotting
    print("\nAnalyzing results...")
    df = result.analyze(plot=True, save_data=True, save_image=True)

    print("\nResults summary:")
    print(df)

    # Print detailed analysis if successful
    if not df.empty and "probability" in df.columns:
        print("\n✅ Grover's Algorithm Analysis Results:")
        print("=" * 50)

        # Show measurement probabilities for all states
        print("Measurement Distribution:")
        for _, row in df.iterrows():
            state = int(row["state"])
            binary = row["state_binary"]
            prob = row["probability"]
            is_marked = row["is_marked"]
            marker = "🎯" if is_marked else "  "
            print(
                f"  {marker} State {state} |{binary}⟩: {prob:.4f} ({prob * 100:.2f}%)"
            )

        # Calculate success metrics
        marked_probs = df[df["is_marked"]]["probability"]
        unmarked_probs = df[~df["is_marked"]]["probability"]

        success_probability = marked_probs.sum()
        failure_probability = unmarked_probs.sum()

        print("\n📊 Performance Metrics:")
        print(
            f"   Success probability: {success_probability:.4f} ({success_probability * 100:.2f}%)"
        )
        print(
            f"   Failure probability: {failure_probability:.4f} ({failure_probability * 100:.2f}%)"
        )

        # Compare with theoretical expectation
        # For perfect Grover with optimal iterations
        import math

        num_marked = len(grover.marked_states)
        theta = math.asin(math.sqrt(num_marked / grover.search_space_size))
        theoretical_success = math.sin((2 * grover.num_iterations + 1) * theta) ** 2

        print(
            f"   Theoretical success: {theoretical_success:.4f} ({theoretical_success * 100:.2f}%)"
        )
        print(
            f"   Error (noise impact): {abs(theoretical_success - success_probability):.4f}"
        )

        # Speedup analysis
        classical_probability = num_marked / grover.search_space_size
        print("\n🚀 Quantum Advantage:")
        print(
            f"   Classical random search: {classical_probability:.4f} ({classical_probability * 100:.2f}%)"
        )
        print(
            f"   Grover's algorithm: {success_probability:.4f} ({success_probability * 100:.2f}%)"
        )

        if success_probability > classical_probability:
            speedup = success_probability / classical_probability
            print(f"   Speedup factor: {speedup:.2f}x")
            print("   ✅ Quantum advantage demonstrated!")
        else:
            print("   ⚠️  Noise reduced quantum advantage")
            print("   (This is expected with NISQ device noise)")

        print(
            "\nCheck the generated plot for visual analysis of the measurement distribution."
        )
        print(
            "The plot shows marked vs unmarked states and success probability comparison."
        )
    else:
        print("\n⚠️  Analysis failed or returned empty results")

    print("\nExperiment completed. Total shots: 2000")
    print("✅ Grover's algorithm implementation successfully demonstrated!")

    # Additional recommendations
    print("\n💡 Try these variations:")
    print("   • Change n_qubits (2-4 for clear visualization)")
    print("   • Modify marked_states or use 'random'")
    print("   • Compare different num_iterations values")
    print("   • Test with noiseless backend: LocalBackend(device='noiseless')")


if __name__ == "__main__":
    main()
