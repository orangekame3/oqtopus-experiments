#!/usr/bin/env python3
"""
Grover's Quantum Search Algorithm experiment with QPU Backend (Real Hardware)
Demonstration of quantum search on real superconducting quantum processors
"""

from oqtopus_experiments.backends import OqtopusBackend
from oqtopus_experiments.experiments import Grover


def main():
    # Initialize OQTOPUS backend for real hardware
    backend = OqtopusBackend(device="anemone")
    print(f"Backend: {backend.device_name} (Real quantum hardware)")
    print("Note: Real hardware noise and decoherence will affect search performance")

    # Create Grover's algorithm experiment
    # For real hardware, use smaller qubit counts to minimize noise impact
    grover = Grover(
        experiment_name="grover_qpu_demo",
        n_qubits=3,  # 8-state search space (good for NISQ devices)
        marked_states=[2, 6],  # Search for states |010⟩ and |110⟩
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

    # Theoretical analysis
    import math

    num_marked = len(grover.marked_states)
    theta = math.asin(math.sqrt(num_marked / grover.search_space_size))
    theoretical_success = math.sin((2 * grover.num_iterations + 1) * theta) ** 2
    classical_probability = num_marked / grover.search_space_size

    print("\n📐 Theoretical Prediction (Noiseless):")
    print(
        f"   Success probability: {theoretical_success:.4f} ({theoretical_success * 100:.2f}%)"
    )
    print(
        f"   Classical random: {classical_probability:.4f} ({classical_probability * 100:.2f}%)"
    )
    print(f"   Theoretical speedup: {theoretical_success / classical_probability:.2f}x")

    # Hardware considerations
    print("\n🔧 Hardware Considerations:")
    print(f"   Circuit depth: ~{grover.num_iterations * 20} gates (estimate)")
    print("   Coherence impact: T1/T2 decoherence will reduce success probability")
    print("   Gate errors: Each gate introduces ~0.1-1% error")
    print("   Readout errors: SPAM errors affect final measurement")

    # Run the experiment on real hardware
    print("\nRunning Grover's algorithm on real QPU...")
    print("⏳ Submitting to quantum hardware queue...")
    result = grover.run(backend=backend, shots=4096)  # More shots for better statistics

    # Analyze results with plotting
    print("\nAnalyzing results...")
    df = result.analyze(plot=True, save_data=True, save_image=True)

    print("\nResults summary:")
    print(df)

    # Print detailed analysis
    if not df.empty and "probability" in df.columns:
        print("\n✅ Grover's Algorithm Analysis Results (Real Hardware):")
        print("=" * 65)

        # Show measurement probabilities for key states
        print("Measurement Distribution:")
        marked_data = df[df["is_marked"]].sort_values("state")

        print("\n🎯 Marked States (target for amplification):")
        for _, row in marked_data.iterrows():
            state = int(row["state"])
            binary = row["state_binary"]
            prob = row["probability"]
            count = int(row["count"])
            print(f"   State {state} |{binary}⟩: {prob:.4f} ({count:4d} counts)")

        print("\n   Unmarked States (should be suppressed):")
        unmarked_data = df[~df["is_marked"]].sort_values("state")
        for _, row in unmarked_data.iterrows():
            state = int(row["state"])
            binary = row["state_binary"]
            prob = row["probability"]
            count = int(row["count"])
            print(f"   State {state} |{binary}⟩: {prob:.4f} ({count:4d} counts)")

        # Calculate performance metrics
        measured_success = df[df["is_marked"]]["probability"].sum()

        print("\n📊 Performance Metrics:")
        print(
            f"   Measured success: {measured_success:.4f} ({measured_success * 100:.2f}%)"
        )
        print(
            f"   Theoretical (ideal): {theoretical_success:.4f} ({theoretical_success * 100:.2f}%)"
        )
        print(f"   Hardware degradation: {theoretical_success - measured_success:.4f}")
        print(
            f"   Relative degradation: {(theoretical_success - measured_success) / theoretical_success * 100:.1f}%"
        )

        # Quantum advantage analysis
        print("\n🚀 Quantum Advantage (Real Hardware):")
        print(
            f"   Classical random search: {classical_probability:.4f} ({classical_probability * 100:.2f}%)"
        )
        print(
            f"   Grover's algorithm (real): {measured_success:.4f} ({measured_success * 100:.2f}%)"
        )

        if measured_success > classical_probability:
            speedup = measured_success / classical_probability
            print(f"   Realized speedup: {speedup:.2f}x")
            print("   ✅ Quantum advantage maintained on real hardware!")
        else:
            print("   ⚠️  Quantum advantage lost due to hardware noise")
            print("   (This can happen on NISQ devices)")

        # Error analysis
        total_shots = df["count"].sum()
        marked_counts = df[df["is_marked"]]["count"].sum()
        statistical_error = math.sqrt(marked_counts) / total_shots

        print("\n🔍 Error Analysis:")
        print(f"   Total shots: {total_shots}")
        print(f"   Marked hits: {marked_counts}")
        print(f"   Statistical error: ±{statistical_error:.4f}")

        # Identify likely error sources
        ideal_to_measured_ratio = measured_success / theoretical_success
        print(f"   Fidelity ratio: {ideal_to_measured_ratio:.3f}")

        if ideal_to_measured_ratio > 0.8:
            print("   ✅ High fidelity - excellent hardware performance")
        elif ideal_to_measured_ratio > 0.6:
            print("   ✨ Good fidelity - typical NISQ performance")
        elif ideal_to_measured_ratio > 0.4:
            print("   ⚠️  Moderate fidelity - noticeable decoherence")
        else:
            print("   ❌ Low fidelity - strong noise/decoherence effects")

        print("\nCheck the generated plot for visual analysis.")
        print(
            "Compare marked vs unmarked state distributions to assess algorithm performance."
        )
    else:
        print("\n⚠️  Analysis failed or returned empty results")

    print("\nExperiment completed on real quantum hardware!")
    print("✅ Grover's algorithm successfully demonstrated on QPU!")

    # Additional insights for hardware experiments
    print("\n💡 Hardware Experiment Insights:")
    print("   • Shorter circuits generally perform better on NISQ devices")
    print("   • Try different num_iterations to find optimal performance")
    print("   • Compare with simulator results to quantify hardware noise")
    print("   • Consider error mitigation techniques for improved fidelity")
    print("   • Monitor device calibration data for performance correlation")


if __name__ == "__main__":
    main()
