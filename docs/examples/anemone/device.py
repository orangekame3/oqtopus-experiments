#!/usr/bin/env python3
"""
Anemone device information example
"""

from oqtopus_experiments.backends import OqtopusBackend


def main():
    backend = OqtopusBackend(device="anemone", timeout_seconds=60)

    print(f"Device: {backend.device_name}")

    # Show device information
    backend.show_device(show_qubits=True, show_couplings=True)

    # Get best qubits
    best_qubits = backend.get_best_qubits(5)
    if best_qubits is not None:
        print("\nBest qubits:")
        print(best_qubits)

    # Get device stats
    stats = backend.get_device_stats()
    if stats:
        print("\nDevice stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    # Save device info
    try:
        filename = backend.save_device_info("anemone_device.json")
        print(f"\nDevice info saved to: {filename}")
    except Exception as e:
        print(f"Could not save device info: {e}")


if __name__ == "__main__":
    main()
