#!/usr/bin/env python3
"""
Anemone device information example
"""


from oqtopus_experiments.backends import OqtopusBackend


def main():
    backend = OqtopusBackend(device="anemone")

    print(f"Device: {backend.device_name}")

    # Show device information
    backend.show_device()

    # Create and show topology plot
    print("\nCreating device topology plot...")
    try:
        if hasattr(backend, "plot_device_layout"):
            # Use the backend's plotting method (opens automatically in browser)
            backend.plot_device_layout(color_by="fidelity")
            print("Interactive plot opened in browser")
        else:
            print("Plot functionality not available")
    except Exception as e:
        print(f"Plot error: {e}")


if __name__ == "__main__":
    main()
