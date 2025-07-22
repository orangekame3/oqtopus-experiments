#!/usr/bin/env python3
"""
Rabi CLI - Simplified Rabi experiment CLI
"""

import typer

from ..backends import LocalBackend, OqtopusBackend
from ..experiments import Rabi

app = typer.Typer(name="rabi", help="Rabi oscillation experiment")


def create_backend(backend_name: str):
    """Create backend instance from name"""
    if backend_name.lower() == "local":
        return LocalBackend()
    elif backend_name.lower() == "oqtopus":
        return OqtopusBackend()
    else:
        raise ValueError(f"Unknown backend: {backend_name}. Use 'local' or 'oqtopus'")


@app.command()
def run(
    shots: int = typer.Option(1000, help="Number of shots"),
    backend: str = typer.Option("local", help="Backend: 'local' or 'oqtopus'"),
    parallel: bool = typer.Option(False, help="Run in parallel"),
    amplitude_points: int = typer.Option(20, help="Number of amplitude points"),
    max_amplitude: float = typer.Option(1.0, help="Maximum pulse amplitude"),
    plot: bool = typer.Option(True, help="Generate plots"),
    save: bool = typer.Option(True, help="Save results"),
):
    """Run Rabi oscillation experiment"""
    try:
        # Create backend
        backend_obj = create_backend(backend)

        # Create and run experiment
        experiment = Rabi()

        if parallel:
            result = experiment.run_parallel(
                backend=backend_obj,
                shots=shots,
                amplitude_points=amplitude_points,
                max_amplitude=max_amplitude,
            )
        else:
            result = experiment.run(
                backend=backend_obj,
                shots=shots,
                amplitude_points=amplitude_points,
                max_amplitude=max_amplitude,
            )

        # Analyze results
        analysis = result.analyze(plot=plot, save=save)

        # Extract results
        if "rabi_estimates" in analysis:
            for device, rabi_data in analysis["rabi_estimates"].items():
                if "pi_amplitude" in rabi_data:
                    pi_amplitude = f"{rabi_data['pi_amplitude']:.3f}"
                    frequency = rabi_data.get("frequency_mhz", "N/A")
                    print("✅ Rabi experiment completed")
                    print(f"   Device: {device}")
                    print(f"   π pulse amplitude: {pi_amplitude}")
                    print(f"   Rabi frequency: {frequency} MHz")
                    break
        else:
            print("✅ Rabi experiment completed. Results saved to analysis.")

    except Exception as e:
        print(f"❌ Rabi experiment failed: {e}")


def main():
    """Main entry point for Rabi CLI"""
    app()


if __name__ == "__main__":
    main()
