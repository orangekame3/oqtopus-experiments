#!/usr/bin/env python3
"""
Ramsey CLI - Simplified Ramsey experiment CLI
"""

import typer

from ..backends import LocalBackend, OqtopusBackend
from ..experiments import Ramsey

app = typer.Typer(name="ramsey", help="Ramsey oscillation experiment")


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
    delay_points: int = typer.Option(20, help="Number of delay points"),
    max_delay: float = typer.Option(50000.0, help="Maximum delay in ns"),
    detuning: float = typer.Option(0.0, help="Detuning frequency in MHz"),
    plot: bool = typer.Option(True, help="Generate plots"),
    save: bool = typer.Option(True, help="Save results"),
):
    """Run Ramsey oscillation experiment"""
    try:
        # Create backend
        backend_obj = create_backend(backend)

        # Create and run experiment
        experiment = Ramsey()

        if parallel:
            result = experiment.run_parallel(
                backend=backend_obj,
                shots=shots,
                delay_points=delay_points,
                max_delay=max_delay,
                detuning=detuning,
            )
        else:
            result = experiment.run(
                backend=backend_obj,
                shots=shots,
                delay_points=delay_points,
                max_delay=max_delay,
                detuning=detuning,
            )

        # Analyze results
        analysis = result.analyze(plot=plot, save=save)

        # Extract results
        if "ramsey_estimates" in analysis:
            for device, ramsey_data in analysis["ramsey_estimates"].items():
                if "frequency_mhz" in ramsey_data and "t2_star_ns" in ramsey_data:
                    freq = f"{ramsey_data['frequency_mhz']:.3f}"
                    t2_star_ns = f"{ramsey_data['t2_star_ns']:.1f}"
                    t2_star_us = f"{ramsey_data['t2_star_us']:.2f}"
                    print("✅ Ramsey experiment completed")
                    print(f"   Device: {device}")
                    print(f"   Frequency: {freq} MHz")
                    print(f"   T₂* time: {t2_star_ns} ns ({t2_star_us} μs)")
                    break
        else:
            print("✅ Ramsey experiment completed. Results saved to analysis.")

    except Exception as e:
        print(f"❌ Ramsey experiment failed: {e}")


def main():
    """Main entry point for Ramsey CLI"""
    app()


if __name__ == "__main__":
    main()
