#!/usr/bin/env python3
"""
T1 CLI - Simplified T1 experiment CLI
"""

import typer

from ..backends import LocalBackend, OqtopusBackend
from ..experiments import T1

app = typer.Typer(name="t1", help="T1 decay experiment")


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
    plot: bool = typer.Option(True, help="Generate plots"),
    save: bool = typer.Option(True, help="Save results"),
):
    """Run T1 decay experiment"""
    try:
        # Create backend
        backend_obj = create_backend(backend)

        # Create and run experiment
        experiment = T1()

        if parallel:
            result = experiment.run_parallel(
                backend=backend_obj,
                shots=shots,
                delay_points=delay_points,
                max_delay=max_delay,
            )
        else:
            result = experiment.run(
                backend=backend_obj,
                shots=shots,
                delay_points=delay_points,
                max_delay=max_delay,
            )

        # Analyze results
        analysis = result.analyze(plot=plot, save=save)

        # Extract results
        if "t1_estimates" in analysis:
            for device, t1_data in analysis["t1_estimates"].items():
                if "t1_ns" in t1_data:
                    t1_ns = f"{t1_data['t1_ns']:.1f}"
                    t1_us = f"{t1_data['t1_us']:.2f}"
                    print("✅ T1 experiment completed")
                    print(f"   Device: {device}")
                    print(f"   T₁ time: {t1_ns} ns ({t1_us} μs)")
                    break
        else:
            print("✅ T1 experiment completed. Results saved to analysis.")

    except Exception as e:
        print(f"❌ T1 experiment failed: {e}")


def main():
    """Main entry point for T1 CLI"""
    app()


if __name__ == "__main__":
    main()
