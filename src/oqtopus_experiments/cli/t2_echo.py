#!/usr/bin/env python3
"""
T2 Echo CLI - Simplified T2 Echo experiment CLI
"""

import typer

from ..backends import LocalBackend, OqtopusBackend
from ..experiments import T2Echo

app = typer.Typer(name="t2-echo", help="T2 Echo experiment")


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
    max_delay: float = typer.Option(100000.0, help="Maximum delay in ns"),
    echo_type: str = typer.Option("hahn", help="Echo type: 'hahn' or 'cpmg'"),
    num_echoes: int = typer.Option(1, help="Number of echo pulses (for CPMG)"),
    plot: bool = typer.Option(True, help="Generate plots"),
    save: bool = typer.Option(True, help="Save results"),
):
    """Run T2 Echo experiment"""
    try:
        # Create backend
        backend_obj = create_backend(backend)

        # Create and run experiment
        experiment = T2Echo()

        if parallel:
            result = experiment.run_parallel(
                backend=backend_obj,
                shots=shots,
                delay_points=delay_points,
                max_delay=max_delay,
                echo_type=echo_type,
                num_echoes=num_echoes,
            )
        else:
            result = experiment.run(
                backend=backend_obj,
                shots=shots,
                delay_points=delay_points,
                max_delay=max_delay,
                echo_type=echo_type,
                num_echoes=num_echoes,
            )

        # Analyze results
        analysis = result.analyze(plot=plot, save=save)

        # Extract results
        if "t2_echo_estimates" in analysis:
            for device, t2_data in analysis["t2_echo_estimates"].items():
                if "t2_echo_ns" in t2_data:
                    t2_echo_ns = f"{t2_data['t2_echo_ns']:.1f}"
                    t2_echo_us = f"{t2_data['t2_echo_us']:.2f}"
                    echo_label = f"T₂({echo_type.upper()})"
                    if echo_type.lower() == "cpmg":
                        echo_label += f"(n={num_echoes})"
                    print("✅ T2 Echo experiment completed")
                    print(f"   Device: {device}")
                    print(f"   {echo_label}: {t2_echo_ns} ns ({t2_echo_us} μs)")
                    break
        else:
            print("✅ T2 Echo experiment completed. Results saved to analysis.")

    except Exception as e:
        print(f"❌ T2 Echo experiment failed: {e}")


def main():
    """Main entry point for T2 Echo CLI"""
    app()


if __name__ == "__main__":
    main()
