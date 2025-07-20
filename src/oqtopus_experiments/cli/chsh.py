#!/usr/bin/env python3
"""
CHSH CLI - Simplified CHSH experiment CLI
"""

import typer

from ..backends import LocalBackend, OqtopusBackend
from ..experiments import CHSH

app = typer.Typer(name="chsh", help="CHSH Bell inequality experiment")


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
    phase_points: int = typer.Option(20, help="Number of phase points"),
    theta_a: float = typer.Option(0.0, help="Alice measurement angle"),
    theta_b: float = typer.Option(0.785398, help="Bob measurement angle (π/4)"),
    plot: bool = typer.Option(True, help="Generate plots"),
    save: bool = typer.Option(True, help="Save results"),
):
    """Run CHSH Bell inequality experiment"""
    try:
        # Create backend
        backend_obj = create_backend(backend)

        # Create and run experiment
        experiment = CHSH()

        if parallel:
            result = experiment.run_parallel(
                backend=backend_obj,
                shots=shots,
                phase_points=phase_points,
                theta_a=theta_a,
                theta_b=theta_b,
            )
        else:
            result = experiment.run(
                backend=backend_obj,
                shots=shots,
                phase_points=phase_points,
                theta_a=theta_a,
                theta_b=theta_b,
            )

        # Analyze results
        analysis = result.analyze(plot=plot, save=save)

        # Extract max S value from analysis
        if "max_violation" in analysis:
            for device, violation_data in analysis["max_violation"].items():
                if "max_S" in violation_data:
                    max_s = f"{violation_data['max_S']:.3f}"
                    violation = "✓" if violation_data.get("violation", False) else "✗"
                    print("✅ CHSH experiment completed")
                    print(f"   Device: {device}")
                    print(f"   Max S-value: {max_s}")
                    print(f"   Bell inequality violation: {violation}")
                    break
        else:
            print("✅ CHSH experiment completed. Results saved to analysis.")

    except Exception as e:
        print(f"❌ CHSH experiment failed: {e}")


def main():
    """Main entry point for CHSH CLI"""
    app()


if __name__ == "__main__":
    main()
