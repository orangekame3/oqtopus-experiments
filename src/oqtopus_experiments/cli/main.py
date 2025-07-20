#!/usr/bin/env python3
"""
OQTOPUS Experiments CLI - Simplified CLI interface using usage.py style API

This replaces the complex BaseExperimentCLI framework with a simple, direct approach
based on the streamlined usage.py style API.
"""

import typer

from ..backends import LocalBackend, OqtopusBackend
from ..experiments import CHSH, T1, ParityOscillation, Rabi, Ramsey, T2Echo

app = typer.Typer(
    name="oqtopus-experiments",
    help="OQTOPUS Quantum Experiments CLI",
    rich_markup_mode="rich",
)


def create_backend(backend_name: str):
    """Create backend instance from name"""
    if backend_name.lower() == "local":
        return LocalBackend()
    elif backend_name.lower() == "oqtopus":
        return OqtopusBackend()
    else:
        raise ValueError(f"Unknown backend: {backend_name}. Use 'local' or 'oqtopus'")


@app.command()
def chsh(
    shots: int = typer.Option(1000, help="Number of shots"),
    backend: str = typer.Option("local", help="Backend: 'local' or 'oqtopus'"),
    parallel: bool = typer.Option(False, help="Run in parallel"),
    workers: int = typer.Option(4, help="Number of parallel workers"),
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
                workers=workers,
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
        max_s = "N/A"
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


@app.command()
def rabi(
    shots: int = typer.Option(1000, help="Number of shots"),
    backend: str = typer.Option("local", help="Backend: 'local' or 'oqtopus'"),
    parallel: bool = typer.Option(False, help="Run in parallel"),
    workers: int = typer.Option(4, help="Number of parallel workers"),
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
                workers=workers,
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
        pi_amplitude = "N/A"
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


@app.command()
def t1(
    shots: int = typer.Option(1000, help="Number of shots"),
    backend: str = typer.Option("local", help="Backend: 'local' or 'oqtopus'"),
    parallel: bool = typer.Option(False, help="Run in parallel"),
    workers: int = typer.Option(4, help="Number of parallel workers"),
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
                workers=workers,
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


@app.command()
def ramsey(
    shots: int = typer.Option(1000, help="Number of shots"),
    backend: str = typer.Option("local", help="Backend: 'local' or 'oqtopus'"),
    parallel: bool = typer.Option(False, help="Run in parallel"),
    workers: int = typer.Option(4, help="Number of parallel workers"),
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
                workers=workers,
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


@app.command(name="t2-echo")
def t2_echo(
    shots: int = typer.Option(1000, help="Number of shots"),
    backend: str = typer.Option("local", help="Backend: 'local' or 'oqtopus'"),
    parallel: bool = typer.Option(False, help="Run in parallel"),
    workers: int = typer.Option(4, help="Number of parallel workers"),
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
                workers=workers,
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


@app.command(name="parity")
def parity_oscillation(
    shots: int = typer.Option(1000, help="Number of shots"),
    backend: str = typer.Option("local", help="Backend: 'local' or 'oqtopus'"),
    parallel: bool = typer.Option(False, help="Run in parallel"),
    workers: int = typer.Option(4, help="Number of parallel workers"),
    num_qubits: int = typer.Option(2, help="Number of qubits"),
    rotation_points: int = typer.Option(20, help="Number of rotation points"),
    max_rotation: float = typer.Option(6.283185, help="Maximum rotation angle (2π)"),
    plot: bool = typer.Option(True, help="Generate plots"),
    save: bool = typer.Option(True, help="Save results"),
):
    """Run Parity Oscillation experiment"""
    try:
        # Create backend
        backend_obj = create_backend(backend)

        # Create and run experiment
        experiment = ParityOscillation()

        if parallel:
            result = experiment.run_parallel(
                backend=backend_obj,
                shots=shots,
                workers=workers,
                num_qubits=num_qubits,
                rotation_points=rotation_points,
                max_rotation=max_rotation,
            )
        else:
            result = experiment.run(
                backend=backend_obj,
                shots=shots,
                num_qubits=num_qubits,
                rotation_points=rotation_points,
                max_rotation=max_rotation,
            )

        # Analyze results
        analysis = result.analyze(plot=plot, save=save)

        # Extract results
        if "parity_estimates" in analysis:
            for device, parity_data in analysis["parity_estimates"].items():
                if "frequency_hz" in parity_data:
                    freq = f"{parity_data['frequency_hz']:.3f}"
                    amplitude = parity_data.get("amplitude", "N/A")
                    print("✅ Parity Oscillation experiment completed")
                    print(f"   Device: {device}")
                    print(f"   Oscillation frequency: {freq} Hz")
                    print(f"   Amplitude: {amplitude}")
                    break
        else:
            print(
                "✅ Parity Oscillation experiment completed. Results saved to analysis."
            )

    except Exception as e:
        print(f"❌ Parity Oscillation experiment failed: {e}")


def main():
    """Main entry point for CLI"""
    app()


if __name__ == "__main__":
    main()
