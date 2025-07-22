# OQTOPUS Experiments

A modular quantum computing experiment library for the OQTOPUS platform.

## Overview

OQTOPUS Experiments provides a comprehensive framework for running quantum experiments on both real hardware and simulators. The library supports various quantum algorithms and protocols, making it easy to conduct research and development in quantum computing.

## Key Features

- **Multiple Backends**: Support for real quantum hardware via OQTOPUS platform, noiseless simulation with Qulacs, and noisy simulation with Qiskit Aer
- **Rich Experiment Suite**: Includes Rabi oscillations, T1/T2 measurements, Bell tests, Ramsey interference, and more
- **Parallel Execution**: Efficient parallel job submission and data collection
- **Automatic Analysis**: Built-in data analysis with visualization and curve fitting
- **Type Safety**: Comprehensive type annotations for better development experience

## Quick Start

```python
from oqtopus_experiments.backends import LocalBackend, OqtopusBackend
from oqtopus_experiments.experiments import Rabi

# Choose your backend
backend = OqtopusBackend()  # Real hardware
# backend = LocalBackend(device="qulacs")  # Noiseless simulation
# backend = LocalBackend(device="noisy")   # Noisy simulation

# Create and run experiment
exp = Rabi(
    experiment_name="rabi_experiment",
    physical_qubit=0,
    amplitude_points=20,
    max_amplitude=2.0,
)

result = exp.run(backend=backend, shots=1000)
df = result.analyze(plot=True, save_data=True)
```

## Available Experiments

| Experiment | Purpose | Description |
|-----------|---------|-------------|
| **Rabi** | Qubit Calibration | Measure Rabi oscillations for π-pulse calibration |
| **T1** | Relaxation Time | Characterize qubit energy decay time |
| **T2 Echo** | Coherence Time | Measure dephasing time with echo sequences |
| **CHSH** | Entanglement | Bell inequality tests for quantum correlations |
| **Ramsey** | Frequency Measurement | Precision qubit frequency characterization |
| **Parity Oscillation** | Decoherence Studies | Multi-qubit GHZ state evolution |

## Next Steps

- [Installation Guide](getting-started/installation.md) - Set up your environment
- [Quick Start Tutorial](getting-started/quickstart.md) - Run your first experiment
- [Experiment Documentation](experiments/index.md) - Detailed experiment guides
- [API Reference](reference/SUMMARY.md) - Complete API documentation