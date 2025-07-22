# OQTOPUS Experiments

A modular quantum computing experiment library for the OQTOPUS platform.

## Installation

```bash
pip install git+https://github.com/orangekame3/oqtopus-experiments.git
```

## Usage

### Quick Start

```python
from oqtopus_experiments.backends import LocalBackend, OqtopusBackend
from oqtopus_experiments.experiments import Rabi

# Choose backend
backend = OqtopusBackend(device="anemone") # Real hardware (OQTOPUS)
# backend = OqtopusBackend(device="qulacs") # Noiseless simulation
# backend = LocalBackend(device="noisy")  # Noisy simulation

# Create experiment
exp = Rabi(
    experiment_name="rabi_experiment",
    physical_qubit=0,
    amplitude_points=20,
    max_amplitude=2.0,
)

# Run and analyze
result = exp.run(backend=backend, shots=1000)
df = result.analyze(plot=True, save_data=True, save_image=True)
print(df.head())
```

### Available Experiments

- **Rabi**: Rabi oscillation measurements for qubit calibration
- **T1**: Relaxation time measurements
- **T2 Echo**: Coherence time measurements using echo sequences
- **CHSH**: Bell inequality tests for quantum entanglement
- **Ramsey**: Ramsey interference for precise frequency measurements
- **Parity Oscillation**: GHZ state decoherence studies

### Backends

- **OqtopusBackend()**: Run on real quantum hardware via OQTOPUS platform
- **OqtopusBackend(device="qulacs")**: Fast noiseless simulation
- **LocalBackend(device="noisy")**: Realistic noisy simulation using Qiskit Aer

## Examples

See `docs/examples/` for comprehensive examples:

```bash
# Real hardware
python docs/examples/qpu/rabi.py

# Noiseless simulation (OQTOPUS with qulacs)
python docs/examples/qulacs/rabi.py

# Noisy simulation
python docs/examples/local/rabi.py
```

Each experiment supports parallel execution, automatic data analysis, and visualization.

## Requirements

- Python 3.12+
- OQTOPUS account and credentials (for real hardware)
- Quantum simulators: Qulacs, Qiskit

## License

[Apache License 2.0](LICENSE)
