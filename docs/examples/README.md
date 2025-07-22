# Examples

Simple examples for different backends.

## Directory Structure

```
examples/
├── qpu/         # Real quantum hardware examples (OQTOPUS platform)
├── qulacs/      # Noiseless simulation examples  
└── local/       # Noisy simulation examples
```

## QPU (Real Hardware)

Run experiments on real quantum hardware via OQTOPUS platform:

```bash
python docs/examples/qpu/device.py        # Device information
python docs/examples/qpu/rabi.py          # Rabi oscillation
python docs/examples/qpu/t1.py            # T1 measurement
python docs/examples/qpu/t2_echo_parallel.py  # T2 echo measurement
python docs/examples/qpu/chsh.py          # CHSH Bell test
python docs/examples/qpu/ramsey.py        # Ramsey interference
python docs/examples/qpu/parity_oscillation.py  # Parity oscillation
```

### Parallel execution examples:
```bash
python docs/examples/qpu/rabi_parallel.py         # Parallel Rabi
python docs/examples/qpu/chsh_parallel.py         # Parallel CHSH
python docs/examples/qpu/t1_parallel.py           # Parallel T1
python docs/examples/qpu/ramsey_parallel.py       # Parallel Ramsey
```

Requirements: OQTOPUS account and credentials

## Qulacs (Noiseless Simulation)

Fast noiseless quantum simulation:

```bash
python docs/examples/qulacs/rabi.py       # Ideal Rabi oscillation
python docs/examples/qulacs/t1.py         # Ideal T1 measurement  
python docs/examples/qulacs/t2_echo.py    # Ideal T2 echo
python docs/examples/qulacs/chsh.py       # Perfect Bell test
python docs/examples/qulacs/ramsey.py     # Ideal Ramsey interference
python docs/examples/qulacs/parity_oscillation.py  # Ideal parity oscillation
```

### Parallel execution examples:
```bash
python docs/examples/qulacs/rabi_parallel.py      # Parallel Rabi
python docs/examples/qulacs/chsh_parallel.py      # Parallel CHSH
```

## Local (Noisy Simulation)

Realistic noisy quantum simulation using Qiskit Aer:

```bash
python docs/examples/local/rabi.py        # Noisy Rabi oscillation
python docs/examples/local/t1.py          # Noisy T1 measurement
python docs/examples/local/t2_echo.py     # Noisy T2 echo
python docs/examples/local/chsh.py        # Noisy Bell test
python docs/examples/local/ramsey.py      # Noisy Ramsey interference
python docs/examples/local/parity_oscillation.py  # Noisy parity oscillation
```

### Phase scan experiments:
```bash
python docs/examples/local/chsh_phase_scan.py     # CHSH phase scan
```

### Parallel execution examples:
```bash
python docs/examples/local/rabi_parallel.py       # Parallel Rabi
```

## Usage Pattern

All examples follow the same pattern:

```python
from oqtopus_experiments.backends import LocalBackend, OqtopusBackend
from oqtopus_experiments.experiments import Rabi

# Choose backend
backend = OqtopusBackend()                # Real hardware (OQTOPUS)
# backend = LocalBackend(device="qulacs") # Noiseless simulation
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

## Available Experiments

- **Rabi**: Rabi oscillation measurements for qubit calibration
- **T1**: Relaxation time measurements  
- **T2 Echo**: Coherence time measurements using echo sequences
- **CHSH**: Bell inequality tests for quantum entanglement
- **Ramsey**: Ramsey interference for precise frequency measurements
- **Parity Oscillation**: GHZ state decoherence studies

Each experiment supports:
- Multiple backends (QPU, Qulacs, Local/noisy)
- Parallel execution for faster data collection
- Automatic data analysis and visualization
- Results saving in JSON and CSV formats