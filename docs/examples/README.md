# Examples

Simple examples for different backends.

## Directory Structure

```
examples/
├── anemone/     # Real Anemone device examples
├── qulacs/      # Noiseless simulation examples  
└── local/       # Noisy simulation examples
```

## Anemone (Real Hardware)

Run experiments on real Anemone quantum device:

```bash
python docs/examples/anemone/device.py    # Device information
python docs/examples/anemone/rabi.py      # Rabi oscillation
python docs/examples/anemone/t1.py        # T1 measurement
python docs/examples/anemone/chsh.py      # CHSH Bell test
```

Requirements: OQTOPUS account and credentials

## Qulacs (Noiseless Simulation)

Fast noiseless quantum simulation:

```bash
python docs/examples/qulacs/rabi.py       # Ideal Rabi oscillation
python docs/examples/qulacs/t1.py         # Ideal T1 measurement  
python docs/examples/qulacs/chsh.py       # Perfect Bell test
```

## Local (Noisy Simulation)

Realistic noisy quantum simulation:

```bash
python docs/examples/local/rabi.py        # Noisy Rabi oscillation
python docs/examples/local/t1.py          # Noisy T1 measurement
python docs/examples/local/chsh.py        # Noisy Bell test
python docs/examples/local/compare.py     # Compare noisy vs clean
```

## Usage Pattern

All examples follow the same pattern:

```python
from oqtopus_experiments.backends import LocalBackend, OqtopusBackend
from oqtopus_experiments.experiments import Rabi

# Choose backend
backend = OqtopusBackend(device="anemone")  # Real hardware
# backend = LocalBackend(noise=False)       # Qulacs-like
# backend = LocalBackend(noise=True)        # Noisy simulation

# Create experiment
rabi = Rabi()
circuits = rabi.circuits(qubits=[0], amplitude_points=20, max_amplitude=2.0)

# Run and analyze
result = rabi.run(backend=backend, shots=1000)
result.analyze()
```