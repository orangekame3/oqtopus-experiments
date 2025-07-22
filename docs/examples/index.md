# Examples Overview

This section provides comprehensive examples demonstrating how to use OQTOPUS Experiments across different backends and use cases.

## Repository Examples

The `docs/examples/` directory contains ready-to-run Python scripts organized by backend:

```
docs/examples/
├── qpu/         # Real quantum hardware examples
├── qulacs/      # Noiseless simulation examples  
└── local/       # Noisy simulation examples
```

## Quick Navigation

### [Real Hardware (QPU)](qpu.md)
Production-ready experiments for OQTOPUS quantum devices:

- Device characterization and calibration
- Parallel execution for efficiency
- Production-quality data collection

### [Simulation Examples](simulation.md)
Development and testing with local simulators:

- Fast iteration with Qulacs (noiseless)
- Realistic testing with noisy simulation
- Algorithm development and validation

## Example Categories

### Basic Experiments
Single-qubit characterization and calibration:

```bash
# Rabi oscillations
python docs/examples/qpu/rabi.py          # Real hardware
python docs/examples/qulacs/rabi.py       # Ideal simulation
python docs/examples/local/rabi.py        # Noisy simulation

# T1 relaxation time
python docs/examples/qpu/t1.py
python docs/examples/qulacs/t1.py
python docs/examples/local/t1.py

# T2 coherence time
python docs/examples/qpu/t2_echo_parallel.py
python docs/examples/qulacs/t2_echo.py
python docs/examples/local/t2_echo.py
```

### Multi-Qubit Experiments
Entanglement and correlation studies:

```bash
# CHSH Bell inequality tests
python docs/examples/qpu/chsh.py
python docs/examples/qulacs/chsh.py
python docs/examples/local/chsh.py

# Parity oscillations (GHZ states)
python docs/examples/qpu/parity_oscillation.py
python docs/examples/qulacs/parity_oscillation.py
python docs/examples/local/parity_oscillation.py
```

### Advanced Techniques
Parallel execution and parameter scans:

```bash
# Parallel Rabi measurements
python docs/examples/qpu/rabi_parallel.py
python docs/examples/qulacs/rabi_parallel.py
python docs/examples/local/rabi_parallel.py

# Phase scan experiments
python docs/examples/qpu/chsh_phase_scan.py
python docs/examples/local/chsh_phase_scan.py
```

## Common Usage Patterns

### Basic Experiment Template

```python
from oqtopus_experiments.backends import LocalBackend, OqtopusBackend
from oqtopus_experiments.experiments import Rabi

# 1. Choose backend
backend = LocalBackend(device="qulacs")  # or OqtopusBackend()

# 2. Configure experiment
exp = Rabi(
    experiment_name="my_experiment",
    physical_qubit=0,
    amplitude_points=20,
    max_amplitude=2.0,
)

# 3. Execute
result = exp.run(backend=backend, shots=1000)

# 4. Analyze
df = result.analyze(plot=True, save_data=True, save_image=True)
print(f"π-pulse amplitude: {df['pi_amplitude'].iloc[0]:.3f}")
```

### Comparative Studies

```python
# Compare backends
backends = {
    'ideal': LocalBackend(device="qulacs"),
    'noisy': LocalBackend(device="noisy"),
    'hardware': OqtopusBackend(),
}

results = {}
for name, backend in backends.items():
    exp = Rabi(experiment_name=f"rabi_{name}", amplitude_points=15, max_amplitude=2.0)
    result = exp.run(backend=backend, shots=1000)
    df = result.analyze(plot=True)
    results[name] = df['pi_amplitude'].iloc[0]

# Compare results
for name, pi_amp in results.items():
    print(f"{name:8}: π-amplitude = {pi_amp:.3f}")
```

### Multi-Qubit Characterization

```python
# Characterize multiple qubits
qubits = [0, 1, 2]
experiments = ['rabi', 't1', 'ramsey']

for qubit in qubits:
    print(f"\\nCharacterizing Qubit {qubit}:")
    
    # Rabi calibration
    rabi = Rabi(physical_qubit=qubit, amplitude_points=20, max_amplitude=2.0)
    rabi_result = rabi.run(backend=backend, shots=1000)
    rabi_df = rabi_result.analyze(plot=True)
    
    # T1 measurement
    t1 = T1(physical_qubit=qubit, delay_points=15, max_delay=50000.0)
    t1_result = t1.run(backend=backend, shots=1000)
    t1_df = t1_result.analyze(plot=True)
    
    print(f"  π-amplitude: {rabi_df['pi_amplitude'].iloc[0]:.3f}")
    # print(f"  T1 time: {t1_df['t1_time'].iloc[0]:.1f} μs")
```

## Running Examples

### Prerequisites

1. Install OQTOPUS Experiments:
```bash
pip install git+https://github.com/orangekame3/oqtopus-experiments.git
```

2. For real hardware access, configure OQTOPUS credentials

3. For enhanced visualization:
```bash
pip install plotly
```

### Execution

Navigate to the repository and run any example:

```bash
# Clone repository
git clone https://github.com/orangekame3/oqtopus-experiments.git
cd oqtopus-experiments

# Run examples
python docs/examples/qulacs/rabi.py
python docs/examples/local/chsh.py
python docs/examples/qpu/t1_parallel.py  # Requires OQTOPUS credentials
```

## Output and Results

Each example generates:

- **Interactive plots**: Plotly visualizations with experiment data
- **Data files**: JSON and CSV formats in timestamped directories
- **Analysis results**: Fitted parameters and quality metrics
- **Console output**: Summary statistics and key findings

Example output structure:
```
experiments_data_YYYYMMDD_HHMMSS/
├── experiment_data.json          # Raw measurement data
├── experiment_metadata.json      # Experiment configuration
├── analysis_results.csv          # Processed results
└── plots/
    └── rabi_plot.html            # Interactive visualization
```

## Next Steps

- Explore specific backend examples:
  - [Real Hardware Examples](qpu.md)
  - [Simulation Examples](simulation.md)
- Learn about [experiment types](../experiments/index.md)
- Check [API documentation](../reference/SUMMARY.md) for advanced usage