# Quick Start Guide

This guide will help you run your first quantum experiment using OQTOPUS Experiments.

## Your First Experiment

Let's start with a simple Rabi oscillation experiment using a local simulator:

```python
from oqtopus_experiments.backends import LocalBackend
from oqtopus_experiments.experiments import Rabi

# Create a local backend for simulation
backend = LocalBackend(device="qulacs")  # Noiseless simulation

# Create Rabi experiment
exp = Rabi(
    experiment_name="my_first_rabi",
    physical_qubit=0,
    amplitude_points=10,
    max_amplitude=2.0,
)

# Run the experiment
result = exp.run(backend=backend, shots=1000)

# Analyze results with automatic plotting
df = result.analyze(plot=True, save_data=True)
print("Experiment completed!")
print(df.head())
```

## Understanding the Code

### 1. Backend Selection

```python
# Choose your computation backend
backend = LocalBackend(device="qulacs")    # Fast, noiseless simulation
# backend = LocalBackend(device="noisy")   # Realistic, noisy simulation  
# backend = OqtopusBackend()               # Real quantum hardware
```

### 2. Experiment Configuration

```python
exp = Rabi(
    experiment_name="my_experiment",  # Optional: name for data files
    physical_qubit=0,                # Which qubit to measure
    amplitude_points=20,             # Number of amplitude values
    max_amplitude=2.0,               # Maximum drive amplitude
)
```

### 3. Execution and Analysis

```python
# Execute the experiment
result = exp.run(backend=backend, shots=1000)

# Analyze with options
df = result.analyze(
    plot=True,         # Show interactive plots
    save_data=True,    # Save results to JSON/CSV
    save_image=True,   # Save plot images
)
```

## Try Different Experiments

### T1 Relaxation Measurement

```python
from oqtopus_experiments.experiments import T1

exp = T1(
    experiment_name="t1_measurement",
    physical_qubit=0,
    delay_points=15,
    max_delay=50000.0,  # 50 μs
)

result = exp.run(backend=backend, shots=1000)
df = result.analyze(plot=True)
```

### CHSH Bell Test

```python
from oqtopus_experiments.experiments import CHSH

exp = CHSH(
    experiment_name="bell_test",
    phase_points=20,
)

result = exp.run(backend=backend, shots=2000)
df = result.analyze(plot=True)
```

## Using Real Hardware

To run on real quantum hardware, you need OQTOPUS credentials:

```python
from oqtopus_experiments.backends import OqtopusBackend

# Real hardware backend
backend = OqtopusBackend()

# Same experiment code works!
exp = Rabi(physical_qubit=0, amplitude_points=20, max_amplitude=2.0)
result = exp.run(backend=backend, shots=1000)
df = result.analyze(plot=True)
```

## Next Steps

- Explore [Experiment Documentation](../experiments/index.md) for detailed guides
- Check out [Examples](../examples/index.md) for more complex scenarios
- Learn about [Backend Options](../backends/index.md) for different execution environments
- Browse the [API Reference](../reference/SUMMARY.md) for complete documentation

## Troubleshooting

### Common Issues

**Import Error**: Make sure you've installed the package correctly:
```bash
pip install git+https://github.com/orangekame3/oqtopus-experiments.git
```

**Plot Not Showing**: Install plotly for interactive plots:
```bash
pip install plotly
```

**Real Hardware Access**: Ensure your OQTOPUS credentials are properly configured.