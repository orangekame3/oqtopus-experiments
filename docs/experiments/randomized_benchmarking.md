# Randomized Benchmarking

Randomized Benchmarking (RB) is a protocol for characterizing the average error rate of quantum gates in a manner that is largely insensitive to state preparation and measurement (SPAM) errors.

## Overview

RB works by applying random sequences of Clifford gates of varying lengths, followed by measuring the "survival probability" (probability of measuring the initial state). The decay of this probability as a function of sequence length reveals the average error per gate.

### Key Features

- **SPAM-insensitive**: Results are largely independent of state preparation and measurement errors
- **Scalable**: Can be applied to single qubits or multi-qubit systems  
- **Standardized**: Widely used benchmark in quantum computing
- **Versatile**: Standard and interleaved variants available

## Theory

The RB protocol measures the survival probability P(m) as a function of sequence length m:

```
P(m) = A × r^m + B
```

Where:
- **A**: Initial fidelity (amplitude)
- **r**: Decay rate (related to gate fidelity)
- **B**: Offset (accounts for measurement bias)
- **m**: Number of Clifford gates in the sequence

The average error per Clifford gate is calculated as:
```
error_per_clifford = (1 - r) / 2
```

## Usage

### Standard Randomized Benchmarking

```python
from oqtopus_experiments.backends import LocalBackend
from oqtopus_experiments.experiments import RandomizedBenchmarking

# Choose backend
backend = LocalBackend(device="noisy")  # or OqtopusBackend()

# Create experiment
rb = RandomizedBenchmarking(
    experiment_name="standard_rb",
    physical_qubit=0,
    max_sequence_length=100,  # Maximum Clifford sequence length
    num_lengths=10,           # Number of different lengths to test
    num_samples=50,           # Random sequences per length
    rb_type="standard",       # Standard RB
)

# Run experiment
result = rb.run(backend=backend, shots=1024)

# Analyze with fitting and plotting
df = result.analyze(plot=True, save_data=True, save_image=True)
print(df)
```

### Interleaved Randomized Benchmarking

Interleaved RB isolates the error of a specific target gate:

```python
# Standard RB for reference
rb_ref = RandomizedBenchmarking(
    experiment_name="rb_reference",
    physical_qubit=0,
    max_sequence_length=50,
    num_lengths=8,
    num_samples=30,
    rb_type="standard",
)

# Interleaved RB for X gate
rb_interleaved = RandomizedBenchmarking(
    experiment_name="rb_x_gate", 
    physical_qubit=0,
    max_sequence_length=50,
    num_lengths=8,
    num_samples=30,
    rb_type="interleaved",
    interleaved_gate="x",  # Target gate to characterize
)

# Run both experiments
result_ref = rb_ref.run(backend=backend, shots=1024)
result_int = rb_interleaved.run(backend=backend, shots=1024)

# Compare decay rates to isolate X gate error
df_ref = result_ref.analyze(plot=True)
df_int = result_int.analyze(plot=True)
```

## Parameters

### RandomizedBenchmarking

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `experiment_name` | str \| None | None | Name for the experiment |
| `physical_qubit` | int \| None | None | Target physical qubit |
| `max_sequence_length` | int | 100 | Maximum number of Clifford gates |
| `num_lengths` | int | 10 | Number of sequence lengths to test |
| `num_samples` | int | 50 | Random sequences per length |
| `rb_type` | str | "standard" | "standard" or "interleaved" |
| `interleaved_gate` | str \| None | None | Gate to interleave (for interleaved RB) |

## Results

The `analyze()` method returns a pandas DataFrame with:

| Column | Description |
|--------|-------------|
| `sequence_length` | Number of Clifford gates in sequence |
| `mean_survival_probability` | Average survival probability |
| `std_survival_probability` | Standard deviation |
| `fitted_survival_probability` | Fitted curve values |

### Fitting Results

The analysis includes exponential decay fitting with:

- **Error per Clifford**: Average error rate per gate
- **Decay rate (r)**: Exponential decay parameter  
- **Initial fidelity (A)**: Amplitude parameter
- **Offset (B)**: Measurement bias parameter
- **R-squared**: Goodness of fit metric

## Visualization

Following the project's plot settings guidelines, RB plots feature:

- White background with light gray grid
- Consistent color scheme using `get_experiment_colors()`
- Data points with error bars (green)
- Exponential fit curve (blue)
- Statistics box with fitting parameters
- 1000×500px dimensions for analysis plots

## Interpretation

### Standard RB Results

- **Low error per Clifford (< 10⁻³)**: Excellent gate quality
- **Moderate error (10⁻³ to 10⁻²)**: Typical for current NISQ devices
- **High error (> 10⁻²)**: Poor gate quality, calibration needed

### Quality Indicators

- **R² > 0.95**: Excellent fit, reliable results
- **R² > 0.80**: Good fit, results likely valid
- **R² < 0.80**: Poor fit, check data quality

### Interleaved RB

The ratio of decay rates reveals the target gate error:
```
r_interleaved / r_standard = 1 - 2 × error_target_gate
```

## Examples

See the `docs/examples/` directory for complete examples:

- **Local simulation**: `local/randomized_benchmarking.py`
- **QPU hardware**: `qpu/randomized_benchmarking.py`
- **Qulacs simulation**: `qulacs/randomized_benchmarking.py`

## Typical Workflow

1. **Run standard RB** to characterize overall gate performance
2. **Check fit quality** (R-squared value)
3. **Compare with T1/T2** measurements for consistency  
4. **Use interleaved RB** to isolate specific gate errors
5. **Track over time** to monitor device stability

## Technical Notes

- Uses single-qubit Clifford group (24 elements)
- Sequence lengths are logarithmically spaced
- Includes proper inverse calculation for return to initial state
- Supports both noiseless and noisy simulations
- Compatible with all OQTOPUS backends

## References

- Magesan et al., "Efficient measurement of quantum gate error by interleaved randomized benchmarking" (2012)
- Gambetta et al., "Characterization of addressability by simultaneous randomized benchmarking" (2012)
- Nielsen & Chuang, "Quantum Computation and Quantum Information"