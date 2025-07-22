# Installation

## Requirements

- Python 3.12+
- OQTOPUS account and credentials (for real hardware access)

## Install from GitHub

```bash
pip install git+https://github.com/orangekame3/oqtopus-experiments.git
```

## Development Installation

For development, clone the repository and install with uv:

```bash
git clone https://github.com/orangekame3/oqtopus-experiments.git
cd oqtopus-experiments
uv sync
uv pip install -e .
```

## Verify Installation

Test your installation by running a simple example:

```python
from oqtopus_experiments.backends import LocalBackend
from oqtopus_experiments.experiments import Rabi

# Create a simple Rabi experiment with local backend
backend = LocalBackend(device="qulacs")
exp = Rabi(amplitude_points=5, max_amplitude=1.0)

# This should run without errors
circuits = exp.circuits()
print(f"Generated {len(circuits)} circuits successfully!")
```

## Optional Dependencies

### For Real Hardware Access

To use the OQTOPUS backend for real quantum hardware:

1. Sign up for an OQTOPUS account
2. Configure your credentials (follow platform-specific instructions)
3. Install additional quantum computing libraries as needed

### For Advanced Features

```bash
# For enhanced visualization
pip install plotly

# For Jupyter notebook support
pip install jupyter ipywidgets
```

## Next Steps

Once installed, proceed to the [Quick Start Guide](quickstart.md) to run your first experiment.