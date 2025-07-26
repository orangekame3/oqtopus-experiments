# Plot Settings and Visualization Guidelines

## Standard Plot Configuration

### Background and Layout
- `plot_bgcolor="white"` - White plot background
- `paper_bgcolor="white"` - White figure background
- `showgrid=True` - Display grid lines
- `gridwidth=1` - Standard grid line width
- `gridcolor="LightGray"` - Light gray grid color for readability

### Color Scheme
Use consistent colors from `get_experiment_colors()`:
- Primary data: `colors[1]` (typically green)
- Fit/theoretical: `colors[0]` (typically blue)
- Error bars: Match data point colors
- Annotation background: `rgba(255,255,255,0.95)` with `bordercolor="#CCCCCC"`

### Phase Kickback Specific Visualization
- **Main plot**: Angle (degrees) vs <Z_ancilla> with error bars
- **Secondary plot**: Horizontal histogram of expectation values
- **Layout**: Side-by-side subplots (70% main, 30% histogram)
- **Statistics box**: Comprehensive experiment parameters and results

### Subplot Configuration
```python
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Phase Kickback vs Rotation Angle", "Measurement Distribution"),
    column_widths=[0.7, 0.3]
)
```

### Standard Dimensions
- Width: 1000px for analysis plots, 700px for simple plots
- Height: 500px for analysis plots, 400px for simple plots

### Theoretical Fitting Formulas

#### Phase Kickback
For H-RY(θ)-CP(φ)-H circuit:
- Formula: `P(|0⟩) = 1 - sin²(θ/2) sin²(φ/2)`
- Use scipy.optimize.curve_fit for robust parameter estimation
- Display both applied and measured phase with accuracy metrics

#### Hadamard Test
For H-controlled_U-H circuit:
- Formula: `P(|0⟩) = (1 + Re⟨ψ|U|ψ⟩)/2`
- Complex plane visualization when measuring both real and imaginary parts
- Unit circle reference for complex expectation values

### Hadamard Test Specific Visualization
- **Complex plane plot**: Real vs Imaginary expectation values with unit circle
- **Time evolution plot**: Expectation values vs angle with cosine fitting
- **Layout**: Side-by-side (40% complex plane, 60% time evolution)
- **Color coding**: Angle-dependent colormap for complex plane trajectory