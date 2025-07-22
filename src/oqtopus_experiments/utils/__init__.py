"""
OQTOPUS Experiments Utilities Package

This package provides common utility functions that are shared across
different experiment types to reduce code duplication.
"""

from .data_management import (
    create_backup_copy,
    create_experiment_data_structure,
    generate_experiment_metadata,
    load_experiment_data,
    save_experiment_data,
)
from .display import (
    create_experiment_summary,
    display_experiment_results,
    format_scientific_notation,
    format_time_duration,
)
from .oqtopus_utils import (
    calculate_shots_from_counts,
    convert_decimal_to_binary_counts,
    extract_counts_from_oqtopus_result,
    normalize_counts,
    validate_oqtopus_result,
)
from .statistics import (
    calculate_fidelity,
    calculate_p0_probability,
    calculate_p1_probability,
    calculate_probability,
    calculate_z_expectation,
    damped_oscillation,
    echo_decay,
    estimate_parameters_with_quality,
    exponential_decay,
    rabi_oscillation,
)
from .visualization import (
    apply_experiment_layout,
    get_experiment_colors,
    get_plotly_config,
    save_plotly_figure,
    setup_plotly_environment,
    show_plotly_figure,
)

# Advanced utilities (optional dependencies)
# Plotting functionality is in visualization module
try:
    from .visualization import (
        apply_experiment_layout,
        get_experiment_colors,
        get_plotly_config,
        save_plotly_figure,
        setup_plotly_environment,
        show_plotly_figure,
    )

    _PLOTTING_AVAILABLE = True
except ImportError:
    _PLOTTING_AVAILABLE = False

# Analysis functionality is in statistics module
try:
    from .statistics import (
        calculate_fidelity,
        calculate_probability,
    )

    _ANALYSIS_AVAILABLE = True
except ImportError:
    _ANALYSIS_AVAILABLE = False

__all__ = [
    # OQTOPUS utilities
    "convert_decimal_to_binary_counts",
    "validate_oqtopus_result",
    "extract_counts_from_oqtopus_result",
    "calculate_shots_from_counts",
    "normalize_counts",
    # Display utilities
    "display_experiment_results",
    "create_experiment_summary",
    "format_time_duration",
    "format_scientific_notation",
    # Statistics utilities
    "calculate_probability",
    "calculate_p0_probability",
    "calculate_p1_probability",
    "calculate_z_expectation",
    "estimate_parameters_with_quality",
    "exponential_decay",
    "damped_oscillation",
    "echo_decay",
    "rabi_oscillation",
    "calculate_fidelity",
    # Data management utilities
    "create_experiment_data_structure",
    "save_experiment_data",
    "load_experiment_data",
    "generate_experiment_metadata",
    "create_backup_copy",
    # Visualization utilities
    "setup_plotly_environment",
    "get_plotly_config",
    "get_experiment_colors",
    "apply_experiment_layout",
    "save_plotly_figure",
    "show_plotly_figure",
]

# Conditionally add advanced utilities to __all__ if available
if _PLOTTING_AVAILABLE:
    __all__.extend(
        [
            "create_experiment_plot",
            "create_multi_device_comparison_plot",
            "create_3d_surface_plot",
            "apply_experiment_theme",
        ]
    )

if _ANALYSIS_AVAILABLE:
    __all__.extend(
        [
            "calculate_uncertainty_propagation",
            "perform_cross_experiment_analysis",
            "calculate_quantum_metrics",
            "perform_statistical_tests",
            "optimize_experiment_parameters",
        ]
    )
