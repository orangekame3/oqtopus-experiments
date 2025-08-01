#!/usr/bin/env python3
"""
Visualization utilities for quantum experiments
Provides plotly-based interactive visualization functionality
"""

import datetime
import os
from typing import Any


def setup_plotly_environment():
    """Setup plotly environment for optimal display"""
    try:
        import plotly.io as pio

        # Force inline display for Jupyter environments
        try:
            from IPython import get_ipython

            if get_ipython() is not None:
                pio.renderers.default = "notebook_connected"
            else:
                pio.renderers.default = "browser"
        except ImportError:
            pio.renderers.default = "browser"
    except ImportError:
        pass


def get_plotly_config(
    filename: str = "experiment", width: int = 600, height: int = 300
) -> dict[str, Any]:
    """
    Get standard plotly configuration for experiments

    Args:
        filename: Base filename for image export
        width: Image width for export
        height: Image height for export

    Returns:
        Plotly configuration dictionary
    """
    return {
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": ["pan2d", "lasso2d", "select2d"],
        "toImageButtonOptions": {
            "format": "png",
            "filename": filename,
            "height": height,
            "width": width,
            "scale": 3,
        },
    }


def get_experiment_colors() -> list[str]:
    """
    Get standard color palette for experiment plots

    Returns:
        List of hex color codes
    """
    return [
        "#0C5DA5",  # Blue for fit line
        "#00B945",  # Green for data points
        "#FF9500",  # Orange
        "#FF2C00",  # Red
        "#845B97",  # Purple
        "#474747",  # Dark gray
        "#9e9e9e",  # Light gray
    ]


def save_plotly_figure(
    fig,
    name: str = "experiment",
    *,
    images_dir: str = "./images",
    formats: list[str] | None = None,
    format: str = "png",  # Legacy parameter for backward compatibility
    width: int = 600,
    height: int = 300,
    scale: int = 3,
) -> str | None:
    """
    Save plotly figure as image file with automatic naming

    Args:
        fig: Plotly figure object
        name: Base name for the image file
        images_dir: Directory to save images
        formats: List of formats to save (default: ["png"])
        format: Legacy single format parameter (deprecated, use formats)
        width: Image width in pixels
        height: Image height in pixels
        scale: Scale factor for image resolution

    Returns:
        Path to saved image file (primary format), or None if saving failed
    """
    try:
        # Create images directory if it doesn't exist
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)

        # Handle format parameters (backward compatibility)
        if formats is None:
            formats = [format]

        saved_files = []

        # Generate unique filename with date and counter
        counter = 1
        current_date = datetime.datetime.now().strftime("%Y%m%d")

        for fmt in formats:
            file_path = os.path.join(
                images_dir,
                f"{current_date}_{name}_{counter}.{fmt}",
            )

            # Find available filename
            temp_counter = counter
            while os.path.exists(file_path):
                temp_counter += 1
                file_path = os.path.join(
                    images_dir,
                    f"{current_date}_{name}_{temp_counter}.{fmt}",
                )

            # Save the figure
            fig.write_image(
                file_path,
                format=fmt,
                width=width,
                height=height,
                scale=scale,
            )
            saved_files.append(file_path)
            print(f"Image saved to {file_path}")

        # Return primary format path
        return saved_files[0] if saved_files else None

    except Exception as e:
        print(f"Failed to save image: {e}")
        print("   Note: Image saving requires kaleido: pip install kaleido")
        return None


def show_plotly_figure(fig, config: dict[str, Any] | None = None):
    """
    Display plotly figure with fallback options

    Args:
        fig: Plotly figure object
        config: Optional plotly configuration
    """
    try:
        # Try to use iplot for guaranteed inline display
        from plotly.offline import iplot

        iplot(fig)
        print("Interactive plot displayed inline (iplot)")
    except Exception:
        # Fallback to show() with proper renderer and config
        fig.show(config=config or {})
        print("Interactive plot created with plotly (show)")


def apply_experiment_layout(
    fig,
    title: str,
    xaxis_title: str = "X",
    yaxis_title: str = "Y",
    height: int = 300,
    width: int = 600,
):
    """
    Apply standard layout settings for experiment plots

    Args:
        fig: Plotly figure object
        title: Plot title
        xaxis_title: X-axis label
        yaxis_title: Y-axis label
        height: Plot height
        width: Plot width
    """
    colors = get_experiment_colors()

    fig.update_layout(
        title={
            "text": title,
            "font": {"size": 16, "color": "#2C3E50"},
            "x": 0.5,
        },
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=height,
        width=width,
        showlegend=True,
        hovermode="closest",
        font={"size": 12, "color": "#2C3E50"},
        plot_bgcolor="white",
        paper_bgcolor="white",
        template=None,  # Disable template to avoid default colors
        colorway=colors,  # Use experiment color palette
        margin={"l": 70, "r": 40, "t": 70, "b": 60},  # Better margins for labels
        legend={
            "x": 0.02,
            "y": 0.98,
            "bgcolor": "rgba(255,255,255,0.8)",
            "bordercolor": "#CCCCCC",
            "borderwidth": 1,
        },
    )

    # Update axes styling
    fig.update_xaxes(
        showgrid=True,
        gridwidth=0.5,
        gridcolor="#E8E8E8",
        showline=True,
        linewidth=1.5,
        linecolor="#2C3E50",
        ticks="outside",
        tickwidth=1,
        tickcolor="#2C3E50",
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=0.5,
        gridcolor="#E8E8E8",
        showline=True,
        linewidth=1.5,
        linecolor="#2C3E50",
        ticks="outside",
        tickwidth=1,
        tickcolor="#2C3E50",
    )
