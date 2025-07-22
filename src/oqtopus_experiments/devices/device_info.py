#!/usr/bin/env python3
"""
DeviceInfo Class - Simple device information access and visualization
"""

import json
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

# OQTOPUS imports
try:
    from quri_parts_oqtopus.backend import OqtopusDeviceBackend

    OQTOPUS_AVAILABLE = True
except ImportError:
    OQTOPUS_AVAILABLE = False


class DeviceInfo:
    """
    Simple device information class for quantum devices

    Usage:
        device = DeviceInfo("anemone")
        device.show()  # Display comprehensive device info
        device.plot_layout()  # Show qubit layout
        device.get_best_qubits(5)  # Get top 5 qubits by fidelity
    """

    def __init__(self, device_name: str = "anemone"):
        """
        Initialize device information

        Args:
            device_name: Name of the quantum device to query
        """
        self.device_name = device_name
        self._device_data = None
        self._device_info = None
        self._qubits_df = None
        self._couplings_df = None
        self.console = Console()

        # Load device information
        self._load_device_info()

    def _load_device_info(self):
        """Load device information from OQTOPUS backend"""
        if not OQTOPUS_AVAILABLE:
            print("❌ OQTOPUS not available - device information cannot be loaded")
            return

        try:
            backend = OqtopusDeviceBackend()
            self._device_data = backend.get_device(self.device_name)

            # Parse device_info
            self._device_info = (
                json.loads(self._device_data.device_info)
                if isinstance(self._device_data.device_info, str)
                else self._device_data.device_info
            )

            # Create DataFrames for easy analysis
            self._create_dataframes()

            print(f"✅ Device '{self.device_name}' information loaded successfully")

        except Exception as e:
            print(f"❌ Failed to load device '{self.device_name}': {e}")

    def _create_dataframes(self):
        """Create pandas DataFrames from device information"""
        if not self._device_info:
            return

        # Qubits DataFrame
        qubit_data = []
        for q in self._device_info["qubits"]:
            qubit_data.append(
                {
                    "id": int(q["id"]),
                    "physical_id": int(q["physical_id"]),
                    "x": q["position"]["x"],
                    "y": q["position"]["y"],
                    "fidelity": q["fidelity"],
                    "t1": q["qubit_lifetime"]["t1"],
                    "t2": q["qubit_lifetime"]["t2"],
                    "readout_error": q["meas_error"]["readout_assignment_error"],
                }
            )
        self._qubits_df = pd.DataFrame(qubit_data)
        # Ensure ID columns are integers
        self._qubits_df["id"] = self._qubits_df["id"].astype(int)
        self._qubits_df["physical_id"] = self._qubits_df["physical_id"].astype(int)

        # Couplings DataFrame
        coupling_data = []
        for c in self._device_info["couplings"]:
            coupling_data.append(
                {
                    "control": int(c["control"]),
                    "target": int(c["target"]),
                    "fidelity": c["fidelity"],
                    "gate_duration_ns": c["gate_duration"]["rzx90"],
                }
            )
        self._couplings_df = pd.DataFrame(coupling_data)
        # Ensure ID columns are integers
        self._couplings_df["control"] = self._couplings_df["control"].astype(int)
        self._couplings_df["target"] = self._couplings_df["target"].astype(int)

    @property
    def available(self) -> bool:
        """Check if device information is available"""
        return self._device_data is not None

    @property
    def qubits(self) -> pd.DataFrame | None:
        """Get qubits DataFrame"""
        return self._qubits_df

    @property
    def couplings(self) -> pd.DataFrame | None:
        """Get couplings DataFrame"""
        return self._couplings_df

    @property
    def device_info(self) -> dict | None:
        """Get raw device information dictionary"""
        return self._device_info

    def summary(self) -> dict[str, Any]:
        """Get device summary information"""
        if not self.available:
            return {"error": "Device information not available"}

        device = self._device_data
        return {
            "device_id": device.device_id,
            "description": device.description.strip(),
            "device_type": device.device_type,
            "status": device.status,
            "n_qubits": device.n_qubits,
            "n_pending_jobs": device.n_pending_jobs,
            "basis_gates": device.basis_gates,
            "calibrated_at": device.calibrated_at,
        }

    def get_best_qubits(
        self, n: int = 5, sorted_key: str = "fidelity"
    ) -> pd.DataFrame | None:
        """
        Get top N qubits by fidelity

        Args:
            n: Number of top qubits to return
            sorted_key: Key to sort by (default: "fidelity")

        Returns:
            DataFrame with top N qubits sorted by fidelity
        """
        if self._qubits_df is None:
            return None

        # Sort by fidelity and take top n
        sorted_df = (
            self._qubits_df.sort_values(sorted_key, ascending=False).head(n).copy()
        )
        # Explicitly ensure integer types are preserved
        sorted_df["id"] = sorted_df["id"].astype("int64")
        sorted_df["physical_id"] = sorted_df["physical_id"].astype("int64")
        return sorted_df

    def get_worst_qubits(self, n: int = 5) -> pd.DataFrame | None:
        """
        Get bottom N qubits by fidelity

        Args:
            n: Number of bottom qubits to return

        Returns:
            DataFrame with bottom N qubits sorted by fidelity
        """
        if self._qubits_df is None:
            return None

        return self._qubits_df.nsmallest(n, "fidelity")

    def get_qubit_stats(self) -> dict[str, Any] | None:
        """Get statistical summary of qubit properties"""
        if self._qubits_df is None:
            return None

        stats = {
            "fidelity": {
                "mean": float(self._qubits_df["fidelity"].mean()),
                "std": float(self._qubits_df["fidelity"].std()),
                "min": float(self._qubits_df["fidelity"].min()),
                "max": float(self._qubits_df["fidelity"].max()),
            },
            "t1": {
                "mean": float(self._qubits_df["t1"].mean()),
                "std": float(self._qubits_df["t1"].std()),
                "min": float(self._qubits_df["t1"].min()),
                "max": float(self._qubits_df["t1"].max()),
            },
            "t2": {
                "mean": float(self._qubits_df["t2"].mean()),
                "std": float(self._qubits_df["t2"].std()),
                "min": float(self._qubits_df["t2"].min()),
                "max": float(self._qubits_df["t2"].max()),
            },
            "readout_error": {
                "mean": float(self._qubits_df["readout_error"].mean()),
                "std": float(self._qubits_df["readout_error"].std()),
                "min": float(self._qubits_df["readout_error"].min()),
                "max": float(self._qubits_df["readout_error"].max()),
            },
        }
        return stats

    def show(self, show_qubits: bool = True, show_couplings: bool = True):
        """
        Display comprehensive device information in rich format

        Args:
            show_qubits: Whether to show qubit table
            show_couplings: Whether to show coupling map
        """
        if not self.available:
            self.console.print("[red]❌ Device information not available[/red]")
            return

        device = self._device_data

        # Device summary panel
        self.console.print(
            Panel.fit(
                f"[bold cyan]{device.description.strip()}[/bold cyan]\n"
                f"ID: [green]{device.device_id}[/green] | "
                f"Type: {device.device_type} | "
                f"Status: [red]{device.status}[/red]\n"
                f"Qubits: {device.n_qubits} | "
                f"Pending Jobs: {device.n_pending_jobs}\n"
                f"Basis Gates: {device.basis_gates}\n"
                f"Calibrated at: {device.calibrated_at}\n",
                title=f"🖥️  Device: {self.device_name}",
            )
        )

        # Qubit statistics
        stats = self.get_qubit_stats()
        if stats:
            self.console.print(
                Panel.fit(
                    f"Fidelity: [green]{stats['fidelity']['mean']:.5f} ± {stats['fidelity']['std']:.5f}[/green] "
                    f"(range: {stats['fidelity']['min']:.5f} - {stats['fidelity']['max']:.5f})\n"
                    f"T₁: [blue]{stats['t1']['mean']:.1f} ± {stats['t1']['std']:.1f} μs[/blue] "
                    f"(range: {stats['t1']['min']:.1f} - {stats['t1']['max']:.1f} μs)\n"
                    f"T₂: [blue]{stats['t2']['mean']:.1f} ± {stats['t2']['std']:.1f} μs[/blue] "
                    f"(range: {stats['t2']['min']:.1f} - {stats['t2']['max']:.1f} μs)\n"
                    f"Readout Error: [yellow]{stats['readout_error']['mean']:.4f} ± {stats['readout_error']['std']:.4f}[/yellow] "
                    f"(range: {stats['readout_error']['min']:.4f} - {stats['readout_error']['max']:.4f})",
                    title="📊 Qubit Statistics",
                )
            )

        if show_qubits and self._qubits_df is not None:
            self._show_qubit_table()

        if show_couplings and self._device_info:
            self._show_coupling_tree()

    def _show_qubit_table(self):
        """Display qubit properties table"""
        table = Table(
            title="🎯 Qubit Properties (Top 10 by Fidelity)", show_lines=False
        )
        table.add_column("ID", justify="right")
        table.add_column("Phys.ID", justify="right")
        table.add_column("Position", justify="center")
        table.add_column("T₁ (μs)", justify="right")
        table.add_column("T₂ (μs)", justify="right")
        table.add_column("Fidelity", justify="right")
        table.add_column("R.O. Error", justify="right")

        # Show top 10 qubits by fidelity
        top_qubits = self.get_best_qubits(10)
        for _, q in top_qubits.iterrows():
            table.add_row(
                str(int(q["id"])),
                str(int(q["physical_id"])),
                f"({q['x']:.1f},{q['y']:.1f})",
                f"{q['t1']:.1f}",
                f"{q['t2']:.1f}",
                f"{q['fidelity']:.5f}",
                f"{q['readout_error']:.3f}",
            )

        self.console.print(table)

    def _show_coupling_tree(self):
        """Display coupling map as tree"""
        tree = Tree("🔗 Coupling Map (control → target)")
        for c in self._device_info["couplings"]:
            tree.add(
                f"{c['control']:>2} → {c['target']:<2} | "
                f"Fidelity: {c['fidelity']:.3f}, "
                f"Duration: {c['gate_duration']['rzx90']} ns"
            )
        self.console.print(tree)

    def plot_layout(
        self,
        color_by: str = "fidelity",
        show_edges: bool = True,
        renderer: str = "browser",
    ):
        """
        Plot device qubit layout with interactive visualization

        Args:
            color_by: Property to color qubits by ("fidelity", "t1", "t2", "readout_error")
            size_by: Property to size qubits by ("fidelity", "t1", "t2")
            show_edges: Whether to show coupling edges
            renderer: Plotly renderer ("browser", "notebook", etc.)
        """
        if not self.available or self._qubits_df is None:
            print("❌ Device information not available for plotting")
            return

        pio.renderers.default = renderer

        df = self._qubits_df
        pos = {row["id"]: (row["x"], row["y"]) for _, row in df.iterrows()}

        fig_data = []

        # Add coupling edges if requested
        if show_edges and self._device_info:
            # Collect all coupling fidelities for color mapping
            coupling_fidelities = []
            valid_couplings = []
            
            for c in self._device_info["couplings"]:
                src, tgt = c["control"], c["target"]
                if src in pos and tgt in pos:
                    fidelity = c.get("fidelity", 0.0)
                    if isinstance(fidelity, (int, float)):
                        coupling_fidelities.append(fidelity)
                        valid_couplings.append(c)
            
            # Create individual traces for each coupling edge with color-coded fidelity
            for c in valid_couplings:
                src, tgt = c["control"], c["target"]
                x0, y0 = pos[src]
                x1, y1 = pos[tgt]
                
                # Get coupling properties
                fidelity = c.get("fidelity", 0.0)
                duration_ns = c.get("duration_ns", "N/A")
                
                # Color-code edge based on fidelity (red=low, green=high) with thicker lines
                if fidelity >= 0.95:
                    edge_color = "rgba(34, 139, 34, 0.8)"  # Green for high fidelity
                    edge_width = 6  # Much thicker for better visibility
                elif fidelity >= 0.90:
                    edge_color = "rgba(255, 165, 0, 0.8)"  # Orange for medium fidelity
                    edge_width = 5
                else:
                    edge_color = "rgba(220, 20, 60, 0.8)"  # Red for low fidelity
                    edge_width = 4
                
                # Create hover text with coupling information
                hover_text = (
                    f"<b>Coupling Gate</b><br>"
                    f"{src} → {tgt}<br>"
                    f"<b>Fidelity: {fidelity:.4f}</b><br>"
                    f"Control: {src}<br>"
                    f"Target: {tgt}<br>"
                    f"Duration: {duration_ns} ns" if isinstance(duration_ns, (int, float)) else f"Duration: {duration_ns}"
                )
                
                fig_data.append(
                    go.Scatter(
                        x=[x0, x1],
                        y=[y0, y1],
                        mode="lines",
                        line={"color": edge_color, "width": edge_width},
                        hoverinfo="text",
                        hovertext=hover_text,
                        showlegend=False,
                        opacity=0.8,
                    )
                )
                
                # Add arrowhead to show direction from control to target
                # Calculate arrow position and angle
                arrow_length = 0.5  # Longer arrow for better visibility
                angle = np.arctan2(y1 - y0, x1 - x0)
                
                # Position arrow closer to target (75% along the edge)
                arrow_x = x0 + 0.75 * (x1 - x0)
                arrow_y = y0 + 0.75 * (y1 - y0)
                
                # Calculate arrow endpoints with wider angle for better visibility
                arrow_angle1 = angle + np.pi - np.pi/4  # 45 degrees for wider arrow
                arrow_angle2 = angle + np.pi + np.pi/4  # 45 degrees for wider arrow
                
                arrow_x1 = arrow_x + arrow_length * np.cos(arrow_angle1)
                arrow_y1 = arrow_y + arrow_length * np.sin(arrow_angle1)
                arrow_x2 = arrow_x + arrow_length * np.cos(arrow_angle2)
                arrow_y2 = arrow_y + arrow_length * np.sin(arrow_angle2)
                
                # Add thicker arrow lines with higher contrast
                fig_data.append(
                    go.Scatter(
                        x=[arrow_x1, arrow_x, arrow_x2],
                        y=[arrow_y1, arrow_y, arrow_y2],
                        mode="lines",
                        line={"color": edge_color, "width": max(3, edge_width-1)},  # Thicker arrow
                        hoverinfo="skip",
                        showlegend=False,
                        opacity=1.0,  # Full opacity for better visibility
                    )
                )
                
                # Add a small filled circle at arrow tip for even better visibility
                fig_data.append(
                    go.Scatter(
                        x=[arrow_x],
                        y=[arrow_y],
                        mode="markers",
                        marker={
                            "size": 6,
                            "color": edge_color,
                            "opacity": 1.0,
                        },
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
                
                # Add coupling fidelity text at the midpoint of the edge
                mid_x = (x0 + x1) / 2
                mid_y = (y0 + y1) / 2
                
                # Calculate edge length to determine if we should show text
                edge_length = ((x1 - x0)**2 + (y1 - y0)**2)**0.5
                
                # Only show text on edges that are long enough to avoid overcrowding
                if edge_length > 1.0:  # Minimum edge length threshold
                    # Offset text slightly to avoid overlap with the line
                    offset = 0.1
                    angle = np.arctan2(y1 - y0, x1 - x0)
                    text_x = mid_x + offset * np.sin(angle)
                    text_y = mid_y - offset * np.cos(angle)
                    
                    # Add text with black outline for better visibility
                    # First add white background text for outline effect
                    fig_data.append(
                        go.Scatter(
                            x=[text_x],
                            y=[text_y],
                            mode="text",
                            text=[f"{fidelity:.3f}"],
                            textposition="middle center",
                            textfont={
                                "size": 13,
                                "color": "white",
                                "family": "Arial Bold",
                            },
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )
                    
                    # Then add black text on top for better contrast
                    fig_data.append(
                        go.Scatter(
                            x=[text_x],
                            y=[text_y],
                            mode="text",
                            text=[f"{fidelity:.3f}"],
                            textposition="middle center",
                            textfont={
                                "size": 11,
                                "color": "black",
                                "family": "Arial Bold",
                            },
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )

        # Add qubit nodes
        color_values = df[color_by]

        # Enhanced node appearance with larger size
        fig_data.append(
            go.Scatter(
                x=df["x"],
                y=df["y"],
                mode="markers+text",
                marker={
                    "size": 40,  # Much larger for better text accommodation
                    "color": color_values,
                    "colorscale": "RdYlGn",  # Red-Yellow-Green for intuitive fidelity mapping
                    "colorbar": {
                        "title": {
                            "text": f"<b>Node {color_by.replace('_', ' ').title()}</b>",
                            "font": {"size": 14}
                        },
                        "tickfont": {"size": 12},
                        "len": 0.4,  # Shorter colorbar
                        "thickness": 15,
                        "x": 1.02,  # Position to the right
                        "y": 0.8,   # Position at top
                    },
                    "showscale": True,
                    "line": {"color": "black", "width": 3},  # Thicker border for larger nodes
                    "opacity": 0.9,
                },
                text=[f"{int(row['id'])}<br><span style='font-size:10px'>{row['fidelity']:.3f}</span>" for _, row in df.iterrows()],
                textposition="middle center",
                textfont={
                    "size": 14, 
                    "color": "white", 
                    "family": "Arial Black"
                },
                hoverinfo="text",
                hovertext=[
                    f"<b>Qubit {row['id']}</b><br>"
                    f"Physical ID: {row['physical_id']}<br>"
                    f"Position: ({row['x']:.1f}, {row['y']:.1f})<br>"
                    f"<b>Fidelity: {row['fidelity']:.5f}</b><br>"
                    f"T₁: {row['t1']:.1f} μs<br>"
                    f"T₂: {row['t2']:.1f} μs<br>"
                    f"Readout Error: {row['readout_error']:.4f}"
                    for _, row in df.iterrows()
                ],
                showlegend=False,
            )
        )

        # Add invisible legend traces for coupling edge colors
        legend_traces = [
            go.Scatter(
                x=[None], y=[None],
                mode="lines",
                line={"color": "rgba(34, 139, 34, 0.8)", "width": 6},
                name="High Fidelity (≥0.95)",
                showlegend=True,
            ),
            go.Scatter(
                x=[None], y=[None],
                mode="lines", 
                line={"color": "rgba(255, 165, 0, 0.8)", "width": 5},
                name="Medium Fidelity (0.90-0.95)",
                showlegend=True,
            ),
            go.Scatter(
                x=[None], y=[None],
                mode="lines",
                line={"color": "rgba(220, 20, 60, 0.8)", "width": 4},
                name="Low Fidelity (<0.90)",
                showlegend=True,
            ),
        ]
        
        # Create figure with all data and legend
        fig = go.Figure(data=fig_data + legend_traces)
        fig.update_layout(
            title={
                "text": f"<b>{self.device_name.upper()} Device Topology</b><br>"
                        f"<sub>Nodes: {color_by.replace('_', ' ').title()} | "
                        f"Edges: Coupling Gate Fidelity</sub>",
                "x": 0.5,
                "font": {"size": 16}
            },
            xaxis={
                "title": {
                    "text": "<b>X Position</b>",
                    "font": {"size": 14}
                },
                "showgrid": True, 
                "gridcolor": "lightgray"
            },
            yaxis={
                "title": {
                    "text": "<b>Y Position</b>",
                    "font": {"size": 14}
                },
                "showgrid": True, 
                "gridcolor": "lightgray"
            },
            plot_bgcolor="white",
            height=800,
            width=1000,
            legend={
                "title": {
                    "text": "<b>Edge Coupling Fidelity</b>",
                    "font": {"size": 12}
                },
                "font": {"size": 11},
                "x": 1.02,
                "y": 0.3,  # Position below the colorbar
                "bgcolor": "rgba(255, 255, 255, 0.8)",
                "bordercolor": "black",
                "borderwidth": 1,
            },
            annotations=[
                {
                    "text": "📊 Nodes show Qubit ID + Fidelity | Edges show Coupling Fidelity",
                    "x": 0.5, "y": -0.1,
                    "xref": "paper", "yref": "paper",
                    "showarrow": False,
                    "font": {"size": 12, "color": "darkblue"}
                }
            ]
        )

        fig.show()

    def plot_statistics(self, renderer: str = "browser"):
        """
        Plot device statistics histograms

        Args:
            renderer: Plotly renderer ("browser", "notebook", etc.)
        """
        if not self.available or self._qubits_df is None:
            print("❌ Device information not available for plotting")
            return

        pio.renderers.default = renderer

        df = self._qubits_df

        # Create subplots
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Fidelity Distribution",
                "T₁ Distribution",
                "T₂ Distribution",
                "Readout Error Distribution",
            ),
        )

        # Fidelity histogram
        fig.add_trace(
            go.Histogram(
                x=df["fidelity"], name="Fidelity", nbinsx=20, marker_color="green"
            ),
            row=1,
            col=1,
        )

        # T1 histogram
        fig.add_trace(
            go.Histogram(x=df["t1"], name="T₁", nbinsx=20, marker_color="blue"),
            row=1,
            col=2,
        )

        # T2 histogram
        fig.add_trace(
            go.Histogram(x=df["t2"], name="T₂", nbinsx=20, marker_color="cyan"),
            row=2,
            col=1,
        )

        # Readout error histogram
        fig.add_trace(
            go.Histogram(
                x=df["readout_error"],
                name="Readout Error",
                nbinsx=20,
                marker_color="orange",
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            title_text=f"Device Statistics: {self.device_name}",
            showlegend=False,
            height=600,
        )

        # Update x-axis labels
        fig.update_xaxes(title_text="Fidelity", row=1, col=1)
        fig.update_xaxes(title_text="T₁ (μs)", row=1, col=2)
        fig.update_xaxes(title_text="T₂ (μs)", row=2, col=1)
        fig.update_xaxes(title_text="Readout Error", row=2, col=2)

        # Update y-axis labels
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=2)

        fig.show()

    def compare_qubits(self, qubit_ids: list[int]) -> pd.DataFrame | None:
        """
        Compare specific qubits

        Args:
            qubit_ids: List of qubit IDs to compare

        Returns:
            DataFrame with comparison of specified qubits
        """
        if self._qubits_df is None:
            return None

        return self._qubits_df[self._qubits_df["id"].isin(qubit_ids)]

    def save_data(self, filename: str = None) -> str:
        """
        Save device information to JSON file

        Args:
            filename: Output filename (optional)

        Returns:
            Path to saved file
        """
        if not self.available:
            return "❌ No device data available to save"

        if filename is None:
            filename = f"device_{self.device_name}_info.json"

        data = {
            "device_summary": self.summary(),
            "qubit_statistics": self.get_qubit_stats(),
            "qubits": (
                self._qubits_df.to_dict("records")
                if self._qubits_df is not None
                else []
            ),
            "couplings": (
                self._couplings_df.to_dict("records")
                if self._couplings_df is not None
                else []
            ),
        }

        try:
            with open(filename, "w") as f:
                json.dump(data, f, indent=2, default=str)
            print(f"✅ Device information saved to {filename}")
            return filename
        except Exception as e:
            error_msg = f"❌ Failed to save device information: {e}"
            print(error_msg)
            return error_msg
