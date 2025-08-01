#!/usr/bin/env python3
"""
Simple Data Manager for OQTOPUS Experiments Project
Simple and unified data storage system
"""

import json
import os
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt


class ExperimentDataManager:
    """
    Data management system for quantum experiments
    Handles saving results, plots, and metadata
    """

    def __init__(self, experiment_name: str | None = None):
        """
        Initialize experiment data manager

        Args:
            experiment_name: Experiment name (auto-generated if omitted)
        """
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if experiment_name is None:
            experiment_name = f"exp_{self.timestamp}"

        # Simple directory structure (.results is excluded by gitignore)
        self.session_dir = f".results/{experiment_name}_{self.timestamp}"

        # Create minimum necessary directories
        os.makedirs(f"{self.session_dir}/plots", exist_ok=True)
        os.makedirs(f"{self.session_dir}/data", exist_ok=True)

        self.files: list = []
        print(f"Results: {self.session_dir}")

    def save_plot(self, fig, name: str, formats: list[str] = ["png"]) -> str | None:
        """
        Save plot

        Args:
            fig: matplotlib figure
            name: File name
            formats: Save formats

        Returns:
            Save path
        """
        saved_files: list = []
        for fmt in formats:
            filename = f"{name}_{self.timestamp}.{fmt}"
            path = f"{self.session_dir}/plots/{filename}"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            saved_files.append(path)
            self.files.append(path)

        print(f"📊 Saved: {len(saved_files)} plot files")
        first_file: str | None = saved_files[0] if saved_files else None
        return first_file

    def save_data(self, data: dict[str, Any], name: str) -> str:
        """
        Save data (JSON format)

        Args:
            data: Data to save
            name: File name

        Returns:
            Save path
        """
        filename = f"{name}_{self.timestamp}.json"
        path = f"{self.session_dir}/data/{filename}"

        # JSON save with numpy support
        json_data = self._convert_for_json(data)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        self.files.append(path)
        print(f"Saved: {filename}")
        return path

    def save_results(
        self,
        results: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        experiment_type: str = "generic",
    ) -> str:
        """
        Save experiment results with metadata

        Args:
            results: Experiment results to save
            metadata: Additional metadata
            experiment_type: Type of experiment

        Returns:
            Save path
        """
        save_data = {
            "experiment_type": experiment_type,
            "timestamp": self.timestamp,
            "results": results,
            "metadata": metadata or {},
        }

        filename = f"{experiment_type}_results_{self.timestamp}.json"
        path = f"{self.session_dir}/data/{filename}"

        # JSON save with numpy support
        json_data = self._convert_for_json(save_data)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        self.files.append(path)
        print(f"📊 Saved {experiment_type} results: {filename}")
        return path

    def summary(self) -> str:
        """
        Create session summary

        Returns:
            Summary file path
        """
        summary = {
            "session_dir": self.session_dir,
            "timestamp": self.timestamp,
            "total_files": len(self.files),
            "files": [os.path.basename(f) for f in self.files],
        }

        path = f"{self.session_dir}/summary.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"📋 Summary: {path}")
        return path

    def get_data_directory(self) -> str:
        """
        Get the data directory path

        Returns:
            Data directory path
        """
        return f"{self.session_dir}/data"

    def get_plots_directory(self) -> str:
        """
        Get the plots directory path

        Returns:
            Plots directory path
        """
        return f"{self.session_dir}/plots"

    def _convert_for_json(self, obj):
        """Helper for JSON conversion"""
        if isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif hasattr(obj, "tolist"):  # numpy array
            return obj.tolist()
        elif hasattr(obj, "item"):  # numpy scalar
            return obj.item()
        else:
            return obj


def main():
    """Demo"""
    manager = ExperimentDataManager("demo")

    # Save plot
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 2])
    manager.save_plot(fig, "test_plot")

    # Save data
    data = {"results": [1, 2, 3], "config": {"shots": 1000}}
    manager.save_data(data, "test_data")

    # Create summary
    manager.summary()

    print("✅ Demo completed!")


if __name__ == "__main__":
    main()
