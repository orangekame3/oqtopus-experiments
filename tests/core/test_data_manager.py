#!/usr/bin/env python3
"""
Tests for ExperimentDataManager
"""

import json
import os
import shutil
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pytest

from oqtopus_experiments.core.data_manager import ExperimentDataManager


class TestExperimentDataManager:
    """Test ExperimentDataManager functionality"""

    def setup_method(self):
        """Setup test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)

    def teardown_method(self):
        """Cleanup test environment"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)

    def test_init_default_name(self):
        """Test initialization with default experiment name"""
        manager = ExperimentDataManager()

        assert manager.timestamp is not None
        assert manager.session_dir.startswith(".results/exp_")
        assert os.path.exists(f"{manager.session_dir}/plots")
        assert os.path.exists(f"{manager.session_dir}/data")
        assert manager.files == []

    def test_init_custom_name(self):
        """Test initialization with custom experiment name"""
        custom_name = "test_experiment"
        manager = ExperimentDataManager(custom_name)

        assert custom_name in manager.session_dir
        assert manager.timestamp in manager.session_dir
        assert os.path.exists(f"{manager.session_dir}/plots")
        assert os.path.exists(f"{manager.session_dir}/data")

    def test_save_plot_single_format(self):
        """Test saving plot in single format"""
        manager = ExperimentDataManager("test")

        # Create test plot
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])

        path = manager.save_plot(fig, "test_plot", ["png"])

        assert path is not None
        assert path.endswith(".png")
        assert os.path.exists(path)
        assert len(manager.files) == 1
        assert path in manager.files

        plt.close(fig)

    def test_save_plot_multiple_formats(self):
        """Test saving plot in multiple formats"""
        manager = ExperimentDataManager("test")

        # Create test plot
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])

        path = manager.save_plot(fig, "test_plot", ["png", "pdf"])

        assert path is not None
        assert path.endswith(".png")  # Returns first format
        assert len(manager.files) == 2

        # Check both files exist
        png_file = path
        pdf_file = path.replace(".png", ".pdf")
        assert os.path.exists(png_file)
        assert os.path.exists(pdf_file)
        assert png_file in manager.files
        assert pdf_file in manager.files

        plt.close(fig)

    def test_save_data_simple(self):
        """Test saving simple data"""
        manager = ExperimentDataManager("test")

        data = {"results": [1, 2, 3], "config": {"shots": 1000}}
        path = manager.save_data(data, "test_data")

        assert path.endswith(".json")
        assert os.path.exists(path)
        assert len(manager.files) == 1
        assert path in manager.files

        # Verify saved content
        with open(path) as f:
            loaded_data = json.load(f)
        assert loaded_data == data

    def test_save_data_with_numpy(self):
        """Test saving data with numpy arrays"""
        manager = ExperimentDataManager("test")

        data = {
            "numpy_array": np.array([1, 2, 3]),
            "numpy_scalar": np.float64(3.14),
            "regular_data": {"shots": 1000},
        }
        path = manager.save_data(data, "numpy_test")

        assert os.path.exists(path)

        # Verify numpy arrays are converted to lists
        with open(path) as f:
            loaded_data = json.load(f)
        assert loaded_data["numpy_array"] == [1, 2, 3]
        assert loaded_data["numpy_scalar"] == 3.14
        assert loaded_data["regular_data"] == {"shots": 1000}

    def test_save_results(self):
        """Test saving experiment results with metadata"""
        manager = ExperimentDataManager("test")

        results = {"circuit_0": [{"counts": {"0": 50, "1": 50}}]}
        metadata = {"version": "1.0", "shots": 100}
        experiment_type = "rabi"

        path = manager.save_results(results, metadata, experiment_type)

        assert path.endswith(".json")
        assert os.path.exists(path)
        assert "rabi_results" in path
        assert len(manager.files) == 1

        # Verify saved content
        with open(path) as f:
            loaded_data = json.load(f)

        assert loaded_data["experiment_type"] == experiment_type
        assert loaded_data["results"] == results
        assert loaded_data["metadata"] == metadata
        assert loaded_data["timestamp"] == manager.timestamp

    def test_save_results_default_metadata(self):
        """Test saving results with default metadata"""
        manager = ExperimentDataManager("test")

        results = {"test": "data"}
        path = manager.save_results(results)

        # Verify saved content
        with open(path) as f:
            loaded_data = json.load(f)

        assert loaded_data["metadata"] == {}
        assert loaded_data["experiment_type"] == "generic"

    def test_summary(self):
        """Test creating session summary"""
        manager = ExperimentDataManager("test")

        # Create some files first
        manager.save_data({"test": "data"}, "test1")
        manager.save_data({"test": "data"}, "test2")

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        manager.save_plot(fig, "test_plot")
        plt.close(fig)

        summary_path = manager.summary()

        assert summary_path.endswith("summary.json")
        assert os.path.exists(summary_path)

        # Verify summary content
        with open(summary_path) as f:
            summary_data = json.load(f)

        assert summary_data["session_dir"] == manager.session_dir
        assert summary_data["timestamp"] == manager.timestamp
        assert summary_data["total_files"] == 3  # 2 data files + 1 plot
        assert len(summary_data["files"]) == 3

    def test_get_data_directory(self):
        """Test getting data directory path"""
        manager = ExperimentDataManager("test")

        data_dir = manager.get_data_directory()

        assert data_dir == f"{manager.session_dir}/data"
        assert os.path.exists(data_dir)

    def test_convert_for_json_dict(self):
        """Test JSON conversion for dictionary"""
        manager = ExperimentDataManager("test")

        data = {
            "regular": "string",
            "numpy_array": np.array([1, 2, 3]),
            "nested": {"numpy_scalar": np.float64(3.14)},
        }

        converted = manager._convert_for_json(data)

        assert converted["regular"] == "string"
        assert converted["numpy_array"] == [1, 2, 3]
        assert converted["nested"]["numpy_scalar"] == 3.14

    def test_convert_for_json_list(self):
        """Test JSON conversion for list"""
        manager = ExperimentDataManager("test")

        data = [1, np.array([2, 3]), np.float64(4.5)]

        converted = manager._convert_for_json(data)

        assert converted == [1, [2, 3], 4.5]

    def test_convert_for_json_numpy_types(self):
        """Test JSON conversion for various numpy types"""
        manager = ExperimentDataManager("test")

        # Test numpy array
        numpy_array = np.array([1, 2, 3])
        converted_array = manager._convert_for_json(numpy_array)
        assert converted_array == [1, 2, 3]

        # Test numpy scalar
        numpy_scalar = np.float64(3.14)
        converted_scalar = manager._convert_for_json(numpy_scalar)
        assert converted_scalar == 3.14

        # Test regular types
        regular_data = "string"
        converted_regular = manager._convert_for_json(regular_data)
        assert converted_regular == "string"

    def test_file_tracking(self):
        """Test that files are properly tracked"""
        manager = ExperimentDataManager("test")

        # Initially no files
        assert len(manager.files) == 0

        # Add data file
        data_path = manager.save_data({"test": "data"}, "test1")
        assert len(manager.files) == 1
        assert data_path in manager.files

        # Add plot
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        plot_path = manager.save_plot(fig, "test_plot")
        plt.close(fig)

        assert len(manager.files) == 2
        assert plot_path in manager.files

        # Add results
        results_path = manager.save_results({"test": "results"})
        assert len(manager.files) == 3
        assert results_path in manager.files

    def test_directory_structure(self):
        """Test that proper directory structure is created"""
        manager = ExperimentDataManager("structure_test")

        # Check main directory
        assert os.path.exists(manager.session_dir)
        assert os.path.isdir(manager.session_dir)

        # Check subdirectories
        plots_dir = f"{manager.session_dir}/plots"
        data_dir = f"{manager.session_dir}/data"

        assert os.path.exists(plots_dir)
        assert os.path.isdir(plots_dir)
        assert os.path.exists(data_dir)
        assert os.path.isdir(data_dir)

    def test_timestamp_format(self):
        """Test timestamp format is correct"""
        manager = ExperimentDataManager("test")

        # Timestamp should be in format YYYYMMDD_HHMMSS
        assert len(manager.timestamp) == 15  # 8 + 1 + 6
        assert manager.timestamp[8] == "_"
        assert manager.timestamp[:8].isdigit()
        assert manager.timestamp[9:].isdigit()


def test_main_function():
    """Test the main demo function"""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            # Import and run main function
            from oqtopus_experiments.core.data_manager import main

            # This should run without errors
            main()

            # Check that demo files were created
            assert os.path.exists(".results")

        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    pytest.main([__file__])
