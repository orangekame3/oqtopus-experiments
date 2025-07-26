#!/usr/bin/env python3
"""
Tests for DeviceInfo
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from oqtopus_experiments.devices.device_info import DeviceInfo


class TestDeviceInfo:
    """Test DeviceInfo functionality"""

    def test_init_default_device(self):
        """Test initialization with default device"""
        with patch('oqtopus_experiments.devices.device_info.OQTOPUS_AVAILABLE', False):
            device = DeviceInfo()

            assert device.device_name == "anemone"
            assert device._device_data is None
            assert device._device_info is None
            assert device._qubits_df is None
            assert device._couplings_df is None
            assert device.console is not None

    def test_init_custom_device(self):
        """Test initialization with custom device name"""
        with patch('oqtopus_experiments.devices.device_info.OQTOPUS_AVAILABLE', False):
            device = DeviceInfo("custom_device")

            assert device.device_name == "custom_device"
            assert device._device_data is None
            assert device._device_info is None
            assert device._qubits_df is None
            assert device._couplings_df is None

    def test_available_property_no_device_data(self):
        """Test available property when no device data is loaded"""
        with patch('oqtopus_experiments.devices.device_info.OQTOPUS_AVAILABLE', False):
            device = DeviceInfo("anemone")
            assert device.available is False

    def test_available_property_with_device_data(self):
        """Test available property when device data is loaded"""
        with patch('oqtopus_experiments.devices.device_info.OQTOPUS_AVAILABLE', False):
            device = DeviceInfo("anemone")
            device._device_data = MagicMock()  # Mock device data
            assert device.available is True

    def test_device_info_property(self):
        """Test device_info property"""
        with patch('oqtopus_experiments.devices.device_info.OQTOPUS_AVAILABLE', False):
            device = DeviceInfo("anemone")
            device._device_info = {"name": "test", "qubits": 16}

            info = device.device_info
            assert info == {"name": "test", "qubits": 16}

    def test_device_info_property_none(self):
        """Test device_info property when None"""
        with patch('oqtopus_experiments.devices.device_info.OQTOPUS_AVAILABLE', False):
            device = DeviceInfo("anemone")
            device._device_info = None

            info = device.device_info
            assert info is None

    def test_create_dataframes_success(self):
        """Test successful DataFrame creation"""
        with patch('oqtopus_experiments.devices.device_info.OQTOPUS_AVAILABLE', False):
            device = DeviceInfo("anemone")

            # Mock device info with qubit and coupling data
            mock_device_info = {
                "qubits": [
                    {
                        "id": 0, "physical_id": 0, "fidelity": 0.99,
                        "position": {"x": 0, "y": 0},
                        "qubit_lifetime": {"t1": 50.0, "t2": 100.0},
                        "meas_error": {"readout_assignment_error": 0.01}
                    },
                    {
                        "id": 1, "physical_id": 1, "fidelity": 0.98,
                        "position": {"x": 1, "y": 0},
                        "qubit_lifetime": {"t1": 45.0, "t2": 90.0},
                        "meas_error": {"readout_assignment_error": 0.02}
                    }
                ],
                "couplings": [
                    {
                        "control": 0, "target": 1, "fidelity": 0.95,
                        "gate_duration": {"rzx90": 50}
                    }
                ]
            }
            device._device_info = mock_device_info

            device._create_dataframes()

            assert device._qubits_df is not None
            assert len(device._qubits_df) == 2
            assert "id" in device._qubits_df.columns
            assert "fidelity" in device._qubits_df.columns
            assert "t1" in device._qubits_df.columns
            assert "t2" in device._qubits_df.columns

            assert device._couplings_df is not None
            assert len(device._couplings_df) == 1
            assert "control" in device._couplings_df.columns
            assert "target" in device._couplings_df.columns
            assert "fidelity" in device._couplings_df.columns

    def test_create_dataframes_no_device_info(self):
        """Test DataFrame creation without device info"""
        with patch('oqtopus_experiments.devices.device_info.OQTOPUS_AVAILABLE', False):
            device = DeviceInfo("anemone")
            device._device_info = None

            device._create_dataframes()

            # Should not crash, but DataFrames remain None

    def test_qubits_property(self):
        """Test qubits property returns DataFrame"""
        with patch('oqtopus_experiments.devices.device_info.OQTOPUS_AVAILABLE', False):
            device = DeviceInfo("anemone")
            device._qubits_df = pd.DataFrame([{"id": 0, "fidelity": 0.99}])

            qubits = device.qubits
            assert qubits is not None
            assert len(qubits) == 1

    def test_couplings_property(self):
        """Test couplings property returns DataFrame"""
        with patch('oqtopus_experiments.devices.device_info.OQTOPUS_AVAILABLE', False):
            device = DeviceInfo("anemone")
            device._couplings_df = pd.DataFrame([{"control": 0, "target": 1, "fidelity": 0.95}])

            couplings = device.couplings
            assert couplings is not None
            assert len(couplings) == 1

    def test_get_best_qubits_success(self):
        """Test getting best qubits by fidelity"""
        with patch('oqtopus_experiments.devices.device_info.OQTOPUS_AVAILABLE', False):
            device = DeviceInfo("anemone")
            device._qubits_df = pd.DataFrame([
                {"id": 0, "physical_id": 0, "fidelity": 0.99},
                {"id": 1, "physical_id": 1, "fidelity": 0.95},
                {"id": 2, "physical_id": 2, "fidelity": 0.97},
                {"id": 3, "physical_id": 3, "fidelity": 0.98}
            ])

            best_qubits = device.get_best_qubits(2)

            assert len(best_qubits) == 2
            assert best_qubits.iloc[0]["id"] == 0  # Highest fidelity
            assert best_qubits.iloc[1]["id"] == 3  # Second highest fidelity

    def test_get_best_qubits_no_dataframe(self):
        """Test getting best qubits when no DataFrame available"""
        with patch('oqtopus_experiments.devices.device_info.OQTOPUS_AVAILABLE', False):
            device = DeviceInfo("anemone")
            device._qubits_df = None

            best_qubits = device.get_best_qubits(5)
            assert best_qubits is None

    def test_get_best_qubits_more_than_available(self):
        """Test getting more qubits than available"""
        with patch('oqtopus_experiments.devices.device_info.OQTOPUS_AVAILABLE', False):
            device = DeviceInfo("anemone")
            device._qubits_df = pd.DataFrame([
                {"id": 0, "physical_id": 0, "fidelity": 0.99},
                {"id": 1, "physical_id": 1, "fidelity": 0.95}
            ])

            best_qubits = device.get_best_qubits(5)  # Request more than available

            assert len(best_qubits) == 2  # Should return all available

    def test_get_worst_qubits_success(self):
        """Test getting worst qubits by fidelity"""
        with patch('oqtopus_experiments.devices.device_info.OQTOPUS_AVAILABLE', False):
            device = DeviceInfo("anemone")
            device._qubits_df = pd.DataFrame([
                {"id": 0, "fidelity": 0.99},
                {"id": 1, "fidelity": 0.95},
                {"id": 2, "fidelity": 0.97},
                {"id": 3, "fidelity": 0.98}
            ])

            worst_qubits = device.get_worst_qubits(2)

            assert len(worst_qubits) == 2
            assert worst_qubits.iloc[0]["fidelity"] == 0.95  # Lowest fidelity
            assert worst_qubits.iloc[1]["fidelity"] == 0.97  # Second lowest fidelity

    def test_get_qubit_stats_success(self):
        """Test getting qubit statistics"""
        with patch('oqtopus_experiments.devices.device_info.OQTOPUS_AVAILABLE', False):
            device = DeviceInfo("anemone")
            device._qubits_df = pd.DataFrame([
                {"fidelity": 0.99, "t1": 50.0, "t2": 100.0, "readout_error": 0.01},
                {"fidelity": 0.95, "t1": 45.0, "t2": 90.0, "readout_error": 0.02}
            ])

            stats = device.get_qubit_stats()

            assert stats is not None
            assert "fidelity" in stats
            assert "t1" in stats
            assert "t2" in stats
            assert "readout_error" in stats
            assert "mean" in stats["fidelity"]
            assert "std" in stats["fidelity"]

    def test_get_qubit_stats_no_dataframe(self):
        """Test getting qubit statistics when no DataFrame available"""
        with patch('oqtopus_experiments.devices.device_info.OQTOPUS_AVAILABLE', False):
            device = DeviceInfo("anemone")
            device._qubits_df = None

            stats = device.get_qubit_stats()
            assert stats is None

    def test_summary_no_device_data(self):
        """Test summary when no device data available"""
        with patch('oqtopus_experiments.devices.device_info.OQTOPUS_AVAILABLE', False):
            device = DeviceInfo("anemone")
            device._device_data = None

            summary = device.summary()
            assert "error" in summary

    def test_summary_with_device_data(self):
        """Test summary with device data"""
        with patch('oqtopus_experiments.devices.device_info.OQTOPUS_AVAILABLE', False):
            device = DeviceInfo("anemone")

            mock_device_data = MagicMock()
            mock_device_data.device_id = "anemone"
            mock_device_data.description = "Test device"
            mock_device_data.device_type = "quantum"
            mock_device_data.status = "available"
            mock_device_data.n_qubits = 32
            mock_device_data.n_pending_jobs = 0
            mock_device_data.basis_gates = ["sx", "x", "rz", "cx"]
            mock_device_data.calibrated_at = "2024-01-01"

            device._device_data = mock_device_data

            summary = device.summary()

            assert summary["device_id"] == "anemone"
            assert summary["n_qubits"] == 32
            assert summary["status"] == "available"

    def test_show_device_unavailable(self):
        """Test show method when device is unavailable"""
        with patch('oqtopus_experiments.devices.device_info.OQTOPUS_AVAILABLE', False):
            device = DeviceInfo("anemone")
            device._device_data = None  # Make device unavailable

            with patch.object(device.console, 'print') as mock_print:
                device.show()
                mock_print.assert_called_with("[red]❌ Device information not available[/red]")

    def test_show_device_available(self):
        """Test show method when device is available"""
        with patch('oqtopus_experiments.devices.device_info.OQTOPUS_AVAILABLE', False):
            device = DeviceInfo("anemone")

            # Mock device data
            mock_device_data = MagicMock()
            mock_device_data.device_id = "anemone"
            mock_device_data.description = "Test device"
            mock_device_data.device_type = "quantum"
            mock_device_data.status = "available"
            mock_device_data.n_qubits = 32
            mock_device_data.n_pending_jobs = 0
            mock_device_data.basis_gates = ["sx", "x", "rz", "cx"]
            mock_device_data.calibrated_at = "2024-01-01"

            device._device_data = mock_device_data
            # Create complete DataFrame with all required columns
            device._qubits_df = pd.DataFrame([{
                "id": 0, "physical_id": 0, "x": 0, "y": 0, "fidelity": 0.99,
                "t1": 50.0, "t2": 100.0, "readout_error": 0.01
            }])
            device._device_info = {"couplings": []}

            with patch.object(device.console, 'print') as mock_print:
                device.show()
                # Should print device information multiple times
                assert mock_print.call_count > 0

    def test_plot_layout_device_unavailable(self):
        """Test plot_layout when device is unavailable"""
        with patch('oqtopus_experiments.devices.device_info.OQTOPUS_AVAILABLE', False):
            device = DeviceInfo("anemone")
            device._device_data = None  # Make device unavailable

            with patch('builtins.print') as mock_print:
                device.plot_layout()
                mock_print.assert_called_with("❌ Device information not available for plotting")

    @patch('oqtopus_experiments.devices.device_info.go.Figure')
    @patch('oqtopus_experiments.devices.device_info.pio')
    def test_plot_layout_device_available(self, mock_pio, mock_figure):
        """Test plot_layout when device is available"""
        with patch('oqtopus_experiments.devices.device_info.OQTOPUS_AVAILABLE', False):
            device = DeviceInfo("anemone")
            device._device_data = MagicMock()  # Make device available
            # Create complete DataFrame with all required columns
            device._qubits_df = pd.DataFrame([
                {"id": 0, "physical_id": 0, "x": 0, "y": 0, "fidelity": 0.99, "t1": 50.0, "t2": 100.0, "readout_error": 0.01},
                {"id": 1, "physical_id": 1, "x": 1, "y": 0, "fidelity": 0.95, "t1": 45.0, "t2": 90.0, "readout_error": 0.02}
            ])
            device._device_info = {
                "couplings": [
                    {"control": 0, "target": 1, "fidelity": 0.95}
                ]
            }

            mock_fig = MagicMock()
            mock_figure.return_value = mock_fig

            device.plot_layout()

            mock_figure.assert_called_once()
            mock_fig.show.assert_called_once()

    def test_compare_qubits_success(self):
        """Test comparing specific qubits"""
        with patch('oqtopus_experiments.devices.device_info.OQTOPUS_AVAILABLE', False):
            device = DeviceInfo("anemone")
            device._qubits_df = pd.DataFrame([
                {"id": 0, "fidelity": 0.99},
                {"id": 1, "fidelity": 0.95},
                {"id": 2, "fidelity": 0.97}
            ])

            comparison = device.compare_qubits([0, 2])

            assert len(comparison) == 2
            assert 0 in comparison["id"].values
            assert 2 in comparison["id"].values
            assert 1 not in comparison["id"].values

    def test_compare_qubits_no_dataframe(self):
        """Test comparing qubits when no DataFrame available"""
        with patch('oqtopus_experiments.devices.device_info.OQTOPUS_AVAILABLE', False):
            device = DeviceInfo("anemone")
            device._qubits_df = None

            comparison = device.compare_qubits([0, 1])
            assert comparison is None

    def test_save_data_unavailable(self):
        """Test saving data when device is unavailable"""
        with patch('oqtopus_experiments.devices.device_info.OQTOPUS_AVAILABLE', False):
            device = DeviceInfo("anemone")
            device._device_data = None

            result = device.save_data()
            assert "❌ No device data available to save" in result

    @patch('builtins.open', create=True)
    @patch('json.dump')
    def test_save_data_success(self, mock_json_dump, mock_open):
        """Test successful data saving"""
        with patch('oqtopus_experiments.devices.device_info.OQTOPUS_AVAILABLE', False):
            device = DeviceInfo("anemone")
            device._device_data = MagicMock()
            device._qubits_df = pd.DataFrame([{"id": 0, "fidelity": 0.99}])
            device._couplings_df = pd.DataFrame([{"control": 0, "target": 1, "fidelity": 0.95}])

            with patch.object(device, 'summary', return_value={"device_id": "anemone"}):
                with patch.object(device, 'get_qubit_stats', return_value={"mean": 0.97}):
                    with patch('builtins.print'):
                        result = device.save_data("test.json")

                        mock_open.assert_called_once_with("test.json", "w")
                        mock_json_dump.assert_called_once()
                        assert result == "test.json"


if __name__ == "__main__":
    pytest.main([__file__])
