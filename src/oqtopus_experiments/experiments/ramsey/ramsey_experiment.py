#!/usr/bin/env python3
"""
Ramsey Experiment Class - Ramsey oscillation experiment specialized class
Inherits from BaseExperiment and provides Ramsey experiment-specific implementation
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np

from ...core.base_experiment import BaseExperiment
from ...core.parallel_execution import ParallelExecutionMixin


class RamseyExperiment(BaseExperiment, ParallelExecutionMixin):
    """
    Ramsey oscillation experiment class

    Specialized features:
    - Automatic Ramsey circuit generation
    - Oscillation fitting
    - Delay time scan experiments
    - T2* time constant estimation
    """

    def __init__(
        self, experiment_name: str = None, enable_fitting: bool = True, **kwargs
    ):
        # Extract Ramsey experiment-specific parameters (not passed to BaseExperiment)
        ramsey_specific_params = {
            "delay_points",
            "max_delay",
            "detuning",
            "delay_times",
            "enable_fitting",
        }

        # Filter kwargs to pass to BaseExperiment
        base_kwargs = {
            k: v for k, v in kwargs.items() if k not in ramsey_specific_params
        }

        super().__init__(experiment_name, **base_kwargs)

        # Ramsey experiment-specific settings
        self.expected_t2_star = (
            1000  # Initial estimate [ns] - used only for fitting initial values
        )
        self.expected_detuning = 0.0  # Expected detuning [MHz]
        self.enable_fitting = enable_fitting  # Fitting enable flag

        # Enable readout mitigation for Ramsey experiments
        self.mitigation_options = {"ro_error_mitigation": "pseudo_inverse"}
        self.mitigation_info = self.mitigation_options

        if enable_fitting:
            print("Ramsey experiment: Standard Ramsey measurement with fitting enabled")
        else:
            print("Ramsey experiment: Standard Ramsey measurement (fitting disabled)")

    def create_circuits(self, **kwargs) -> list[Any]:
        """
        Create Ramsey experiment circuits

        Args:
            delay_points: Number of delay time points (default: 51)
            max_delay: Maximum delay time [ns] (default: 50000)
            detuning: Frequency detuning [MHz] (default: 0.0)
            delay_times: Directly specified delay time list [ns] (optional)

        Returns:
            Ramsey circuit list
        """
        delay_points = kwargs.get("delay_points", 51)
        max_delay = kwargs.get("max_delay", 200000)
        detuning = kwargs.get("detuning", 0.0)

        # Delay time range
        if "delay_times" in kwargs:
            delay_times = np.array(kwargs["delay_times"])
        else:
            # Default: 51 points on logarithmic scale from 50ns to 200μs
            delay_times = np.logspace(np.log10(50), np.log10(200 * 1000), num=51)
            if delay_points != 51:
                delay_times = np.linspace(50, max_delay, delay_points)

        # Save metadata
        self.experiment_params = {
            "delay_times": delay_times.tolist(),
            "delay_points": len(delay_times),
            "max_delay": max_delay,
            "detuning": detuning,
        }

        # Create Ramsey circuits
        circuits = []
        for delay_time in delay_times:
            circuit = self._create_single_ramsey_circuit(delay_time, detuning)
            circuits.append(circuit)

        print(
            f"Ramsey circuits: Delay range {len(delay_times)} points from {delay_times[0]:.1f} to {delay_times[-1]:.1f} ns, detuning={detuning} MHz"
        )

        return circuits

    def run_ramsey_experiment_parallel(
        self,
        devices: list[str] = ["qulacs"],
        shots: int = 1024,
        parallel_workers: int = 4,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Parallel execution of Ramsey experiment (preserving delay time order)
        """
        print(f"🔬 Running Ramsey experiment with {parallel_workers} parallel workers")

        # Create circuits
        circuits = self.create_circuits(**kwargs)
        delay_times = self.experiment_params["delay_times"]

        print(
            f"   📊 {len(circuits)} circuits × {len(devices)} devices = {len(circuits) * len(devices)} jobs"
        )

        # Parallel execution (preserving order)
        job_data = self._submit_ramsey_circuits_parallel_with_order(
            circuits, devices, shots, parallel_workers
        )

        # Collect results (preserving order)
        raw_results = self._collect_ramsey_results_parallel_with_order(
            job_data, parallel_workers
        )

        # Analyze results (with error handling)
        try:
            analysis = self.analyze_results(raw_results)
        except Exception as e:
            print(f"Analysis failed: {e}, creating minimal analysis")
            analysis = {
                "experiment_info": {"delay_points": len(delay_times), "error": str(e)},
                "device_results": {},
            }

        return {
            "delay_times": delay_times,
            "device_results": analysis["device_results"],
            "analysis": analysis,
            "method": "ramsey_parallel_quantumlib",
        }

    def _submit_ramsey_circuits_parallel_with_order(
        self, circuits: list[Any], devices: list[str], shots: int, parallel_workers: int
    ) -> dict[str, list[dict]]:
        """
        Parallel submission of Ramsey circuits (using ParallelExecutionMixin)
        """
        print(f"Enhanced Ramsey parallel submission: {parallel_workers} workers")

        if not self.oqtopus_available:
            return self._submit_ramsey_circuits_locally_parallel(
                circuits, devices, shots, parallel_workers
            )

        # Use ParallelExecutionMixin for parallel execution
        def submit_single_ramsey_circuit(device, circuit, shots, circuit_idx):
            """Submit a single Ramsey circuit"""
            try:
                job_id = self.submit_circuit_to_oqtopus(circuit, shots, device)
                if job_id:
                    return {
                        "job_id": job_id,
                        "device": device,
                        "circuit_idx": circuit_idx,
                        "shots": shots,
                        "submitted": True,
                        "submission_time": time.time(),
                    }
                else:
                    return None
            except Exception as e:
                delay_time = self.experiment_params["delay_times"][circuit_idx]
                print(
                    f"Ramsey Circuit {circuit_idx} (τ={delay_time:.0f}ns) → {device}: {e}"
                )
                return None

        return self.submit_circuits_parallel_with_order(
            circuits=circuits,
            devices=devices,
            shots=shots,
            parallel_workers=parallel_workers,
            submit_function=submit_single_ramsey_circuit,
            progress_name="Ramsey Submission",
        )

    def _submit_ramsey_circuits_locally_parallel(
        self, circuits: list[Any], devices: list[str], shots: int, parallel_workers: int
    ) -> dict[str, list[dict]]:
        """Parallel execution of Ramsey circuits on local simulator"""
        print(f"Ramsey Local parallel execution: {parallel_workers} workers")

        all_job_data: dict[str, list[dict[str, Any] | None]] = {
            device: [None] * len(circuits) for device in devices
        }

        circuit_device_pairs = []
        for circuit_idx, circuit in enumerate(circuits):
            for device in devices:
                circuit_device_pairs.append((circuit_idx, circuit, device))

        def run_single_ramsey_circuit_locally(args):
            circuit_idx, circuit, device = args
            try:
                result = self.run_circuit_locally(circuit, shots)
                if result:
                    job_id = result["job_id"]
                    if not hasattr(self, "_local_results"):
                        self._local_results = {}
                    self._local_results[job_id] = result
                    return device, job_id, circuit_idx, True
                else:
                    return device, None, circuit_idx, False
            except Exception as e:
                delay_time = self.experiment_params["delay_times"][circuit_idx]
                print(
                    f"Local Ramsey circuit {circuit_idx} (τ={delay_time:.0f}ns) → {device}: {e}"
                )
                return device, None, circuit_idx, False

        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = [
                executor.submit(run_single_ramsey_circuit_locally, args)
                for args in circuit_device_pairs
            ]

            for future in as_completed(futures):
                device, job_id, circuit_idx, success = future.result()
                if success and job_id:
                    all_job_data[device][circuit_idx] = {
                        "job_id": job_id,
                        "circuit_index": circuit_idx,
                        "delay_time": self.experiment_params["delay_times"][
                            circuit_idx
                        ],
                        "submitted": True,
                    }
                else:
                    all_job_data[device][circuit_idx] = {
                        "job_id": None,
                        "circuit_index": circuit_idx,
                        "delay_time": self.experiment_params["delay_times"][
                            circuit_idx
                        ],
                        "submitted": False,
                    }

        for device in devices:
            successful = sum(
                1 for job in all_job_data[device] if job and job["submitted"]
            )
            print(
                f"✅ {device}: {successful} Ramsey circuits completed locally (order preserved)"
            )

        return all_job_data

    def _collect_ramsey_results_parallel_with_order(
        self, job_data: dict[str, list[dict]], parallel_workers: int
    ) -> dict[str, list[dict]]:
        """Parallel collection of Ramsey results (preserving order CHSH-style)"""

        # Calculate total jobs and log collection start
        total_jobs_to_collect = sum(
            1
            for device_jobs in job_data.values()
            for job in device_jobs
            if job and job.get("submitted", False)
        )
        print(
            f"📊 Starting Ramsey results collection: {total_jobs_to_collect} jobs from {len(job_data)} devices"
        )

        # Handle local results
        if hasattr(self, "_local_results"):
            print("Using cached local Ramsey simulation results...")
            all_results = {}
            for device, device_job_data in job_data.items():
                device_results = []
                for job_info in device_job_data:
                    if (
                        job_info
                        and job_info["submitted"]
                        and job_info["job_id"] in self._local_results
                    ):
                        result = self._local_results[job_info["job_id"]]
                        device_results.append(result)
                    else:
                        device_results.append(None)
                all_results[device] = device_results
                successful = sum(1 for r in device_results if r is not None)
                print(f"✅ {device}: {successful} Ramsey local results collected")
            return all_results

        if not self.oqtopus_available:
            print("OQTOPUS not available for Ramsey collection")
            return {
                device: [None] * len(device_job_data)
                for device, device_job_data in job_data.items()
            }

        all_results = {
            device: [None] * len(device_job_data)
            for device, device_job_data in job_data.items()
        }

        job_collection_tasks = []
        for device, device_job_data in job_data.items():
            for circuit_idx, job_info in enumerate(device_job_data):
                if job_info and job_info["submitted"] and job_info["job_id"]:
                    job_collection_tasks.append(
                        (job_info["job_id"], device, circuit_idx)
                    )

        def collect_single_ramsey_result(args):
            job_id, device, circuit_idx = args
            try:
                # Poll until job completion
                result = self._poll_job_until_completion(job_id, timeout_minutes=5)
                # Success determination based on OQTOPUS job structure: status == 'succeeded'
                if result and result.get("status") == "succeeded":
                    # Try multiple methods to obtain measurement results
                    counts = None
                    shots = 0

                    # Method 1: When BaseExperiment's get_oqtopus_result directly returns counts
                    if "counts" in result:
                        counts = result["counts"]
                        shots = result.get("shots", 0)

                    # Method 2: Get from result structure within job_info
                    if not counts:
                        job_info = result.get("job_info", {})
                        if isinstance(job_info, dict):
                            # Explore OQTOPUS result structure
                            sampling_result = job_info.get("result", {}).get(
                                "sampling", {}
                            )
                            if sampling_result:
                                counts = sampling_result.get("counts", {})

                    # Method 3: When job_info itself is in result format
                    if not counts and "job_info" in result:
                        job_info = result["job_info"]
                        if isinstance(job_info, dict) and "job_info" in job_info:
                            inner_job_info = job_info["job_info"]
                            if isinstance(inner_job_info, dict):
                                result_data = inner_job_info.get("result", {})
                                if "sampling" in result_data:
                                    counts = result_data["sampling"].get("counts", {})
                                elif "counts" in result_data:
                                    counts = result_data["counts"]

                    if counts:
                        # Convert successful data to standard format
                        processed_result = {
                            "success": True,
                            "counts": dict(counts),  # Convert Counter to dictionary
                            "status": result.get("status"),
                            "execution_time": result.get("execution_time", 0),
                            "shots": shots or sum(counts.values()) if counts else 0,
                        }
                        return device, processed_result, job_id, circuit_idx, True
                    else:
                        delay_time = self.experiment_params["delay_times"][circuit_idx]
                        print(
                            f"⚠️ {device}[{circuit_idx}] (τ={delay_time:.0f}ns): {job_id[:8]}... no measurement data"
                        )
                        return device, None, job_id, circuit_idx, False
                else:
                    # When job failed
                    delay_time = self.experiment_params["delay_times"][circuit_idx]
                    status = result.get("status", "unknown") if result else "no_result"
                    # Display more detailed failure information
                    message = ""
                    if result:
                        job_info = result.get("job_info", {})
                        message = job_info.get("message", "")
                        if message:
                            message = f" - {message}"
                    print(
                        f"⚠️ {device}[{circuit_idx}] (τ={delay_time:.0f}ns): {job_id[:8]}... {status}{message}"
                    )
                    return device, None, job_id, circuit_idx, False
            except Exception as e:
                delay_time = self.experiment_params["delay_times"][circuit_idx]
                print(
                    f"❌ {device}[{circuit_idx}] (τ={delay_time:.0f}ns): {job_id[:8]}... error: {str(e)[:50]}"
                )
                return device, None, job_id, circuit_idx, False

        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = [
                executor.submit(collect_single_ramsey_result, args)
                for args in job_collection_tasks
            ]

            completed_jobs = 0
            successful_jobs = 0
            total_jobs = len(futures)
            last_progress_percent = 0

            for future in as_completed(futures):
                device, result, job_id, circuit_idx, success = future.result()
                completed_jobs += 1

                if success and result:
                    successful_jobs += 1
                    all_results[device][circuit_idx] = result
                    delay_time = self.experiment_params["delay_times"][circuit_idx]
                    print(
                        f"✅ {device}[{circuit_idx}] (τ={delay_time:.0f}ns): {job_id[:8]}... collected ({completed_jobs}/{total_jobs})"
                    )
                else:
                    # Failure cases already logged in individual methods
                    pass

                # Display progress summary every 20%
                progress_percent = (completed_jobs * 100) // total_jobs
                if (
                    progress_percent >= last_progress_percent + 20
                    and progress_percent < 100
                ):
                    print(
                        f"📈 Ramsey Collection Progress: {completed_jobs}/{total_jobs} ({progress_percent}%) - {successful_jobs} successful"
                    )
                    last_progress_percent = progress_percent

        # Final results summary
        total_successful = sum(
            1
            for device_results in all_results.values()
            for r in device_results
            if r is not None
        )
        total_attempted = sum(
            1
            for device_jobs in job_data.values()
            for job in device_jobs
            if job and job.get("submitted", False)
        )
        success_rate = (
            (total_successful / total_attempted * 100) if total_attempted > 0 else 0
        )

        print(
            f"🎉 Ramsey Collection Complete: {total_successful}/{total_attempted} successful ({success_rate:.1f}%)"
        )

        # Display result statistics and report failed jobs
        for device in job_data.keys():
            successful = sum(1 for r in all_results[device] if r is not None)
            total = len(job_data[device])
            failed = total - successful

            if failed > 0:
                device_success_rate = (successful / total * 100) if total > 0 else 0
                print(
                    f"✅ {device}: {successful}/{total} Ramsey results collected (success rate: {device_success_rate:.1f}%)"
                )
                print(
                    f"   ⚠️ {failed} jobs failed - analysis will continue with available data"
                )
            else:
                print(
                    f"✅ {device}: {successful}/{total} Ramsey results collected (100% success)"
                )

        return all_results

    def _poll_job_until_completion(
        self, job_id: str, timeout_minutes: int = 5, poll_interval: float = 2.0
    ):
        """
        Poll until job completion

        Args:
            job_id: Job ID
            timeout_minutes: Timeout duration (minutes)
            poll_interval: Polling interval (seconds)

        Returns:
            Completed job result, or None
        """
        import time

        timeout_seconds = timeout_minutes * 60
        start_time = time.time()
        last_status = None

        while time.time() - start_time < timeout_seconds:
            try:
                result = self.get_oqtopus_result(
                    job_id, timeout_minutes=1, verbose_log=False
                )  # 短いタイムアウトで取得

                if not result:
                    time.sleep(poll_interval)
                    continue

                status = result.get("status", "unknown")

                # 状態が変わった場合のみログ出力（進捗状態のみ）
                if status != last_status:
                    if status in ["running", "submitted", "pending"]:
                        print(f"⏳ {job_id[:8]}... {status}")
                    elif status in ["succeeded", "failed", "cancelled"]:
                        print(f"🏁 {job_id[:8]}... {status}")
                    last_status = status

                # 終了状態をチェック
                if status in ["succeeded", "failed", "cancelled"]:
                    return result
                elif status in ["running", "submitted", "pending"]:
                    # Still running - continue
                    time.sleep(poll_interval)
                    continue
                else:
                    # 不明な状態 - 少し待ってリトライ
                    time.sleep(poll_interval)
                    continue

            except Exception:
                # 一時的なエラーの場合はリトライ
                time.sleep(poll_interval)
                continue

        # タイムアウト
        print(f"⏰ Job {job_id[:8]}... timed out after {timeout_minutes} minutes")
        return None

    def run_experiment(
        self,
        devices: list[str] = ["qulacs"],
        shots: int = 1024,
        parallel_workers: int = 4,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Ramsey実験実行（base_cliの統一フローに従う）
        """
        # base_cliが直接並列メソッドを呼び出すため、ここでは基本的な結果収集のみ
        print("⚠️ run_experiment called directly - use CLI framework instead")
        return self.run_ramsey_experiment_parallel(
            devices=devices, shots=shots, parallel_workers=parallel_workers, **kwargs
        )

    def _create_single_ramsey_circuit(self, delay_time: float, detuning: float = 0.0):
        """
        単一Ramsey回路作成（Qiskitスタイル）

        Args:
            delay_time: 遅延時間 [ns]
            detuning: 周波数デチューニング [MHz] (default: 0.0)

        Ramsey sequence: H - delay - Rz(φ) - H - measure
        where φ = 2π × detuning [MHz] × delay_time [ns] × 1e-3
        """
        try:
            from qiskit import QuantumCircuit
        except ImportError:
            raise ImportError("Qiskit is required for circuit creation") from None

        # 1量子ビット + 1測定ビット（Qiskitスタイル）
        circuit = QuantumCircuit(1, 1)

        # First π/2 pulse (Hadamard gate - Qiskitスタイル)
        circuit.h(0)

        # 遅延時間の間自由進化
        if delay_time > 0:
            circuit.delay(int(delay_time), 0, unit="ns")

        # デチューニング効果（z軸回転）
        if detuning != 0.0:
            # φ = 2π × detuning [MHz] × delay_time [ns] × 1e-9 [s/ns] × 1e6 [Hz/MHz]
            # φ = 2π × detuning × delay_time × 1e-3
            phase = 2 * np.pi * detuning * delay_time * 1e-3
            circuit.rz(phase, 0)

        # Second π/2 pulse (analysis pulse - Hadamard)
        circuit.h(0)

        # Z基底測定
        circuit.measure(0, 0)

        return circuit

    def analyze_results(
        self, results: dict[str, list[dict[str, Any]]], **kwargs
    ) -> dict[str, Any]:
        """
        Ramsey実験結果解析
        """
        if not results:
            return {"error": "No results to analyze"}

        delay_times = np.array(self.experiment_params["delay_times"])

        analysis = {
            "experiment_info": {
                "delay_points": len(delay_times),
                "expected_t2_star": self.expected_t2_star,
                "detuning": self.experiment_params.get("detuning", 0.0),
            },
            "device_results": {},
        }

        for device, device_results in results.items():
            if not device_results:
                continue

            device_analysis = self._analyze_device_results(device_results, delay_times)
            analysis["device_results"][device] = device_analysis

            # フィッティングが有効な場合のみ実行
            if self.enable_fitting:
                try:
                    t2_star_fitted, detuning_fitted, fitting_quality = (
                        self._estimate_ramsey_params_with_quality(
                            device_analysis["p1_values"], delay_times
                        )
                    )
                    quality_str = f"({fitting_quality['method']}, R²={fitting_quality['r_squared']:.3f})"
                    print(
                        f"{device}: T2* = {t2_star_fitted:.1f} ns, detuning = {detuning_fitted:.3f} MHz {quality_str} [with RO mitigation]"
                    )
                except Exception as e:
                    print(f"Fitting error for {device}: {e}, using default values")
                    t2_star_fitted, detuning_fitted, fitting_quality = (
                        float(self.expected_t2_star),
                        0.0,
                        {
                            "method": "error_fallback",
                            "r_squared": 0.0,
                            "error": "exception",
                        },
                    )
            else:
                # フィッティングなし：統計情報のみ表示
                t2_star_fitted, detuning_fitted, fitting_quality = (
                    0.0,
                    0.0,
                    {"method": "no_fitting", "r_squared": 0.0, "error": "disabled"},
                )
                stats = device_analysis["statistics"]
                oscillation_amp = stats.get("oscillation_amplitude", 0.0)
                print(
                    f"{device}: Raw data oscillation amplitude = {oscillation_amp:.3f} [with RO mitigation]"
                )

            analysis["device_results"][device]["t2_star_fitted"] = t2_star_fitted
            analysis["device_results"][device]["detuning_fitted"] = detuning_fitted
            analysis["device_results"][device]["fitting_quality"] = fitting_quality

        return analysis

    def _analyze_device_results(
        self, device_results: list[dict[str, Any]], delay_times: np.ndarray
    ) -> dict[str, Any]:
        """
        単一デバイス結果解析
        """
        p0_values = []
        p1_values = []

        for _i, result in enumerate(device_results):
            if result and result.get("success", False):
                counts = result.get("counts", {})
                if counts:
                    p0 = self._calculate_p0_probability(counts)
                    p0_values.append(p0)
                    p1_values.append(1.0 - p0)  # P(1) = 1 - P(0)
                else:
                    p0_values.append(np.nan)
                    p1_values.append(np.nan)
            else:
                p0_values.append(np.nan)
                p1_values.append(np.nan)

        np.array([p for p in p0_values if not np.isnan(p)])
        valid_p1s = np.array([p for p in p1_values if not np.isnan(p)])

        total_jobs = len(p0_values)
        successful_jobs = len(valid_p1s)
        failed_jobs = total_jobs - successful_jobs

        return {
            "p0_values": p0_values,
            "p1_values": p1_values,
            "delay_times": delay_times.tolist(),
            "statistics": {
                "initial_p1": float(valid_p1s[0]) if len(valid_p1s) > 0 else 0.5,
                "final_p1": float(valid_p1s[-1]) if len(valid_p1s) > 0 else 0.5,
                "success_rate": successful_jobs / total_jobs if total_jobs > 0 else 0,
                "successful_jobs": successful_jobs,
                "failed_jobs": failed_jobs,
                "total_jobs": total_jobs,
                "oscillation_amplitude": (
                    float(max(valid_p1s) - min(valid_p1s))
                    if len(valid_p1s) > 1
                    else 0.0
                ),
            },
        }

    def _calculate_p1_probability(self, counts: dict[str, int]) -> float:
        """
        P(1)確率計算（OQTOPUS 10進数counts対応）
        """
        # OQTOPUSの10進数countsを2進数形式に変換
        binary_counts = self._convert_decimal_to_binary_counts(counts)

        total = sum(binary_counts.values())
        if total == 0:
            return 0.0

        # デバッグ情報表示（初回のみ）
        if not hasattr(self, "_counts_debug_shown"):
            print(f"🔍 Raw decimal counts: {dict(counts)}")
            print(f"🔍 Converted binary counts: {dict(binary_counts)}")
            self._counts_debug_shown = True

        # 標準的なP(1)確率計算
        n_1 = binary_counts.get("1", 0)
        p1 = n_1 / total
        return p1

    def _calculate_p0_probability(self, counts: dict[str, int]) -> float:
        """
        P(0)確率計算（Ramsey実験用 - 物理的に正しい表現）

        Ramsey実験では基底状態|0⟩への確率的緩和を観測するため、
        P(0) = (1 - cos(φ)) × exp(-t/T2*) / 2 が期待される。
        """
        # OQTOPUSの10進数countsを2進数形式に変換
        binary_counts = self._convert_decimal_to_binary_counts(counts)

        total = sum(binary_counts.values())
        if total == 0:
            return 0.5  # デフォルト値（完全混合状態）

        n_0 = binary_counts.get("0", 0)
        p0 = n_0 / total
        return p0

    def _convert_decimal_to_binary_counts(
        self, decimal_counts: dict[str, int]
    ) -> dict[str, int]:
        """
        OQTOPUSの10進数countsを2進数形式に変換

        1量子ビットの場合:
        0 -> "0"  (|0⟩状態)
        1 -> "1"  (|1⟩状態)
        """
        binary_counts = {}

        for decimal_key, count in decimal_counts.items():
            # キーが数値の場合と文字列の場合に対応
            if isinstance(decimal_key, str):
                try:
                    decimal_value = int(decimal_key)
                except ValueError:
                    # すでにバイナリ形式の場合
                    binary_counts[decimal_key] = count
                    continue
            else:
                decimal_value = int(decimal_key)

            # 1量子ビットの場合の変換
            if decimal_value == 0:
                binary_key = "0"
            elif decimal_value == 1:
                binary_key = "1"
            else:
                # 予期しない値の場合はスキップして警告
                print(
                    f"⚠️ Unexpected count key: {decimal_key} (decimal value: {decimal_value})"
                )
                continue

            # 既存のキーがある場合は加算
            if binary_key in binary_counts:
                binary_counts[binary_key] += count
            else:
                binary_counts[binary_key] = count

        return binary_counts

    def _estimate_ramsey_params_with_quality(
        self, p1_values: list[float], delay_times: np.ndarray
    ) -> tuple[float, float, dict[str, Any]]:
        """
        Ramseyパラメータ推定（T2*とデチューニング）
        """
        # NaNを除去
        valid_data = [
            (delay, p1)
            for delay, p1 in zip(delay_times, p1_values, strict=False)
            if not np.isnan(p1)
        ]

        if len(valid_data) < 5:
            return (
                0.0,
                0.0,
                {"method": "insufficient_data", "r_squared": 0.0, "error": "inf"},
            )

        delays = np.array([d[0] for d in valid_data])
        p1s = np.array([d[1] for d in valid_data])

        # detuningに応じてフィッティングモデルを選択
        expected_detuning = self.experiment_params.get("detuning", 0.0)

        try:
            from scipy.optimize import curve_fit

            # detuning=0の場合：純粋なT2*減衰（振動なし）
            if abs(expected_detuning) < 0.001:  # detuning ≈ 0

                def t2_star_decay(t, A, T2_star, offset):
                    return offset - A * np.exp(-t / T2_star)

                # 初期推定値
                p0 = [0.5, self.expected_t2_star, 0.5]  # A, T2*, offset

                # フィッティング実行
                popt, pcov = curve_fit(
                    t2_star_decay,
                    delays,
                    p1s,
                    p0=p0,
                    bounds=([0, 10, 0], [1.0, 100000, 1.0]),
                    maxfev=2000,
                )

                t2_star_fitted = popt[1]
                detuning_fitted = 0.0  # detuning=0として固定

                # 予測値計算とR²算出
                p1_pred = t2_star_decay(delays, *popt)
                ss_res = np.sum((p1s - p1_pred) ** 2)
                ss_tot = np.sum((p1s - np.mean(p1s)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

                # 標準誤差計算
                param_error = "inf"
                if pcov is not None and np.all(np.isfinite(pcov)):
                    param_errors = np.sqrt(np.diag(pcov))
                    t2_error = param_errors[1]
                    param_error = f"{t2_error:.1f}"

                    # 高品質フィッティングの条件
                    if t2_error / t2_star_fitted < 0.5 and r_squared > 0.5:
                        return (
                            float(t2_star_fitted),
                            float(detuning_fitted),
                            {
                                "method": "exponential_decay_t2star",
                                "r_squared": r_squared,
                                "error": param_error,
                                "quality": "high" if r_squared > 0.8 else "medium",
                            },
                        )

            else:  # detuning≠0の場合：振動する減衰

                def ramsey_oscillation(t, A, T2_star, freq, phase, offset):
                    return offset - A * np.exp(-t / T2_star) * np.cos(
                        2 * np.pi * freq * t * 1e-3 + phase
                    )

                # 初期推定値
                p0 = [
                    0.5,
                    self.expected_t2_star,
                    expected_detuning,
                    0.0,
                    0.5,
                ]  # A, T2*, freq, phase, offset

                # フィッティング実行
                popt, pcov = curve_fit(
                    ramsey_oscillation,
                    delays,
                    p1s,
                    p0=p0,
                    bounds=([0, 10, -10, -np.pi, 0], [1.0, 100000, 10, np.pi, 1.0]),
                    maxfev=2000,
                )

                t2_star_fitted = popt[1]
                detuning_fitted = popt[2]

                # 予測値計算とR²算出
                p1_pred = ramsey_oscillation(delays, *popt)
                ss_res = np.sum((p1s - p1_pred) ** 2)
                ss_tot = np.sum((p1s - np.mean(p1s)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

                # 標準誤差計算
                param_error = "inf"
                if pcov is not None and np.all(np.isfinite(pcov)):
                    param_errors = np.sqrt(np.diag(pcov))
                    t2_error = param_errors[1]
                    param_error = f"{t2_error:.1f}"

                    # 高品質フィッティングの条件
                    if t2_error / t2_star_fitted < 0.5 and r_squared > 0.5:
                        return (
                            float(t2_star_fitted),
                            float(detuning_fitted),
                            {
                                "method": "ramsey_oscillation",
                                "r_squared": r_squared,
                                "error": param_error,
                                "quality": "high" if r_squared > 0.8 else "medium",
                            },
                        )

        except (ImportError, RuntimeError, ValueError, TypeError, Exception) as e:
            print(f"Ramsey fitting failed: {str(e)[:50]}... using default values")
            pass

        # 全ての手法が失敗した場合 - デフォルト値を返す
        return (
            float(self.expected_t2_star),
            0.0,
            {
                "method": "default_ramsey",
                "r_squared": 0.0,
                "error": "N/A",
                "quality": "poor",
            },
        )

    def save_experiment_data(
        self, results: dict[str, Any], metadata: dict[str, Any] = None
    ) -> str:
        """
        Ramsey実験データ保存
        """
        ramsey_data = {
            "experiment_type": "Ramsey_Oscillation",
            "experiment_timestamp": time.time(),
            "experiment_parameters": self.experiment_params,
            "analysis_results": results,
            "oqtopus_configuration": {
                "transpiler_options": self.transpiler_options,
                "mitigation_options": self.mitigation_options,
                "basis_gates": self.anemone_basis_gates,
            },
            "metadata": metadata or {},
        }

        # メイン結果保存
        main_file = self.data_manager.save_data(
            ramsey_data, "ramsey_experiment_results"
        )

        # 追加ファイル保存
        if "device_results" in results:
            # デバイス別サマリー
            device_summary = {
                device: {
                    "t2_star_fitted": analysis.get("t2_star_fitted", 0.0),
                    "detuning_fitted": analysis.get("detuning_fitted", 0.0),
                    "statistics": analysis["statistics"],
                }
                for device, analysis in results["device_results"].items()
            }
            self.data_manager.save_data(device_summary, "device_ramsey_summary")

            # P(1)データ（プロット用）
            p1_data = {
                "delay_times": self.experiment_params["delay_times"],
                "device_p1_values": {
                    device: analysis["p1_values"]
                    for device, analysis in results["device_results"].items()
                },
            }
            self.data_manager.save_data(p1_data, "ramsey_p1_values_for_plotting")

        return main_file

    def generate_ramsey_plot(
        self, results: dict[str, Any], save_plot: bool = True, show_plot: bool = False
    ) -> str | None:
        """Generate Ramsey experiment plot"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available - skipping plot generation")
            return None

        delay_times = results.get("delay_times", np.linspace(50, 50000, 51))
        device_results = results.get("device_results", {})

        if not device_results:
            print("No device results for plotting")
            return None

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot experimental data for each device
        colors = ["blue", "red", "green", "orange", "purple"]

        for i, (device, device_data) in enumerate(device_results.items()):
            if "p0_values" in device_data:
                p0_values = device_data["p0_values"]
                t2_star_fitted = device_data.get("t2_star_fitted", 0.0)
                detuning_fitted = device_data.get("detuning_fitted", 0.0)
                fitting_quality = device_data.get("fitting_quality", {})
                r_squared = fitting_quality.get("r_squared", 0.0)
                color = colors[i % len(colors)]

                ax.semilogx(
                    delay_times,
                    p0_values,
                    "o",
                    markersize=4,
                    label=f"{device} data (P0)",
                    alpha=0.8,
                    color=color,
                )

                if self.enable_fitting and t2_star_fitted > 0:
                    fit_delays = np.logspace(
                        np.log10(min(delay_times)), np.log10(max(delay_times)), 200
                    )
                    A = 0.5
                    offset = 0.5

                    fitting_method = fitting_quality.get("method", "unknown")

                    if (
                        fitting_method == "exponential_decay_t2star"
                        or abs(detuning_fitted) < 0.001
                    ):
                        p1_fit_curve = offset - A * np.exp(-fit_delays / t2_star_fitted)
                        label_text = f"{device} fit (T2*={t2_star_fitted:.0f}ns, R²={r_squared:.3f})"
                    else:
                        p1_fit_curve = offset - A * np.exp(
                            -fit_delays / t2_star_fitted
                        ) * np.cos(2 * np.pi * detuning_fitted * fit_delays * 1e-3)
                        label_text = f"{device} fit (T2*={t2_star_fitted:.0f}ns, f={detuning_fitted:.3f}MHz, R²={r_squared:.3f})"

                    p0_fit_curve = 1.0 - p1_fit_curve
                    ax.semilogx(
                        fit_delays,
                        p0_fit_curve,
                        "-",
                        linewidth=2,
                        color=color,
                        alpha=0.7,
                        label=label_text,
                    )

        ax.set_xlabel("Delay time τ [ns] (log scale)", fontsize=14)
        ax.set_ylabel("P(0)", fontsize=14)
        title_suffix = " (with fitting)" if self.enable_fitting else " (raw data)"
        ax.set_title(
            f"OQTOPUS Experiments Ramsey Oscillation Experiment{title_suffix}",
            fontsize=16,
            fontweight="bold",
        )
        ax.grid(True, which="both", ls="--", linewidth=0.5)
        ax.legend(fontsize=12)
        ax.set_ylim(0, 1.1)

        plot_filename = None
        if save_plot:
            plt.tight_layout()
            plot_filename = f"ramsey_plot_{self.experiment_name}_{int(time.time())}.png"

            # Always save to experiment results directory
            if hasattr(self, "data_manager") and hasattr(
                self.data_manager, "session_dir"
            ):
                plot_path = f"{self.data_manager.session_dir}/plots/{plot_filename}"
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                print(f"Plot saved: {plot_path}")
                plot_filename = plot_path
            else:
                plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
                print(f"⚠️ Plot saved to current directory: {plot_filename}")

        if show_plot:
            try:
                plt.show()
            except Exception:
                pass

        plt.close()
        return plot_filename

    def save_complete_experiment_data(self, results: dict[str, Any]) -> str:
        """Save experiment data and generate comprehensive report"""
        # Save main experiment data
        main_file = self.save_experiment_data(results["analysis"])

        # Generate and save plot
        plot_file = self.generate_ramsey_plot(results, save_plot=True, show_plot=False)

        # Create experiment summary
        summary = self._create_ramsey_experiment_summary(results)
        summary_file = self.data_manager.save_data(summary, "experiment_summary")

        print("📊 Complete Ramsey experiment data saved:")
        print(f"  • Main results: {main_file}")
        print(f"  • Plot: {plot_file if plot_file else 'Not generated'}")
        print(f"  • Summary: {summary_file}")

        return main_file

    def _create_ramsey_experiment_summary(
        self, results: dict[str, Any]
    ) -> dict[str, Any]:
        """Create human-readable Ramsey experiment summary"""
        device_results = results.get("device_results", {})
        delay_times = results.get("delay_times", [])

        summary = {
            "experiment_overview": {
                "experiment_name": self.experiment_name,
                "timestamp": time.time(),
                "method": results.get("method", "ramsey_oscillation"),
                "delay_points": len(delay_times),
                "devices_tested": list(device_results.keys()),
            },
            "key_results": {},
            "ramsey_analysis": {
                "expected_t2_star": self.expected_t2_star,
                "clear_oscillation_detected": False,
            },
        }

        # Analyze each device
        min_oscillation_threshold = 0.1  # Minimum oscillation amplitude

        for device, device_data in device_results.items():
            if "p1_values" in device_data:
                p1_values = device_data["p1_values"]
                valid_p1s = [p for p in p1_values if not np.isnan(p)]

                if valid_p1s and len(valid_p1s) >= 5:
                    oscillation_amplitude = max(valid_p1s) - min(valid_p1s)
                    t2_star_fitted = device_data.get("t2_star_fitted", 0.0)
                    detuning_fitted = device_data.get("detuning_fitted", 0.0)

                    summary["key_results"][device] = {
                        "oscillation_amplitude": oscillation_amplitude,
                        "t2_star_fitted": t2_star_fitted,
                        "detuning_fitted": detuning_fitted,
                        "clear_oscillation": oscillation_amplitude
                        > min_oscillation_threshold,
                    }

                    if oscillation_amplitude > min_oscillation_threshold:
                        summary["ramsey_analysis"]["clear_oscillation_detected"] = True

        return summary

    def display_results(self, results: dict[str, Any], use_rich: bool = True) -> None:
        """Display Ramsey experiment results in formatted table"""
        device_results = results.get("device_results", {})

        if not device_results:
            print("No device results found")
            return

        if use_rich:
            try:
                from rich.console import Console
                from rich.table import Table

                console = Console()
                table = Table(
                    title="Ramsey Oscillation Results",
                    show_header=True,
                    header_style="bold blue",
                )
                table.add_column("Device", style="cyan")
                table.add_column("T2* Fitted [ns]", justify="right")
                table.add_column("Detuning [MHz]", justify="right")
                table.add_column("P(0) Oscillation", justify="right")
                table.add_column("Success Rate", justify="right")
                table.add_column("Clear Signal", justify="center")

                for device, device_data in device_results.items():
                    if "p0_values" in device_data:
                        p0_values = device_data["p0_values"]
                        valid_p0s = [p for p in p0_values if not np.isnan(p)]

                        if valid_p0s and len(valid_p0s) >= 2:
                            oscillation_amplitude = max(valid_p0s) - min(valid_p0s)
                            t2_star_fitted = device_data.get("t2_star_fitted", 0.0)
                            detuning_fitted = device_data.get("detuning_fitted", 0.0)

                            stats = device_data.get("statistics", {})
                            success_rate = stats.get("success_rate", 0.0)
                            successful_jobs = stats.get("successful_jobs", 0)
                            total_jobs = stats.get("total_jobs", 0)

                            clear_signal = (
                                "YES" if oscillation_amplitude > 0.1 else "NO"
                            )
                            signal_style = (
                                "green" if oscillation_amplitude > 0.1 else "yellow"
                            )

                            table.add_row(
                                device.upper(),
                                f"{t2_star_fitted:.1f}",
                                f"{detuning_fitted:.3f}",
                                f"{oscillation_amplitude:.3f}",
                                f"{success_rate * 100:.1f}% ({successful_jobs}/{total_jobs})",
                                clear_signal,
                                style=(
                                    signal_style
                                    if oscillation_amplitude > 0.1
                                    else None
                                ),
                            )

                console.print(table)
                console.print(f"\nExpected T2*: {self.expected_t2_star} ns")
                expected_detuning = self.experiment_params.get("detuning", 0.0)
                if abs(expected_detuning) < 0.001:
                    console.print(
                        f"Detuning: {expected_detuning} MHz → Pure T2* decay mode"
                    )
                else:
                    console.print(
                        f"Detuning: {expected_detuning} MHz → Ramsey oscillation mode"
                    )
                fitting_status = "enabled" if self.enable_fitting else "disabled"
                console.print(f"Parameter fitting: {fitting_status}")
                console.print("Clear oscillation threshold: 0.1")

            except ImportError:
                use_rich = False

        if not use_rich:
            # Fallback to simple text display
            print("\n" + "=" * 60)
            print("Ramsey Oscillation Results")
            print("=" * 60)

            for device, device_data in device_results.items():
                if "p0_values" in device_data:
                    p0_values = device_data["p0_values"]
                    valid_p0s = [p for p in p0_values if not np.isnan(p)]

                    if valid_p0s and len(valid_p0s) >= 2:
                        oscillation_amplitude = max(valid_p0s) - min(valid_p0s)
                        t2_star_fitted = device_data.get("t2_star_fitted", 0.0)
                        detuning_fitted = device_data.get("detuning_fitted", 0.0)

                        stats = device_data.get("statistics", {})
                        success_rate = stats.get("success_rate", 0.0)
                        successful_jobs = stats.get("successful_jobs", 0)
                        total_jobs = stats.get("total_jobs", 0)

                        clear_signal = "YES" if oscillation_amplitude > 0.1 else "NO"

                        print(f"Device: {device.upper()}")
                        print(f"  T2* Fitted: {t2_star_fitted:.1f} ns")
                        print(f"  Detuning: {detuning_fitted:.3f} MHz")
                        print(f"  P(0) Oscillation: {oscillation_amplitude:.3f}")
                        print(
                            f"  Success Rate: {success_rate * 100:.1f}% ({successful_jobs}/{total_jobs})"
                        )
                        print(f"  Clear Signal: {clear_signal}")
                        print()

            print(f"Expected T2*: {self.expected_t2_star} ns")
            expected_detuning = self.experiment_params.get("detuning", 0.0)
            if abs(expected_detuning) < 0.001:
                print(f"Detuning: {expected_detuning} MHz → Pure T2* decay mode")
            else:
                print(f"Detuning: {expected_detuning} MHz → Ramsey oscillation mode")
            fitting_status = "enabled" if self.enable_fitting else "disabled"
            print(f"Parameter fitting: {fitting_status}")
            print("Clear oscillation threshold: 0.1")
            print("=" * 60)
