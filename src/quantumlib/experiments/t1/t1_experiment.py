#!/usr/bin/env python3
"""
T1 Experiment Class - T1減衰実験専用クラス
BaseExperimentを継承し、T1実験に特化した実装を提供
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import numpy as np

from ...circuit.t1_circuits import (
    create_multiple_t1_circuits,
)
from ...core.base_experiment import BaseExperiment


class T1Experiment(BaseExperiment):
    """
    T1減衰実験クラス

    特化機能:
    - T1減衰回路の自動生成
    - 指数減衰フィッティング
    - 遅延時間スキャン実験
    - T1時定数推定
    """

    def __init__(self, experiment_name: str = None, disable_mitigation: bool = False, **kwargs):
        # T1実験固有のパラメータを抽出（BaseExperimentには渡さない）
        t1_specific_params = {"delay_points", "max_delay", "delay_times", "disable_mitigation"}

        # BaseExperimentに渡すkwargsをフィルタリング
        base_kwargs = {k: v for k, v in kwargs.items() if k not in t1_specific_params}

        super().__init__(experiment_name, **base_kwargs)

        # T1実験固有の設定（実験値のみ使用、理論値は参考程度）
        self.expected_t1 = 1000  # 初期推定値 [ns] - フィッティングの初期値のみに使用
        self.t1_theoretical = None  # 使用しない
        self.t2_theoretical = None  # 使用しない

        # T1実験ではreadout mitigationを有効化（シングルショット測定の精度向上）
        if disable_mitigation:
            self.mitigation_options = {}  # mitigation無し
            print(f"T1 experiment: Raw measurement data (mitigation disabled for debugging)")
        else:
            self.mitigation_options = {"ro_error_mitigation": "pseudo_inverse"}
            print(f"T1 experiment: Standard T1 measurement with readout mitigation")
        self.mitigation_info = self.mitigation_options

    def create_circuits(self, **kwargs) -> List[Any]:
        """
        T1実験回路作成

        Args:
            delay_points: 遅延時間点数 (default: 16)
            max_delay: 最大遅延時間 [ns] (default: 1000)
            t1: T1緩和時間 [ns] (default: 500)
            t2: T2緩和時間 [ns] (default: 500)
            delay_times: 直接指定する遅延時間リスト [ns] (optional)

        Returns:
            T1回路リスト
        """
        delay_points = kwargs.get("delay_points", 51)
        max_delay = kwargs.get("max_delay", 100000)
        # t1, t2パラメータは使用しない（実測データからフィッティングのため）

        # 遅延時間範囲
        if "delay_times" in kwargs:
            delay_times = np.array(kwargs["delay_times"])
        else:
            # デフォルト: 100ns〜100μsの対数スケールで51点
            delay_times = np.logspace(np.log10(100), np.log10(100 * 1000), num=51)
            if delay_points != 51:
                delay_times = np.linspace(1, max_delay, delay_points)

        # メタデータ保存
        self.experiment_params = {
            "delay_times": delay_times.tolist(),
            "delay_points": len(delay_times),
            "max_delay": max_delay,
        }

        # T1回路作成（実際の回路にはt1, t2パラメータは不要）
        circuits = []
        for delay_time in delay_times:
            circuit = self._create_single_t1_circuit(delay_time)
            circuits.append(circuit)

        print(
            f"T1 circuits: Delay range {len(delay_times)} points from {delay_times[0]:.1f} to {delay_times[-1]:.1f} ns"
        )
        print(f"T1 circuit structure: |0⟩ → X → delay(τ) → measure (期待: P(1)は時間と共に減少)")

        return circuits

    def run_t1_experiment_parallel(
        self,
        devices: List[str] = ["qulacs"],
        shots: int = 1024,
        parallel_workers: int = 4,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        T1実験の並列実行（delay timeの順序を保持）
        """
        print(f"🔬 Running T1 experiment with {parallel_workers} parallel workers")

        # 回路作成
        circuits = self.create_circuits(**kwargs)
        delay_times = self.experiment_params["delay_times"]

        print(
            f"   📊 {len(circuits)} circuits × {len(devices)} devices = {len(circuits) * len(devices)} jobs"
        )

        # 並列実行（順序保持）
        job_data = self._submit_t1_circuits_parallel_with_order(
            circuits, devices, shots, parallel_workers
        )

        # 結果収集（順序保持）
        raw_results = self._collect_t1_results_parallel_with_order(
            job_data, parallel_workers
        )

        # 結果解析
        analysis = self.analyze_results(raw_results)

        return {
            "delay_times": delay_times,
            "device_results": analysis["device_results"],
            "analysis": analysis,
            "method": "t1_parallel_quantumlib",
        }

    def _submit_t1_circuits_parallel_with_order(
        self, circuits: List[Any], devices: List[str], shots: int, parallel_workers: int
    ) -> Dict[str, List[Dict]]:
        """
        T1回路の並列投入（CHSHスタイルで順序保持）
        """
        print(f"Enhanced T1 parallel submission: {parallel_workers} workers")

        if not self.oqtopus_available:
            return self._submit_t1_circuits_locally_parallel(
                circuits, devices, shots, parallel_workers
            )

        # 順序保持のためのデータ構造
        all_job_data = {device: [None] * len(circuits) for device in devices}

        # 回路とデバイスのペア作成（delay_time順序を保持）
        circuit_device_pairs = []
        for circuit_idx, circuit in enumerate(circuits):
            for device in devices:
                circuit_device_pairs.append((circuit_idx, circuit, device))

        def submit_single_t1_circuit(args):
            circuit_idx, circuit, device = args
            try:
                job_id = self.submit_circuit_to_oqtopus(circuit, shots, device)
                if job_id:
                    return device, job_id, circuit_idx, True
                else:
                    return device, None, circuit_idx, False
            except Exception as e:
                delay_time = self.experiment_params["delay_times"][circuit_idx]
                print(
                    f"T1 Circuit {circuit_idx} (τ={delay_time:.0f}ns) → {device}: {e}"
                )
                return device, None, circuit_idx, False

        # 並列投入実行
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = [
                executor.submit(submit_single_t1_circuit, args)
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
                    delay_time = self.experiment_params["delay_times"][circuit_idx]
                    print(
                        f"T1 Circuit {circuit_idx+1} (τ={delay_time:.0f}ns) → {device}: {job_id[:8]}..."
                    )
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
            successful_jobs = sum(
                1
                for job_data in all_job_data[device]
                if job_data and job_data["submitted"]
            )
            print(f"✅ {device}: {successful_jobs} T1 jobs submitted (order preserved)")

        return all_job_data

    def _submit_t1_circuits_locally_parallel(
        self, circuits: List[Any], devices: List[str], shots: int, parallel_workers: int
    ) -> Dict[str, List[Dict]]:
        """T1回路をローカルシミュレーターで並列実行"""
        print(f"T1 Local parallel execution: {parallel_workers} workers")

        all_job_data = {device: [None] * len(circuits) for device in devices}

        circuit_device_pairs = []
        for circuit_idx, circuit in enumerate(circuits):
            for device in devices:
                circuit_device_pairs.append((circuit_idx, circuit, device))

        def run_single_t1_circuit_locally(args):
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
                    f"Local T1 circuit {circuit_idx} (τ={delay_time:.0f}ns) → {device}: {e}"
                )
                return device, None, circuit_idx, False

        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = [
                executor.submit(run_single_t1_circuit_locally, args)
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
                f"✅ {device}: {successful} T1 circuits completed locally (order preserved)"
            )

        return all_job_data

    def _collect_t1_results_parallel_with_order(
        self, job_data: Dict[str, List[Dict]], parallel_workers: int
    ) -> Dict[str, List[Dict]]:
        """T1結果の並列収集（CHSHスタイルで順序保持）"""

        # 総ジョブ数を計算して収集開始をログ
        total_jobs_to_collect = sum(
            1
            for device_jobs in job_data.values()
            for job in device_jobs
            if job and job.get("submitted", False)
        )
        print(
            f"📊 Starting T1 results collection: {total_jobs_to_collect} jobs from {len(job_data)} devices"
        )

        # Handle local results
        if hasattr(self, "_local_results"):
            print("Using cached local T1 simulation results...")
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
                print(f"✅ {device}: {successful} T1 local results collected")
            return all_results

        if not self.oqtopus_available:
            print("OQTOPUS not available for T1 collection")
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

        def collect_single_t1_result(args):
            job_id, device, circuit_idx = args
            try:
                # ジョブ完了までポーリング
                result = self._poll_job_until_completion(job_id, timeout_minutes=5)
                # OQTOPUSジョブ構造に基づく成功判定: status == 'succeeded'
                if result and result.get("status") == "succeeded":
                    # 複数の方法で測定結果を取得を試行
                    counts = None
                    shots = 0
                    
                    # 方法1: BaseExperimentのget_oqtopus_resultが直接countsを返す場合
                    if "counts" in result:
                        counts = result["counts"]
                        shots = result.get("shots", 0)
                    
                    # 方法2: job_info内のresult構造から取得
                    if not counts:
                        job_info = result.get("job_info", {})
                        if isinstance(job_info, dict):
                            # OQTOPUS result構造を探索
                            sampling_result = job_info.get("result", {}).get("sampling", {})
                            if sampling_result:
                                counts = sampling_result.get("counts", {})
                    
                    # 方法3: job_info自体がresult形式の場合
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
                        # デバッグ: 順序とデータの確認
                        if not hasattr(self, '_sample_shown'):
                            delay_time = self.experiment_params["delay_times"][circuit_idx]
                            total_counts = sum(counts.values())
                            p1_raw = counts.get('1', counts.get(1, 0)) / total_counts if total_counts > 0 else 0
                            print(f"🔍 Sample result [circuit_idx={circuit_idx}] for τ={delay_time:.0f}ns: counts={dict(counts)}, P(1)_raw={p1_raw:.3f}")
                            self._sample_shown = getattr(self, '_sample_shown', 0) + 1
                            if self._sample_shown >= 5:  # 最初の5結果を表示して順序確認
                                self._sample_shown = True
                        
                        # 成功データを標準形式に変換
                        processed_result = {
                            "success": True,
                            "counts": dict(counts),  # Counterを辞書に変換
                            "status": result.get("status"),
                            "execution_time": result.get("execution_time", 0),
                            "shots": shots or sum(counts.values()) if counts else 0,
                        }
                        return device, processed_result, job_id, circuit_idx, True
                    else:
                        delay_time = self.experiment_params["delay_times"][circuit_idx]
                        # デバッグ用: 結果構造をより詳細に表示
                        print(f"⚠️ {device}[{circuit_idx}] (τ={delay_time:.0f}ns): {job_id[:8]}... no measurement data")
                        if hasattr(self, '_debug_count') and self._debug_count < 3:
                            print(f"   Debug - Full result: {result}")
                            self._debug_count = getattr(self, '_debug_count', 0) + 1
                        return device, None, job_id, circuit_idx, False
                else:
                    # ジョブ失敗の場合
                    delay_time = self.experiment_params["delay_times"][circuit_idx]
                    status = result.get("status", "unknown") if result else "no_result"
                    print(
                        f"⚠️ {device}[{circuit_idx}] (τ={delay_time:.0f}ns): {job_id[:8]}... failed ({status})"
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
                executor.submit(collect_single_t1_result, args)
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
                    # 失敗ケースは既に個別メソッド内でログ出力済み
                    pass

                # 進捗サマリーを20%ごとに表示
                progress_percent = (completed_jobs * 100) // total_jobs
                if (
                    progress_percent >= last_progress_percent + 20
                    and progress_percent < 100
                ):
                    print(
                        f"📈 T1 Collection Progress: {completed_jobs}/{total_jobs} ({progress_percent}%) - {successful_jobs} successful"
                    )
                    last_progress_percent = progress_percent

        # 最終結果サマリー
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
            f"🎉 T1 Collection Complete: {total_successful}/{total_attempted} successful ({success_rate:.1f}%)"
        )

        for device in job_data.keys():
            successful = sum(1 for r in all_results[device] if r is not None)
            total = len(job_data[device])
            print(f"✅ {device}: {successful}/{total} T1 results collected")

        return all_results

    def _poll_job_until_completion(
        self, job_id: str, timeout_minutes: int = 5, poll_interval: float = 2.0
    ):
        """
        ジョブが完了するまでポーリング

        Args:
            job_id: ジョブID
            timeout_minutes: タイムアウト時間（分）
            poll_interval: ポーリング間隔（秒）

        Returns:
            完了したジョブの結果、またはNone
        """
        import time

        timeout_seconds = timeout_minutes * 60
        start_time = time.time()
        last_status = None

        while time.time() - start_time < timeout_seconds:
            try:
                print(
                    f"🔍 Polling {job_id[:8]}... (elapsed: {time.time() - start_time:.1f}s)"
                )
                result = self.get_oqtopus_result(
                    job_id, timeout_minutes=1, verbose_log=False
                )  # 短いタイムアウトで取得

                # 簡略なステータスログのみ
                if not result:
                    continue

                if not result:
                    time.sleep(poll_interval)
                    continue

                status = result.get("status", "unknown")

                # 重要な状態変更のみログ出力
                if status != last_status and status in ["succeeded", "failed", "cancelled"]:
                    print(f"🏁 {job_id[:8]}... {status}")
                    last_status = status

                # 終了状態をチェック（実際の結果構造に合わせて柔軟に判定）
                if status in ["succeeded", "failed", "cancelled"]:
                    print(f"🏁 Job {job_id[:8]} completed with status: {status}")
                    return result
                elif status in ["running", "submitted", "pending"]:
                    # まだ実行中 - 続行
                    time.sleep(poll_interval)
                    continue
                elif result and result.get(
                    "success"
                ):  # BaseExperimentのget_oqtopus_resultが返す成功フラグ
                    print(f"🏁 Job {job_id[:8]} completed successfully (legacy format)")
                    return result
                elif not status:  # statusがセットされていない場合は続行
                    print(f"⚠️ Job {job_id[:8]} has no status field, continuing...")
                    time.sleep(poll_interval)
                    continue
                else:
                    # 不明な状態 - 少し待ってリトライ
                    print(
                        f"❓ Job {job_id[:8]} unknown status: {status}, continuing..."
                    )
                    time.sleep(poll_interval)
                    continue

            except Exception as e:
                # 一時的なエラーの場合はリトライ
                print(f"⚠️ Polling error for {job_id[:8]}: {e}")
                time.sleep(poll_interval)
                continue

        # タイムアウト
        print(f"⏰ Job {job_id[:8]}... timed out after {timeout_minutes} minutes")
        return None

    def _collect_single_t1_result(
        self, device: str, job_id: str, circuit_idx: int
    ) -> tuple:
        """
        単一T1結果の収集
        """
        try:
            result = self.get_oqtopus_result(job_id, wait_minutes=10)
            return device, result, job_id, circuit_idx, True
        except Exception as e:
            delay_time = self.experiment_params["delay_times"][circuit_idx]
            print(
                f"   ❌ {device}[{circuit_idx}] τ={delay_time:.0f}ns: Collection failed - {e}"
            )
            return device, None, job_id, circuit_idx, False

    def run_experiment(
        self,
        devices: List[str] = ["qulacs"],
        shots: int = 1024,
        parallel_workers: int = 4,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        T1実験実行（並列化版でrun_t1_experiment_parallelを呼び出し）
        """
        return self.run_t1_experiment_parallel(
            devices=devices, shots=shots, parallel_workers=parallel_workers, **kwargs
        )

    def _create_single_t1_circuit(self, delay_time: float):
        """
        単一T1回路作成（t1, t2パラメータ不要）
        """
        try:
            from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
        except ImportError:
            raise ImportError("Qiskit is required for circuit creation")

        # 1量子ビット + 1古典ビット
        qubits = QuantumRegister(1, "q")
        bits = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qubits, bits)

        # |1⟩状態に励起
        qc.x(0)

        # 遅延時間の間待機
        qc.delay(int(delay_time), 0, unit="ns")

        # Z基底測定
        qc.measure(0, 0)

        return qc

    def analyze_results(
        self, results: Dict[str, List[Dict[str, Any]]], **kwargs
    ) -> Dict[str, Any]:
        """
        T1実験結果解析

        Args:
            results: 生測定結果

        Returns:
            T1解析結果
        """
        if not results:
            return {"error": "No results to analyze"}

        delay_times = np.array(self.experiment_params["delay_times"])

        analysis = {
            "experiment_info": {
                "delay_points": len(delay_times),
                "expected_t1": self.expected_t1,
            },
            "device_results": {},
        }

        for device, device_results in results.items():
            if not device_results:
                continue

            device_analysis = self._analyze_device_results(device_results, delay_times)
            analysis["device_results"][device] = device_analysis

            # T1時定数推定（readout mitigationで補正済みデータを使用）
            t1_fitted, fitting_quality = self._estimate_t1_with_quality(
                device_analysis["p1_values"], delay_times
            )

            analysis["device_results"][device]["t1_fitted"] = t1_fitted
            analysis["device_results"][device]["fitting_quality"] = fitting_quality

            quality_str = (
                f"({fitting_quality['method']}, R²={fitting_quality['r_squared']:.3f})"
            )
            print(
                f"{device}: T1 = {t1_fitted:.1f} ns {quality_str} [with RO mitigation]"
            )
            
            # 最初と最後のP(1)値を確認
            if device_analysis["p1_values"]:
                p1_initial = device_analysis["p1_values"][0]
                p1_final = device_analysis["p1_values"][-1]
                print(f"   P(1) trend: {p1_initial:.3f} → {p1_final:.3f} ({'decreasing' if p1_final < p1_initial else 'INCREASING - CHECK DATA!'})")

        # デバイス間比較
        analysis["comparison"] = self._compare_devices(analysis["device_results"])

        return analysis

    def _analyze_device_results(
        self, device_results: List[Dict[str, Any]], delay_times: np.ndarray
    ) -> Dict[str, Any]:
        """
        単一デバイス結果解析（順序デバッグ付き）
        """
        print(f"🔍 Analyzing {len(device_results)} results in order...")
        
        p1_values = []

        for i, result in enumerate(device_results):
            delay_time = delay_times[i] if i < len(delay_times) else f"unknown[{i}]"
            
            if result and result["success"]:
                counts = result["counts"]

                # P(1)確率計算（readout mitigationで補正済み）
                p1 = self._calculate_p1_probability(counts)
                p1_values.append(p1)
                
                # 最初の5点で順序デバッグ
                if i < 5:
                    print(f"🔍 Point {i}: τ={delay_time}ns, P(1)={p1:.3f}, counts={dict(counts)}")
            else:
                p1_values.append(np.nan)
                if i < 5:
                    print(f"🔍 Point {i}: τ={delay_time}ns, FAILED")

        # 順序確認のためのサマリー
        valid_p1s = np.array([p for p in p1_values if not np.isnan(p)])
        if len(valid_p1s) >= 2:
            trend = "decreasing" if valid_p1s[-1] < valid_p1s[0] else "increasing"
            print(f"📈 T1 trend: P(1) {valid_p1s[0]:.3f} → {valid_p1s[-1]:.3f} ({trend})")
        
        # 統計計算

        return {
            "p1_values": p1_values,
            "delay_times": delay_times.tolist(),
            "statistics": {
                "initial_p1": (
                    float(p1_values[0])
                    if len(p1_values) > 0 and not np.isnan(p1_values[0])
                    else 1.0
                ),
                "final_p1": (
                    float(p1_values[-1])
                    if len(p1_values) > 0 and not np.isnan(p1_values[-1])
                    else 0.0
                ),
                "success_rate": len(valid_p1s) / len(p1_values) if p1_values else 0,
                "decay_observed": (
                    float(p1_values[0] - p1_values[-1])
                    if len(p1_values) >= 2
                    and not any(np.isnan([p1_values[0], p1_values[-1]]))
                    else 0.0
                ),
            },
        }

    def _calculate_p1_probability(self, counts: Dict[str, int]) -> float:
        """
        P(1)確率計算（OQTOPUSの10進数countsから2進数変換）
        """
        # OQTOPUSからの10進数countsを2進数形式に変換
        binary_counts = self._convert_decimal_to_binary_counts(counts)
        
        total = sum(binary_counts.values())
        if total == 0:
            return 0.0

        # デバッグ情報表示（初回のみ）
        if not hasattr(self, '_counts_debug_shown'):
            print(f"🔍 Raw decimal counts: {dict(counts)}")
            print(f"🔍 Converted binary counts: {dict(binary_counts)}")
            self._counts_debug_shown = True

        # 標準的なP(1)確率計算
        n_1 = binary_counts.get("1", 0)
        p1 = n_1 / total
        return p1
            
    def _convert_decimal_to_binary_counts(self, decimal_counts: Dict[str, int]) -> Dict[str, int]:
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
                print(f"⚠️ Unexpected count key: {decimal_key} (decimal value: {decimal_value})")
                continue
            
            # 既存のキーがある場合は加算
            if binary_key in binary_counts:
                binary_counts[binary_key] += count
            else:
                binary_counts[binary_key] = count
        
        return binary_counts

    def _calculate_z_expectation(self, counts: Dict[str, int]) -> float:
        """
        <Z>期待値計算（readout error耐性）
        """
        total = sum(counts.values())
        if total == 0:
            return 0.0

        # カウント取得
        if isinstance(list(counts.keys())[0], str):
            n_0 = counts.get("0", 0)
            n_1 = counts.get("1", 0)
        else:
            n_0 = counts.get(0, 0)
            n_1 = counts.get(1, 0)

        # <Z> = P(0) - P(1) = (n_0 - n_1) / total
        z_expectation = (n_0 - n_1) / total
        return z_expectation

    def _estimate_t1(self, p1_values: List[float], delay_times: np.ndarray) -> float:
        """
        T1時定数推定（改善された指数減衰フィッティング）
        """
        # NaNと非正値を除去
        valid_data = [
            (delay, p1)
            for delay, p1 in zip(delay_times, p1_values)
            if not np.isnan(p1) and p1 > 0
        ]

        if len(valid_data) < 3:
            return 0.0

        delays = np.array([d[0] for d in valid_data])
        p1s = np.array([d[1] for d in valid_data])

        try:
            # 非線形フィッティングを試行: P(t) = A * exp(-t/T1) + offset
            from scipy.optimize import curve_fit

            def exponential_decay(t, A, T1, offset):
                return A * np.exp(-t / T1) + offset

            # 初期推定値
            p0 = [p1s[0], self.expected_t1, 0.0]  # A, T1, offset

            # フィッティング実行
            popt, pcov = curve_fit(
                exponential_decay,
                delays,
                p1s,
                p0=p0,
                bounds=([0, 10, -0.1], [2.0, 10000, 0.1]),
                maxfev=2000,
            )

            t1_fitted = popt[1]

            # フィッティング品質チェック
            if pcov is not None and np.all(np.isfinite(pcov)):
                # 対角成分から標準誤差を計算
                param_errors = np.sqrt(np.diag(pcov))
                t1_error = param_errors[1]

                # 相対誤差が50%以下の場合のみ採用
                if t1_error / t1_fitted < 0.5:
                    return float(t1_fitted)

        except (ImportError, RuntimeError, ValueError, TypeError):
            # scipyが利用できない場合やフィッティングが失敗した場合は線形回帰にフォールバック
            pass

        try:
            # フォールバック: 線形回帰によるフィッティング
            log_p1s = np.log(p1s)

            # 線形フィッティング
            coeffs = np.polyfit(delays, log_p1s, 1)
            slope = coeffs[0]

            # T1 = -1/slope
            t1_fitted = -1.0 / slope if slope < 0 else float("inf")

            # 合理的な範囲に制限
            if t1_fitted < 10 or t1_fitted > 10000:
                t1_fitted = self.expected_t1

        except (ValueError, np.linalg.LinAlgError):
            # フィッティングが完全に失敗した場合はデフォルト値
            t1_fitted = self.expected_t1

        return float(t1_fitted)

    def _estimate_t1_with_quality(
        self, p1_values: List[float], delay_times: np.ndarray
    ) -> tuple[float, Dict[str, Any]]:
        """
        T1時定数推定と品質評価
        """
        # NaNと非正値を除去
        valid_data = [
            (delay, p1)
            for delay, p1 in zip(delay_times, p1_values)
            if not np.isnan(p1) and p1 > 0
        ]

        if len(valid_data) < 3:
            return 0.0, {
                "method": "insufficient_data",
                "r_squared": 0.0,
                "error": "inf",
            }

        delays = np.array([d[0] for d in valid_data])
        p1s = np.array([d[1] for d in valid_data])

        # 非線形フィッティングを試行
        try:
            from scipy.optimize import curve_fit

            def exponential_decay(t, A, T1, offset):
                return A * np.exp(-t / T1) + offset

            # 初期推定値
            p0 = [p1s[0], self.expected_t1, 0.0]

            # フィッティング実行
            popt, pcov = curve_fit(
                exponential_decay,
                delays,
                p1s,
                p0=p0,
                bounds=([0, 10, -0.1], [2.0, 10000, 0.1]),
                maxfev=2000,
            )

            t1_fitted = popt[1]

            # 予測値計算とR²算出
            p1_pred = exponential_decay(delays, *popt)
            ss_res = np.sum((p1s - p1_pred) ** 2)
            ss_tot = np.sum((p1s - np.mean(p1s)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            # 標準誤差計算
            param_error = "inf"
            if pcov is not None and np.all(np.isfinite(pcov)):
                param_errors = np.sqrt(np.diag(pcov))
                t1_error = param_errors[1]
                param_error = f"{t1_error:.1f}"

                # 高品質フィッティングの条件
                if t1_error / t1_fitted < 0.5 and r_squared > 0.7:
                    return float(t1_fitted), {
                        "method": "nonlinear",
                        "r_squared": r_squared,
                        "error": param_error,
                        "quality": "high" if r_squared > 0.9 else "medium",
                    }

        except (ImportError, RuntimeError, ValueError, TypeError):
            pass

        # フォールバック: 線形回帰
        try:
            log_p1s = np.log(p1s)
            coeffs = np.polyfit(delays, log_p1s, 1)
            slope, intercept = coeffs[0], coeffs[1]

            t1_fitted = -1.0 / slope if slope < 0 else self.expected_t1

            # 線形回帰のR²計算
            log_p1_pred = slope * delays + intercept
            ss_res = np.sum((log_p1s - log_p1_pred) ** 2)
            ss_tot = np.sum((log_p1s - np.mean(log_p1s)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            if 10 <= t1_fitted <= 10000:
                return float(t1_fitted), {
                    "method": "linear",
                    "r_squared": r_squared,
                    "error": "N/A",
                    "quality": "medium" if r_squared > 0.7 else "low",
                }

        except (ValueError, np.linalg.LinAlgError):
            pass

        # 全ての手法が失敗した場合
        return float(self.expected_t1), {
            "method": "default",
            "r_squared": 0.0,
            "error": "N/A",
            "quality": "poor",
        }

    def _estimate_t1_from_z_expectation(
        self, z_values: List[float], delay_times: np.ndarray
    ) -> tuple[float, Dict[str, Any]]:
        """
        <Z>期待値からT1時定数推定（readout error耐性）
        理論: <Z>(t) = -exp(-t/T1) (|1⟩状態から開始)
        """
        # NaNを除去
        valid_data = [
            (delay, z) for delay, z in zip(delay_times, z_values) if not np.isnan(z)
        ]

        if len(valid_data) < 3:
            return 0.0, {
                "method": "insufficient_data",
                "r_squared": 0.0,
                "error": "inf",
            }

        delays = np.array([d[0] for d in valid_data])
        z_vals = np.array([d[1] for d in valid_data])

        # 非線形フィッティングを試行: <Z>(t) = A * exp(-t/T1) + offset
        try:
            from scipy.optimize import curve_fit

            def z_exponential_decay(t, A, T1, offset):
                return A * np.exp(-t / T1) + offset

            # 初期推定値: A≈-1 (|1⟩から開始), T1≈expected, offset≈0
            p0 = [z_vals[0], self.expected_t1, 0.0]

            # フィッティング実行
            popt, pcov = curve_fit(
                z_exponential_decay,
                delays,
                z_vals,
                p0=p0,
                bounds=([-2.0, 10, -0.1], [0.0, 50000, 0.1]),
                maxfev=2000,
            )

            t1_fitted = popt[1]

            # 予測値計算とR²算出
            z_pred = z_exponential_decay(delays, *popt)
            ss_res = np.sum((z_vals - z_pred) ** 2)
            ss_tot = np.sum((z_vals - np.mean(z_vals)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            # 標準誤差計算
            param_error = "inf"
            if pcov is not None and np.all(np.isfinite(pcov)):
                param_errors = np.sqrt(np.diag(pcov))
                t1_error = param_errors[1]
                param_error = f"{t1_error:.1f}"

                # 高品質フィッティングの条件
                if t1_error / t1_fitted < 0.5 and r_squared > 0.7:
                    return float(t1_fitted), {
                        "method": "nonlinear_z",
                        "r_squared": r_squared,
                        "error": param_error,
                        "quality": "high" if r_squared > 0.9 else "medium",
                    }

        except (ImportError, RuntimeError, ValueError, TypeError):
            pass

        # フォールバック: 線形回帰 (log(-<Z>) vs t)
        try:
            # <Z>が負の値のみ使用（|1⟩状態なので）
            negative_z_data = [(delay, -z) for delay, z in zip(delays, z_vals) if z < 0]

            if len(negative_z_data) >= 3:
                delays_neg = np.array([d[0] for d in negative_z_data])
                neg_z_vals = np.array([d[1] for d in negative_z_data])

                log_neg_z = np.log(neg_z_vals)
                coeffs = np.polyfit(delays_neg, log_neg_z, 1)
                slope, intercept = coeffs[0], coeffs[1]

                t1_fitted = -1.0 / slope if slope < 0 else self.expected_t1

                # 線形回帰のR²計算
                log_pred = slope * delays_neg + intercept
                ss_res = np.sum((log_neg_z - log_pred) ** 2)
                ss_tot = np.sum((log_neg_z - np.mean(log_neg_z)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

                if 10 <= t1_fitted <= 50000:
                    return float(t1_fitted), {
                        "method": "linear_z",
                        "r_squared": r_squared,
                        "error": "N/A",
                        "quality": "medium" if r_squared > 0.7 else "low",
                    }

        except (ValueError, np.linalg.LinAlgError):
            pass

        # 全ての手法が失敗した場合
        return float(self.expected_t1), {
            "method": "default_z",
            "r_squared": 0.0,
            "error": "N/A",
            "quality": "poor",
        }

    def _compare_devices(
        self, device_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        デバイス間比較分析
        """
        if len(device_results) < 2:
            return {"note": "Multiple devices required for comparison"}

        comparison = {
            "device_count": len(device_results),
            "t1_comparison": {},
            "decay_comparison": {},
        }

        for device, analysis in device_results.items():
            stats = analysis["statistics"]
            comparison["t1_comparison"][device] = analysis.get("t1_fitted", 0.0)
            comparison["decay_comparison"][device] = stats["decay_observed"]

        return comparison

    def save_experiment_data(
        self, results: Dict[str, Any], metadata: Dict[str, Any] = None
    ) -> str:
        """
        T1実験データ保存
        """
        # T1実験専用の保存形式
        t1_data = {
            "experiment_type": "T1_Decay",
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
        main_file = self.data_manager.save_data(t1_data, "t1_experiment_results")

        # 追加ファイル保存
        if "device_results" in results:
            # デバイス別サマリー
            device_summary = {
                device: {
                    "t1_fitted": analysis.get("t1_fitted", 0.0),
                    "statistics": analysis["statistics"],
                }
                for device, analysis in results["device_results"].items()
            }
            self.data_manager.save_data(device_summary, "device_t1_summary")

            # P(1)データ（プロット用）
            p1_data = {
                "delay_times": self.experiment_params["delay_times"],
                "device_p1_values": {
                    device: analysis["p1_values"]
                    for device, analysis in results["device_results"].items()
                },
            }
            self.data_manager.save_data(p1_data, "p1_values_for_plotting")

        return main_file

    def generate_t1_plot(
        self, results: Dict[str, Any], save_plot: bool = True, show_plot: bool = False
    ) -> Optional[str]:
        """Generate T1 experiment plot with all formatting"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available - skipping plot generation")
            return None

        delay_times = results.get("delay_times", np.linspace(1, 1000, 16))
        device_results = results.get("device_results", {})

        if not device_results:
            print("No device results for plotting")
            return None

        # 理論値は使用しない

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot experimental data for each device
        colors = ["blue", "red", "green", "orange", "purple"]

        for i, (device, device_data) in enumerate(device_results.items()):
            if "p1_values" in device_data:
                p1_values = device_data["p1_values"]
                t1_fitted = device_data.get("t1_fitted", 0.0)
                fitting_quality = device_data.get("fitting_quality", {})
                r_squared = fitting_quality.get("r_squared", 0.0)
                color = colors[i % len(colors)]

                # 実測データプロット
                ax.semilogx(
                    delay_times,
                    p1_values,
                    "o",
                    markersize=6,
                    label=f"{device} data",
                    alpha=0.8,
                    color=color,
                )

                # フィット曲線プロット
                if t1_fitted > 0:
                    # フィットされた指数減衰曲線を描画
                    fit_delays = np.logspace(
                        np.log10(min(delay_times)), np.log10(max(delay_times)), 100
                    )
                    # 簡単な指数減衰: P(t) = P0 * exp(-t/T1)
                    p0_estimate = max(p1_values) if p1_values else 1.0
                    fit_curve = p0_estimate * np.exp(-fit_delays / t1_fitted)
                    ax.semilogx(
                        fit_delays,
                        fit_curve,
                        "-",
                        linewidth=2,
                        color=color,
                        alpha=0.7,
                        label=f"{device} fit (T1={t1_fitted:.0f}ns, R²={r_squared:.3f})",
                    )

        # 理論曲線は削除（実測データとフィットのみ表示）

        # Formatting
        ax.set_xlabel("Delay time τ [ns] (log scale)", fontsize=14)
        ax.set_ylabel("P(1)", fontsize=14)
        ax.set_title(
            f"QuantumLib T1 Decay Experiment",
            fontsize=16,
            fontweight="bold",
        )
        ax.grid(True, which="both", ls="--", linewidth=0.5)
        ax.legend(fontsize=12)
        ax.set_ylim(0, 1.1)

        plot_filename = None
        if save_plot:
            # Save plot in experiment results directory
            plt.tight_layout()
            plot_filename = f"t1_plot_{self.experiment_name}_{int(time.time())}.png"

            # Always save to experiment results directory
            if hasattr(self, "data_manager") and hasattr(
                self.data_manager, "session_dir"
            ):
                plot_path = f"{self.data_manager.session_dir}/plots/{plot_filename}"
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                print(f"Plot saved: {plot_path}")
                plot_filename = plot_path  # Return full path
            else:
                # Fallback: save in current directory but warn
                plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
                print(f"⚠️ Plot saved to current directory: {plot_filename}")
                print("   (data_manager not available)")

        # Try to display plot
        if show_plot:
            try:
                plt.show()
            except:
                pass

        plt.close()
        return plot_filename

    def save_complete_experiment_data(self, results: Dict[str, Any]) -> str:
        """Save experiment data and generate comprehensive report"""
        # Save main experiment data using existing system
        main_file = self.save_experiment_data(results["analysis"])

        # Generate and save plot
        plot_file = self.generate_t1_plot(results, save_plot=True, show_plot=False)

        # Create experiment summary
        summary = self._create_experiment_summary(results)
        summary_file = self.data_manager.save_data(summary, "experiment_summary")

        print(f"📊 Complete experiment data saved:")
        print(f"  • Main results: {main_file}")
        print(f"  • Plot: {plot_file if plot_file else 'Not generated'}")
        print(f"  • Summary: {summary_file}")

        return main_file

    def _create_experiment_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create human-readable experiment summary"""
        device_results = results.get("device_results", {})
        delay_times = results.get("delay_times", [])

        summary = {
            "experiment_overview": {
                "experiment_name": self.experiment_name,
                "timestamp": time.time(),
                "method": results.get("method", "t1_decay"),
                "delay_points": len(delay_times),
                "devices_tested": list(device_results.keys()),
            },
            "key_results": {},
            "t1_analysis": {
                "expected_t1": self.expected_t1,
                "clear_decay_detected": False,
            },
        }

        # Analyze each device
        min_decay_threshold = 0.3  # Minimum decay to consider significant

        for device, device_data in device_results.items():
            if "p1_values" in device_data:
                p1_values = device_data["p1_values"]
                valid_p1s = [p for p in p1_values if not np.isnan(p)]

                if valid_p1s and len(valid_p1s) >= 2:
                    initial_p1 = valid_p1s[0]
                    final_p1 = valid_p1s[-1]
                    decay = initial_p1 - final_p1
                    t1_fitted = device_data.get("t1_fitted", 0.0)

                    summary["key_results"][device] = {
                        "initial_p1": initial_p1,
                        "final_p1": final_p1,
                        "decay_observed": decay,
                        "t1_fitted": t1_fitted,
                        "clear_decay": decay > min_decay_threshold,
                    }

                    if decay > min_decay_threshold:
                        summary["t1_analysis"]["clear_decay_detected"] = True

        return summary

    def display_results(self, results: Dict[str, Any], use_rich: bool = True) -> None:
        """Display T1 experiment results in formatted table"""
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
                    title="T1 Decay Results", show_header=True, header_style="bold blue"
                )
                table.add_column("Device", style="cyan")
                table.add_column("T1 Fitted [ns]", justify="right")
                table.add_column("Initial P(1)", justify="right")
                table.add_column("Final P(1)", justify="right")
                table.add_column("Decay", justify="right")
                table.add_column("Clear Decay", justify="center")

                method = results.get("method", "quantumlib_t1")

                for device, device_data in device_results.items():
                    if "p1_values" in device_data:
                        p1_values = device_data["p1_values"]
                        valid_p1s = [p for p in p1_values if not np.isnan(p)]

                        if valid_p1s and len(valid_p1s) >= 2:
                            initial_p1 = valid_p1s[0]
                            final_p1 = valid_p1s[-1]
                            decay = initial_p1 - final_p1
                            t1_fitted = device_data.get("t1_fitted", 0.0)

                            clear_decay = "YES" if decay > 0.3 else "NO"
                            decay_style = "green" if decay > 0.3 else "yellow"

                            table.add_row(
                                device.upper(),
                                f"{t1_fitted:.1f}",
                                f"{initial_p1:.3f}",
                                f"{final_p1:.3f}",
                                f"{decay:.3f}",
                                clear_decay,
                                style=decay_style if decay > 0.3 else None,
                            )

                console.print(table)
                console.print(f"\nExpected T1: {self.expected_t1} ns")
                console.print(f"Clear decay threshold: 0.3")

            except ImportError:
                use_rich = False

        if not use_rich:
            # Fallback to simple text display
            print("\n" + "=" * 60)
            print("T1 Decay Results")
            print("=" * 60)

            method = results.get("method", "quantumlib_t1")

            for device, device_data in device_results.items():
                if "p1_values" in device_data:
                    p1_values = device_data["p1_values"]
                    valid_p1s = [p for p in p1_values if not np.isnan(p)]

                    if valid_p1s and len(valid_p1s) >= 2:
                        initial_p1 = valid_p1s[0]
                        final_p1 = valid_p1s[-1]
                        decay = initial_p1 - final_p1
                        t1_fitted = device_data.get("t1_fitted", 0.0)

                        clear_decay = "YES" if decay > 0.3 else "NO"

                        print(f"Device: {device.upper()}")
                        print(f"  T1 Fitted: {t1_fitted:.1f} ns")
                        print(f"  Initial P(1): {initial_p1:.3f}")
                        print(f"  Final P(1): {final_p1:.3f}")
                        print(f"  Decay: {decay:.3f}")
                        print(f"  Clear Decay: {clear_decay}")
                        print()

            print(f"Expected T1: {self.expected_t1} ns")
            print(f"Clear decay threshold: 0.3")
            print("=" * 60)

    def run_complete_t1_experiment(
        self,
        devices: List[str] = ["qulacs"],
        delay_points: int = 16,
        max_delay: float = 1000,
        t1: float = 500,
        t2: float = 500,
        shots: int = 1024,
        parallel_workers: int = 4,
        save_data: bool = True,
        save_plot: bool = True,
        show_plot: bool = False,
        display_results: bool = True,
    ) -> Dict[str, Any]:
        """
        Run complete T1 experiment with all post-processing
        This is the main entry point for CLI usage
        """
        print(f"🔬 Running complete T1 experiment: {self.experiment_name}")
        print(f"   Devices: {devices}")
        print(f"   Delay points: {delay_points}, Max delay: {max_delay} ns")
        print(f"   T1: {t1} ns, T2: {t2} ns")
        print(f"   Shots: {shots}, Parallel workers: {parallel_workers}")

        # Run the T1 experiment
        results = self.run_experiment(
            devices=devices,
            shots=shots,
            delay_points=delay_points,
            max_delay=max_delay,
            t1=t1,
            t2=t2,
        )

        # Save data if requested
        if save_data:
            self.save_complete_experiment_data(results)
        elif save_plot:
            # Just save plot without full data
            self.generate_t1_plot(results, save_plot=True, show_plot=show_plot)

        # Display results if requested
        if display_results:
            self.display_results(results, use_rich=True)

        return results
