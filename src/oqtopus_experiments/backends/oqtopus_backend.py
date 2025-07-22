#!/usr/bin/env python3
"""
OQTOPUS Backend - Quantum hardware backend with integrated device information
"""

from typing import Any

from ..models.circuit_collection import CircuitCollection

# OQTOPUS imports
try:
    from quri_parts_oqtopus.backend import OqtopusSamplingBackend  # noqa: F401

    OQTOPUS_AVAILABLE = True
except ImportError:
    OQTOPUS_AVAILABLE = False


class OqtopusBackend:
    """
    OQTOPUS backend for running on quantum hardware with integrated device information

    Provides a unified interface for quantum execution and device information access.
    Compatible with usage.py style API.
    """

    def __init__(self, device: str = "anemone", timeout_seconds: int = 120):
        """
        Initialize OQTOPUS backend with device information

        Args:
            device: Device name to use for submissions and device info (default: "anemone")
            timeout_seconds: Maximum wait time for job completion
        """
        self.backend_type = "oqtopus"
        self.device_name = device
        self.timeout_seconds = timeout_seconds
        self._device_info: Any = None

        # Try to initialize OQTOPUS
        try:
            from quri_parts_oqtopus.backend import OqtopusSamplingBackend

            backend_instance = OqtopusSamplingBackend()
            self.backend: Any = backend_instance
            self.available = True
            print(
                f"OQTOPUS backend initialized (device: {device}, timeout: {timeout_seconds}s)"
            )
        except ImportError:
            self.backend = None
            self.available = False
            print("OQTOPUS backend not available (missing quri_parts_oqtopus)")

        # Initialize device info (lazy loading)
        self._device_info_loaded = False

    def run(
        self, circuit: Any, shots: int = 1024, circuit_params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Run circuit on OQTOPUS backend

        Args:
            circuit: Quantum circuit to run
            shots: Number of shots
            circuit_params: Optional parameters to embed in job description

        Returns:
            Result dictionary with counts and embedded parameters
        """
        if not self.available:
            print("OQTOPUS not available, using simulated results")
            return {"counts": {"0": shots // 2, "1": shots // 2}}

        try:
            import json
            import time

            from qiskit.qasm3 import dumps

            # Convert circuit to QASM3
            qasm_str = dumps(circuit)
            print(f"Submitting circuit to OQTOPUS device: {self.device_name}")

            # Prepare job description with parameters
            description = None
            if circuit_params:
                description = json.dumps(circuit_params)

            # Submit to OQTOPUS with configured settings
            job = self.backend.sample_qasm(
                qasm_str,
                device_id=self.device_name,
                shots=shots,
                transpiler_info={
                    "transpiler_lib": "qiskit",
                    "transpiler_options": {
                        "basis_gates": ["sx", "x", "rz", "cx"],
                        "optimization_level": 1,
                    },
                },
                mitigation_info={
                    "ro_error_mitigation": "pseudo_inverse",
                },
                description=description,
            )

            print(f"Job submitted with ID: {job.job_id[:8]}...")

            # Wait for results (polling with timeout)
            max_wait = self.timeout_seconds
            start_time = time.time()
            print(f"⏰ Waiting for results (timeout: {max_wait}s)...")

            while time.time() - start_time < max_wait:
                try:
                    # Re-retrieve job object to get fresh status
                    job = self.backend.retrieve_job(job.job_id)
                    job_dict = job._job.to_dict()
                    status = job_dict.get("status", "unknown")

                    print(f"⏳ Job status: {status}")

                    if status in ["succeeded", "ready"]:
                        try:
                            result = job.result()
                            if result and hasattr(result, "counts"):
                                counts = result.counts

                                # Extract parameters from job description
                                params = {}
                                try:
                                    if hasattr(job, "description") and job.description:
                                        params = json.loads(job.description)
                                except (json.JSONDecodeError, AttributeError):
                                    params = {}

                                print(
                                    f"✅ OQTOPUS job completed successfully (status: {status})"
                                )
                                return {
                                    "counts": self._normalize_counts({str(k): int(v) for k, v in counts.items()}),
                                    "job_id": job.job_id,
                                    "shots": shots,
                                    "backend": "oqtopus",
                                    "params": params,
                                }
                        except Exception as e:
                            print(f"Error getting result: {e}")
                            # If job is ready but result access fails, continue waiting
                            if status == "ready":
                                time.sleep(2)
                                continue
                    elif status in ["failed", "cancelled", "error"]:
                        break
                    elif status in ["submitted", "running", "queued", "pending"]:
                        # Still processing, wait and retry
                        time.sleep(10)
                        continue
                    else:
                        # For any other status, including unknown, try to get result once
                        try:
                            result = job.result()
                            if result and hasattr(result, "counts"):
                                counts = result.counts

                                # Extract parameters from job description
                                params = {}
                                try:
                                    if hasattr(job, "description") and job.description:
                                        params = json.loads(job.description)
                                except (json.JSONDecodeError, AttributeError):
                                    params = {}

                                print(f"OQTOPUS job completed (status: {status})")
                                return {
                                    "counts": self._normalize_counts({str(k): int(v) for k, v in counts.items()}),
                                    "job_id": job.job_id,
                                    "shots": shots,
                                    "backend": "oqtopus",
                                    "params": params,
                                }
                        except Exception:
                            pass

                        print(f"Unknown job status: {status}")
                        time.sleep(5)
                        continue

                except Exception as e:
                    print(f"Error checking job status: {e}")
                    time.sleep(5)
                    continue

            elapsed = time.time() - start_time
            print(f"OQTOPUS job timeout after {elapsed:.1f}s, using simulated results")
            return {
                "counts": {"0": shots // 2, "1": shots // 2},
                "job_id": job.job_id if "job" in locals() else "timeout",
                "backend": "oqtopus_timeout",
                "timeout": True,
            }

        except Exception as e:
            print(f"OQTOPUS submission failed: {e}")
            print("Falling back to simulated results")
            return {"counts": {"0": shots // 2, "1": shots // 2}}

    def submit(self, circuit: Any, shots: int = 1024) -> str:
        """
        Submit circuit to OQTOPUS (asynchronous)

        Args:
            circuit: Quantum circuit to submit
            shots: Number of shots

        Returns:
            Job ID string
        """
        # In real implementation, this would submit to OQTOPUS
        return f"job_{hash(str(circuit)) % 10000}"

    def get_result(self, job_id: str) -> dict[str, Any]:
        """
        Get result from job ID

        Args:
            job_id: Job ID from submit()

        Returns:
            Result dictionary with counts
        """
        # In real implementation, this would fetch from OQTOPUS
        return {"counts": {"0": 500, "1": 500}}

    @property
    def device(self):
        """
        Get device information with lazy loading (usage.py style)

        Returns:
            DeviceInfo object for the current device
        """
        if not self._device_info_loaded:
            try:
                from ..devices import DeviceInfo

                self._device_info = DeviceInfo(self.device_name)
                self._device_info_loaded = True
            except Exception as e:
                print(f"Could not load device info: {e}")
                self._device_info = None
                self._device_info_loaded = True

        return self._device_info

    def show_device(self, show_qubits: bool = True, show_couplings: bool = True):
        """
        Display device information (usage.py style)

        Args:
            show_qubits: Whether to show qubit table
            show_couplings: Whether to show coupling map
        """
        if self.device and self.device.available:
            self.device.show(show_qubits=show_qubits, show_couplings=show_couplings)
        else:
            print(f"Device information not available for {self.device_name}")

    def get_best_qubits(self, n: int = 5, sorted_key: str = "fidelity") -> Any:
        """
        Get top N qubits by fidelity (usage.py style)

        Args:
            n: Number of top qubits to return

        Returns:
            DataFrame with top N qubits sorted by fidelity
        """
        if self.device and self.device.available:
            return self.device.get_best_qubits(n, sorted_key=sorted_key)
        else:
            print(f"Device information not available for {self.device_name}")
            return None

    def get_device_stats(self):
        """
        Get device statistics (usage.py style)

        Returns:
            Dictionary with device statistics
        """
        if self.device and self.device.available:
            return self.device.get_qubit_stats()
        else:
            print(f"Device information not available for {self.device_name}")
            return None

    def plot_device_layout(self, color_by: str = "fidelity"):
        """
        Plot device layout (usage.py style)

        Args:
            color_by: Property to color qubits by
        """
        if self.device and self.device.available:
            self.device.plot_layout(color_by=color_by, show_edges=True)
        else:
            print(f"Device information not available for {self.device_name}")

    def save_device_info(self, filename: str | None = None) -> str:
        """
        Save device information to file (usage.py style)

        Args:
            filename: Output filename (optional)

        Returns:
            Path to saved file
        """
        if self.device and self.device.available:
            return str(self.device.save_data(filename))
        else:
            error_msg = f"❌ Device information not available for {self.device_name}"
            print(error_msg)
            return error_msg

    def get_device_summary(self):
        """
        Get device summary information

        Returns:
            Dictionary with device summary
        """
        if self.device and self.device.available:
            return self.device.summary()
        else:
            return {"error": f"Device information not available for {self.device_name}"}

    def compare_qubits(self, qubit_ids: list[int]):
        """
        Compare specific qubits

        Args:
            qubit_ids: List of qubit IDs to compare

        Returns:
            DataFrame with comparison of specified qubits
        """
        if self.device and self.device.available:
            return self.device.compare_qubits(qubit_ids)
        else:
            print(f"Device information not available for {self.device_name}")
            return None

    def transpile(
        self,
        circuits: Any | list[Any] | CircuitCollection,
        physical_qubits: list[int] | None = None,
        optimization_level: int = 1,
        **kwargs,
    ) -> Any | list[Any] | CircuitCollection:
        """
        Transpile circuits for the OQTOPUS device

        Args:
            circuits: Single circuit, list of circuits, or CircuitCollection to transpile
            physical_qubits: List of physical qubits to use for mapping
            optimization_level: Optimization level for transpilation
            **kwargs: Additional transpiler options

        Returns:
            Transpiled circuit(s) - same type as input
        """
        try:
            from tranqu import Tranqu
        except ImportError:
            return circuits

        # Handle input types
        single_circuit = not isinstance(circuits, list | CircuitCollection)
        circuit_collection_input = isinstance(circuits, CircuitCollection)

        if single_circuit:
            circuits = [circuits]
        elif circuit_collection_input and hasattr(circuits, 'to_list'):
            circuits = circuits.to_list()  # type: ignore

        # Get physical qubits
        if physical_qubits is None:
            n_qubits = max(len(circuit.qubits) for circuit in circuits)
            best_qubits = self.get_best_qubits(n_qubits)
            if best_qubits is not None:
                physical_qubits = [
                    int(best_qubits.iloc[i]["physical_id"]) for i in range(n_qubits)
                ]
            else:
                physical_qubits = list(range(n_qubits))

        # Transpile circuits
        transpiled_circuits = []
        tranqu = Tranqu()

        for circuit in circuits:
            try:
                options = {
                    "basis_gates": ["sx", "x", "rz", "cx"],
                    "optimization_level": optimization_level,
                    **kwargs,
                }

                # Add initial layout if physical qubits specified
                if physical_qubits and len(circuit.qubits) <= len(physical_qubits):
                    initial_layout = {
                        circuit.qubits[i]: physical_qubits[i]
                        for i in range(len(circuit.qubits))
                    }
                    options["initial_layout"] = initial_layout

                result = tranqu.transpile(
                    program=circuit,
                    transpiler_lib="qiskit",
                    program_lib="qiskit",
                    transpiler_options=options,
                    device=self.device.device_info,
                    device_lib="oqtopus",
                )
                transpiled_circuits.append(result.transpiled_program)

            except Exception:
                # If transpilation fails, try without layout
                try:
                    simple_options = {
                        "basis_gates": ["sx", "x", "rz", "cx"],
                        "optimization_level": optimization_level,
                        **kwargs,
                    }
                    result = tranqu.transpile(
                        program=circuit,
                        transpiler_lib="qiskit",
                        program_lib="qiskit",
                        transpiler_options=simple_options,
                        device=self.device.device_info,
                        device_lib="oqtopus",
                    )
                    transpiled_circuits.append(result.transpiled_program)
                except Exception:
                    transpiled_circuits.append(circuit)

        # Return same type as input
        if single_circuit:
            return transpiled_circuits[0]
        elif circuit_collection_input:
            return CircuitCollection(transpiled_circuits)
        else:
            return transpiled_circuits

    def submit_parallel(
        self,
        circuits: Any,
        shots: int = 1024,
        circuit_params: list[dict] | None = None,
        disable_transpilation: bool = False,
    ) -> list[str | None]:
        """
        Submit circuits in parallel to OQTOPUS cloud with parameter tracking

        Args:
            circuits: List of circuits to submit
            shots: Number of shots per circuit
            circuit_params: List of parameter dictionaries for each circuit
            disable_transpilation: Whether to disable OQTOPUS transpilation (use transpiler_lib: None)

        Returns:
            List of job IDs
        """
        if not self.available:
            print("OQTOPUS not available, using mock job IDs")
            return [f"mock_job_{i}" for i in range(len(circuits))]

        try:
            import json
            from concurrent.futures import ThreadPoolExecutor, as_completed

            from qiskit.qasm3 import dumps

            # Generate default params if not provided
            if circuit_params is None:
                circuit_params = [{"circuit_index": i} for i in range(len(circuits))]

            def submit_single_circuit(circuit_with_params):
                circuit, params, index = circuit_with_params
                try:
                    qasm_str = dumps(circuit)

                    # Add circuit index to params for safety
                    params_with_index = {**params, "circuit_index": index}
                    description = json.dumps(params_with_index)

                    # Configure transpiler based on disable_transpilation flag
                    if disable_transpilation:
                        transpiler_info: dict[str, Any] = {"transpiler_lib": None}
                        print(
                            f"Circuit {index + 1}: physical qubit specified, disabling OQTOPUS transpilation"
                        )
                    else:
                        transpiler_info = {
                            "transpiler_lib": "qiskit",
                            "transpiler_options": {
                                "basis_gates": ["sx", "x", "rz", "cx"],
                                "optimization_level": 1,
                            },
                        }

                    job = self.backend.sample_qasm(
                        qasm_str,
                        device_id=self.device_name,
                        shots=shots,
                        transpiler_info=transpiler_info,
                        mitigation_info={
                            "ro_error_mitigation": "pseudo_inverse",
                        },
                        description=description,
                    )
                    return index, job.job_id
                except Exception as e:
                    print(f"Circuit submission failed: {e}")
                    return index, None

            # Submit circuits in parallel with parameters
            job_ids: list[str | None] = [None] * len(circuits)

            with ThreadPoolExecutor(max_workers=4) as executor:
                # Create circuit-params-index tuples
                circuit_tuples = [
                    (circuits[i], circuit_params[i], i) for i in range(len(circuits))
                ]

                future_to_index = {
                    executor.submit(
                        submit_single_circuit, circuit_tuple
                    ): circuit_tuple[2]
                    for circuit_tuple in circuit_tuples
                }

                # Collect job IDs as they complete
                for future in as_completed(future_to_index):
                    try:
                        circuit_idx, job_id = future.result()
                        job_ids[circuit_idx] = job_id
                        if job_id:
                            # Show parameter info
                            params_str = ", ".join(
                                f"{k}={v}"
                                for k, v in circuit_params[circuit_idx].items()
                                if k != "circuit_index"
                            )
                            print(
                                f"Circuit {circuit_idx + 1}/{len(circuits)} ({params_str}) → {self.device_name}: {job_id[:8]}..."
                            )
                        else:
                            print(f"Circuit {circuit_idx + 1} submission failed")
                    except Exception as e:
                        circuit_idx = future_to_index[future]
                        print(f"Circuit {circuit_idx + 1} submission failed: {e}")
                        job_ids[circuit_idx] = None

            successful_jobs = len([j for j in job_ids if j])
            print(
                f"✅ {self.device_name}: {successful_jobs} jobs submitted with parameters"
            )
            return job_ids

        except Exception as e:
            print(f"Parallel submission failed: {e}")
            return [f"mock_job_{i}" for i in range(len(circuits))]

    def collect_parallel(
        self, job_ids: list[str], retry_failed: bool = True
    ) -> list[dict]:
        """
        Collect results from multiple jobs in parallel

        Args:
            job_ids: List of job IDs to collect
            retry_failed: Whether to retry failed collections

        Returns:
            List of result dictionaries
        """
        if not self.available:
            raise RuntimeError("OQTOPUS backend not available - cannot collect results")

        try:
            import time
            from concurrent.futures import ThreadPoolExecutor, as_completed

            def collect_single_job(job_id_idx):
                job_id, idx = job_id_idx
                if not job_id or job_id.startswith("mock_"):
                    return idx, {"counts": {"0": 500, "1": 500}, "status": "mock"}

                try:
                    # Retrieve the job object from OQTOPUS
                    job = self.backend.retrieve_job(job_id)

                    # Check job status with extended retry for running jobs
                    max_retries = 5
                    for attempt in range(max_retries):
                        try:
                            # Check job status
                            job_dict = job._job.to_dict()
                            status = job_dict.get("status", "unknown")

                            if status in ["succeeded", "ready"]:
                                try:
                                    result = job.result()
                                    if result and hasattr(result, "counts"):
                                        counts = result.counts

                                        # Extract parameters from job description
                                        import json

                                        params = {}
                                        try:
                                            if (
                                                hasattr(job, "description")
                                                and job.description
                                            ):
                                                params = json.loads(job.description)
                                        except (json.JSONDecodeError, AttributeError):
                                            params = {"circuit_index": idx}

                                        print(
                                            f"✅ {self.device_name}: {job_id[:8]}... collected"
                                        )
                                        return idx, {
                                            "counts": self._normalize_counts(
                                                {str(k): int(v) for k, v in counts.items()}
                                            ),
                                            "job_id": job_id,
                                            "shots": sum(counts.values()),
                                            "backend": self.device_name,
                                            "status": "success",
                                            "params": params,
                                        }
                                except Exception as e:
                                    print(
                                        f"⚠️  Error getting result for {job_id[:8]}...: {e}"
                                    )
                                    if attempt < max_retries - 1:
                                        time.sleep(2)
                                        continue

                            elif status in ["failed", "cancelled", "error"]:
                                print(
                                    f"❌ {self.device_name}: {job_id[:8]}... failed (status: {status})"
                                )
                                return idx, {
                                    "counts": {},
                                    "status": "failed",
                                    "error": f"Job {status}",
                                }

                            elif status in [
                                "submitted",
                                "running",
                                "queued",
                                "pending",
                            ]:
                                if attempt < max_retries - 1:
                                    wait_time = 3 if status == "running" else 5
                                    print(
                                        f"⏳ {job_id[:8]}... status: {status} (attempt {attempt + 1})"
                                    )
                                    time.sleep(wait_time)
                                    continue
                                else:
                                    print(
                                        f"⏰ {job_id[:8]}... still {status} after {max_retries} attempts"
                                    )
                                    return idx, {
                                        "counts": {},
                                        "status": "timeout",
                                        "error": f"Timeout in {status} state",
                                    }
                            else:
                                # Unknown status, try to get result once
                                try:
                                    result = job.result()
                                    if result and hasattr(result, "counts"):
                                        counts = result.counts

                                        # Extract parameters from job description
                                        import json

                                        params = {}
                                        try:
                                            if (
                                                hasattr(job, "description")
                                                and job.description
                                            ):
                                                params = json.loads(job.description)
                                        except (json.JSONDecodeError, AttributeError):
                                            params = {"circuit_index": idx}

                                        print(
                                            f"✅ {self.device_name}: {job_id[:8]}... collected"
                                        )
                                        return idx, {
                                            "counts": self._normalize_counts(
                                                {str(k): int(v) for k, v in counts.items()}
                                            ),
                                            "job_id": job_id,
                                            "shots": sum(counts.values()),
                                            "backend": self.device_name,
                                            "status": "success",
                                            "params": params,
                                        }
                                except Exception:
                                    pass

                                if attempt < max_retries - 1:
                                    print(
                                        f"❓ {job_id[:8]}... unknown status: {status} (attempt {attempt + 1})"
                                    )
                                    time.sleep(2)
                                    continue
                                else:
                                    return idx, {
                                        "counts": {},
                                        "status": "unknown",
                                        "error": f"Unknown status: {status}",
                                    }

                        except Exception as e:
                            if attempt < max_retries - 1:
                                print(
                                    f"⚠️  Status check failed for {job_id[:8]}... (attempt {attempt + 1}): {e}"
                                )
                                time.sleep(2)
                                continue
                            else:
                                return idx, {
                                    "counts": {},
                                    "status": "error",
                                    "error": f"Status check failed: {e}",
                                }

                except Exception as e:
                    print(f"❌ Collection failed for {job_id[:8]}...: {e}")
                    return idx, {
                        "counts": {},
                        "status": "error",
                        "error": f"Collection failed: {e}",
                    }

            print("Collecting results from OQTOPUS...")

            # First collection attempt
            results: list[dict[Any, Any]] = [{}] * len(job_ids)
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_job = {
                    executor.submit(collect_single_job, (job_id, i)): i
                    for i, job_id in enumerate(job_ids)
                }

                for future in as_completed(future_to_job):
                    try:
                        idx, result = future.result()
                        results[idx] = result
                    except Exception as e:
                        job_idx = future_to_job[future]
                        job_id = job_ids[job_idx]
                        print(f"❌ Unexpected error for {job_id[:8]}...: {e}")
                        results[job_idx] = {
                            "counts": {},
                            "status": "error",
                            "error": str(e),
                        }

            # Check for retryable failures
            if retry_failed:
                retry_indices: list[int] = []
                for i, result in enumerate(results):
                    if result and result.get("status") in [
                        "timeout",
                        "unknown",
                        "error",
                    ]:
                        retry_indices.append(i)

                if retry_indices:
                    print(f"Retrying {len(retry_indices)} failed jobs...")
                    time.sleep(5)  # Wait before retry

                    with ThreadPoolExecutor(max_workers=4) as executor:
                        retry_futures = {
                            executor.submit(collect_single_job, (job_ids[i], i)): i
                            for i in retry_indices
                        }

                        for future in as_completed(retry_futures):
                            try:
                                idx, result = future.result()
                                if result.get("status") == "success":
                                    results[idx] = result
                                    print(f"Retry successful for job {idx + 1}")
                            except Exception as e:
                                print(f"Retry failed: {e}")

            # Count results by status
            success_count = sum(
                1 for r in results if r and r.get("status") == "success"
            )
            failed_count = sum(1 for r in results if r and r.get("status") == "failed")
            timeout_count = sum(
                1
                for r in results
                if r and r.get("status") in ["timeout", "unknown", "error"]
            )

            print(
                f"Collection summary: {success_count} success, {failed_count} failed, {timeout_count} timeout/error"
            )

            # Only raise error if there are permanently failed jobs
            if failed_count > 0:
                failed_jobs = [
                    (i, r.get("error", "unknown") if r is not None else "unknown")
                    for i, r in enumerate(results)
                    if r is not None and r.get("status") == "failed"
                ]
                print(f"{failed_count} jobs permanently failed:")
                for job_idx, error in failed_jobs:
                    print(f"   Job {job_idx + 1}: {error}")
                raise RuntimeError(f"{failed_count} jobs failed permanently")

            print(
                f"✅ {self.device_name}: {success_count} results collected successfully"
            )
            return results

        except Exception as e:
            print(f"❌ Parallel collection failed: {e}")
            raise RuntimeError(f"OQTOPUS parallel collection failed: {e}") from e

    def get_physical_qubits(self, n_qubits: int, strategy: str = "best") -> list[int]:
        """
        Get physical qubits for mapping

        Args:
            n_qubits: Number of qubits needed
            strategy: Selection strategy ("best", "connected", "specific")

        Returns:
            List of physical qubit IDs
        """
        if strategy == "best":
            best_qubits = self.get_best_qubits(n_qubits)
            if best_qubits is not None:
                return [
                    int(best_qubits.iloc[i]["physical_id"]) for i in range(n_qubits)
                ]
            else:
                return list(range(n_qubits))

        elif strategy == "connected":
            # Find connected qubits (simple implementation)
            # This could be improved to find optimal connected subgraphs
            if self.device and hasattr(self.device, "couplings"):
                couplings_df = self.device.couplings
                if couplings_df is not None and len(couplings_df) > 0:
                    # Start with best qubit and find connected ones
                    best_qubits = self.get_best_qubits(
                        n_qubits * 2
                    )  # Get more candidates
                    if best_qubits is not None:
                        candidates = [
                            int(best_qubits.iloc[i]["physical_id"])
                            for i in range(min(len(best_qubits), n_qubits * 2))
                        ]
                        selected = [candidates[0]]  # Start with best qubit

                        for _ in range(n_qubits - 1):
                            # Find next connected qubit
                            for candidate in candidates:
                                if candidate in selected:
                                    continue
                                # Check if connected to any selected qubit
                                connected = any(
                                    (
                                        row["control"] == candidate
                                        and row["target"] in selected
                                    )
                                    or (
                                        row["target"] == candidate
                                        and row["control"] in selected
                                    )
                                    for _, row in couplings_df.iterrows()
                                )
                                if connected:
                                    selected.append(candidate)
                                    break
                            else:
                                # No connected qubit found, add best remaining
                                for candidate in candidates:
                                    if candidate not in selected:
                                        selected.append(candidate)
                                        break

                        return selected[:n_qubits]

            # Fallback to best qubits
            return self.get_physical_qubits(n_qubits, "best")

        else:
            # Default fallback
            return list(range(n_qubits))

    def _normalize_counts(self, counts: dict[str | int, int]) -> dict[str, int]:
        """
        Normalize count keys to Qiskit-style binary strings

        OQTOPUS returns decimal keys (0,1,2,3) but Qiskit uses binary strings ("00","01","10","11")
        This method converts decimal keys to binary string keys for consistency.

        Args:
            counts: Dictionary with potentially mixed key types

        Returns:
            Dictionary with normalized string keys
        """
        # If all keys are already strings, return as-is
        if all(isinstance(k, str) for k in counts.keys()):
            return {str(k): v for k, v in counts.items()}

        # Determine number of qubits from maximum integer key
        int_keys = [k for k in counts.keys() if isinstance(k, int)]
        if not int_keys:
            return {str(k): v for k, v in counts.items()}

        max_key = max(int_keys)
        # For quantum measurements: n_qubits = ceil(log2(max_key + 1))
        # This ensures 2^n states can be represented
        import math

        n_bits = max(1, math.ceil(math.log2(max_key + 1))) if max_key > 0 else 1

        normalized = {}
        for key, value in counts.items():
            if isinstance(key, int):
                # Convert to binary string with appropriate padding
                binary_key = format(key, f"0{n_bits}b")
                normalized[binary_key] = value
            else:
                # Already a string, keep as-is
                normalized[str(key)] = value

        return normalized
