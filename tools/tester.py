"""Correctness testing + benchmarking — runs the kernel binary and parses output."""

import json
import math
from dataclasses import dataclass

from ..workspace import Score


class Tester:
    def __init__(self, sh_fn, gpu_prefix: str = "", cwd: str = ""):
        """
        Args:
            sh_fn: async function that runs a shell command
            gpu_prefix: command prefix to set CUDA_VISIBLE_DEVICES (eval $(pick_gpu.sh))
            cwd: absolute path to workspace. Binary paths like ``./kernel``
                will be resolved from here, independent of the Python
                process's own cwd.
        """
        self.sh = sh_fn
        self.cwd_prefix = f"cd {cwd} && " if cwd else ""
        self.gpu_prefix = gpu_prefix

    async def test_and_benchmark(self, binary_path: str) -> Score:
        """Run the kernel binary with --test and --bench flags.

        The kernel binary is expected to:
        1. Run correctness checks against a reference (print JSON results)
        2. Run throughput benchmarks (print JSON results)

        This matches the scoring.md protocol from mirage-cuda-agent.
        """
        # Run correctness test
        stdout, stderr, code = await self.sh(f"{self.cwd_prefix}{self.gpu_prefix}{binary_path} 2>&1")
        output = stdout + stderr

        # Parse the output — expect JSON lines with results
        return self._parse_output(output)

    def _parse_output(self, output: str) -> Score:
        """Parse kernel binary output for correctness and benchmark results.

        Expected output format (from the kernel's main()):
        - Lines with "PASS" or "FAIL" for correctness
        - A JSON block with scores
        - A line with "Correctness:" and "Median:" for simple parsing
        """
        correct = False  # Default to incorrect — must see positive signal
        saw_correctness_line = False
        per_config = []
        tflops_values = []
        median_times = []
        geomean = 0.0
        max_error = 0.0

        for line in output.split("\n"):
            line = line.strip()

            # Parse correctness — require specific "Correctness:" or "max_err=" prefix
            if line.startswith("Correctness:") and "max_err=" in line:
                saw_correctness_line = True
                try:
                    err_str = line.split("max_err=")[1].split()[0]
                    max_error = max(max_error, float(err_str))
                except (IndexError, ValueError):
                    pass
                # Correct unless error exceeds threshold or line says FAIL
                if "FAIL" not in line:
                    correct = True
                else:
                    correct = False

            # Explicit FAIL overrides
            if "FAIL" in line and "correctness" in line.lower():
                correct = False

            # Parse TFLOPS — collect all values for geomean
            # Match both "TFLOPS: 123.4" and "TFLOPS=123.4"
            if ("TFLOPS:" in line or "TFLOPS=" in line) and not line.startswith("---"):
                try:
                    if "TFLOPS:" in line:
                        tflops_str = line.split("TFLOPS:")[1].strip().split()[0]
                    else:
                        tflops_str = line.split("TFLOPS=")[1].strip().split()[0]
                    val = float(tflops_str)
                    if val > 0:
                        tflops_values.append(val)
                except (IndexError, ValueError):
                    pass

            # Parse benchmark lines with time and TFLOPS:
            # "kv_len= 8192 | splits= 32 | time=0.643 ms | TFLOPS=3.55"
            # "M=64 N=28672 K=7168 | time=89.7 us | TFLOPS=293.1"
            if "TFLOPS=" in line and "time=" in line and "|" in line:
                try:
                    import re
                    time_m = re.search(r'time=([\d.]+)\s*(ms|us)', line)
                    tflops_m = re.search(r'TFLOPS=([\d.]+)', line)
                    if time_m and tflops_m:
                        time_val = float(time_m.group(1))
                        time_unit = time_m.group(2)
                        median_ms = time_val if time_unit == "ms" else time_val / 1000
                        # Extract config from any key=value pairs before the first |
                        config_part = line.split("|")[0]
                        config = {}
                        for kv in re.findall(r'(\w+)=\s*([\d.]+)', config_part):
                            try:
                                config[kv[0]] = int(kv[1])
                            except ValueError:
                                config[kv[0]] = float(kv[1])
                        # Fall back to kv_len for legacy format
                        kv_m = re.search(r'kv_len=\s*(\d+)', line)
                        if kv_m and "kv_len" not in config:
                            config["kv_len"] = int(kv_m.group(1))
                        per_config.append({
                            "config": config,
                            "tflops": float(tflops_m.group(1)),
                            "median_ms": median_ms,
                        })
                except Exception:
                    pass

            # Also parse "Median time: X.XXX ms" line
            if line.startswith("Median time:"):
                try:
                    median_ms = float(line.split(":")[1].strip().split()[0])
                    median_times.append(median_ms)
                except (IndexError, ValueError):
                    pass

            # Parse per-config results (if kernel outputs JSON)
            if line.startswith("{") and "tflops" in line:
                try:
                    data = json.loads(line)
                    if "config" in data and "tflops" in data:
                        per_config.append(data)
                except json.JSONDecodeError:
                    pass

        # Compute geomean from per-config if available, else from TFLOPS lines
        if per_config and all(r.get("tflops", 0) > 0 for r in per_config):
            tflops_list = [r["tflops"] for r in per_config]
            geomean = math.exp(sum(math.log(t) for t in tflops_list) / len(tflops_list))
        elif tflops_values:
            # Use first TFLOPS value (default kv_len benchmark), not last
            geomean = tflops_values[0]

        # If we never saw a correctness line, kernel didn't run properly
        if not saw_correctness_line:
            correct = False

        return Score(
            correct=correct,
            per_config=per_config,
            geomean_tflops=geomean if correct else 0.0,
            max_error=max_error,
        )
