"""ncu wrapper — run profiling, parse CSV metrics."""

import csv
import io
import re
from dataclasses import dataclass, field


@dataclass
class ProfileMetrics:
    """Structured profiling output for LLM consumption."""
    duration_us: float = 0.0
    dram_read_bytes: float = 0.0
    dram_write_bytes: float = 0.0
    sm_throughput_pct: float = 0.0
    memory_throughput_pct: float = 0.0
    warp_occupancy_pct: float = 0.0
    tensor_active_pct: float = 0.0
    raw_csv: str = ""

    def summary(self, peak_bw_gbs: float = 0, peak_tflops: float = 0,
                flops: float = 0) -> str:
        """Human-readable summary for LLM context."""
        lines = [
            f"Duration: {self.duration_us:.1f} us",
            f"DRAM read: {self.dram_read_bytes / 1e9:.2f} GB",
            f"DRAM write: {self.dram_write_bytes / 1e9:.2f} GB",
            f"SM throughput: {self.sm_throughput_pct:.1f}%",
            f"Memory throughput: {self.memory_throughput_pct:.1f}%",
            f"Warp occupancy: {self.warp_occupancy_pct:.1f}%",
            f"Tensor core active: {self.tensor_active_pct:.1f}%",
        ]
        if peak_bw_gbs > 0 and self.duration_us > 0:
            achieved_bw = (self.dram_read_bytes + self.dram_write_bytes) / (self.duration_us / 1e6) / 1e9
            lines.append(f"Achieved bandwidth: {achieved_bw:.1f} GB/s ({achieved_bw/peak_bw_gbs*100:.1f}% of peak)")
        if peak_tflops > 0 and flops > 0 and self.duration_us > 0:
            achieved = flops / (self.duration_us / 1e6) / 1e12
            lines.append(f"Achieved compute: {achieved:.1f} TFLOPS ({achieved/peak_tflops*100:.1f}% of peak)")
        sm = self.sm_throughput_pct
        mem = self.memory_throughput_pct
        if sm < 5 and mem < 5:
            lines.append("Bottleneck: LATENCY-BOUND (both utilizations near zero)")
        elif sm > mem * 1.5:
            lines.append("Bottleneck: COMPUTE-BOUND")
        elif mem > sm * 1.5:
            lines.append("Bottleneck: MEMORY-BOUND")
        elif sm > 40 or mem > 40:
            lines.append("Bottleneck: BALANCED (both SM and memory active)")
        else:
            lines.append("Bottleneck: LATENCY-BOUND (both utilizations low)")
        return "\n".join(lines)


# Metric name mapping from ncu CSV
_METRIC_MAP = {
    "gpu__time_duration.avg": "duration_us",
    "dram__bytes_read.sum": "dram_read_bytes",
    "dram__bytes_write.sum": "dram_write_bytes",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_throughput_pct",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": "memory_throughput_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "warp_occupancy_pct",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active": "tensor_active_pct",
}

QUICK_METRICS = ",".join(_METRIC_MAP.keys())


def parse_ncu_csv(csv_text: str) -> ProfileMetrics:
    """Parse ncu --csv output into ProfileMetrics.

    ncu output is mixed: kernel stdout (printf), ncu status lines (==...),
    and CSV data at the end. We find the CSV header ("ID","Process ID",...)
    and parse only from there.
    """
    metrics = ProfileMetrics(raw_csv=csv_text)

    # Find the CSV header line — always starts with "ID"
    all_lines = csv_text.strip().split("\n")
    csv_start = -1
    for i, line in enumerate(all_lines):
        if line.startswith('"ID"') or line.startswith('ID,'):
            csv_start = i
            break

    if csv_start < 0:
        return metrics

    csv_lines = all_lines[csv_start:]

    try:
        reader = csv.DictReader(io.StringIO("\n".join(csv_lines)))
        for row in reader:
            metric_name = row.get("Metric Name", "")
            value_str = row.get("Metric Value", "0")
            try:
                value = float(value_str.replace(",", ""))
            except (ValueError, TypeError):
                continue

            field_name = _METRIC_MAP.get(metric_name)
            if field_name:
                # ncu reports duration in ns, convert to us
                if field_name == "duration_us":
                    value /= 1000.0
                setattr(metrics, field_name, value)
    except Exception as e:
        import logging
        logging.getLogger("cuda-agent").warning(f"Failed to parse ncu CSV: {e}")

    return metrics


def extract_kernel_names(kernel_source: str) -> list[str]:
    """Extract __global__ function names from CUDA source code."""
    return re.findall(r'__global__\s+void\s+(?:__launch_bounds__\([^)]*\)\s+)?(\w+)\s*\(', kernel_source)


class Profiler:
    def __init__(self, sh_fn, kernel_name: str = "", agent_root: str = ""):
        self.sh = sh_fn
        self.kernel_name = kernel_name
        self.gpu_prefix = f"eval $({agent_root}/pick_gpu.sh) && " if agent_root else ""

    async def quick_profile(self, binary_path: str,
                            kernel_name: str = "") -> ProfileMetrics:
        """Run quick ncu profile with 7 key metrics.

        Profiles only 1 launch (-c 1) of the specified kernel.
        Skips the first launch (correctness check) to profile a warmed-up run.
        """
        kname = kernel_name or self.kernel_name
        k_flag = f"-k {kname}" if kname else ""
        # -c 1: profile only 1 launch
        # --launch-skip 1: skip first launch (correctness), profile second (first benchmark)
        cmd = (f"{self.gpu_prefix}ncu --csv --metrics {QUICK_METRICS} "
               f"-c 1 --launch-skip 1 {k_flag} {binary_path} 2>&1")
        stdout, stderr, code = await self.sh(cmd)
        return parse_ncu_csv(stdout + stderr)

    async def deep_profile(self, binary_path: str,
                           kernel_name: str = "") -> str:
        """Run full ncu profile, return raw output for LLM analysis."""
        kname = kernel_name or self.kernel_name
        k_flag = f"-k {kname}" if kname else ""
        cmd = (f"{self.gpu_prefix}ncu --set full -o /tmp/ncu_report "
               f"-c 1 --launch-skip 1 {k_flag} {binary_path} 2>&1")
        await self.sh(cmd)
        stdout, _, _ = await self.sh(
            "ncu -i /tmp/ncu_report.ncu-rep --csv --page raw 2>&1")
        return stdout

    async def read_sass(self, cubin_path: str) -> str:
        """Disassemble cubin for SASS analysis."""
        stdout, _, _ = await self.sh(f"cuobjdump --dump-sass {cubin_path} 2>&1")
        return stdout

    async def warp_stall_reasons(self, binary_path: str,
                                 kernel_name: str = "") -> str:
        """Get warp stall reasons for latency-bound kernels."""
        kname = kernel_name or self.kernel_name
        k_flag = f"-k {kname}" if kname else ""
        cmd = (f"{self.gpu_prefix}ncu --csv --section WarpStateStats "
               f"-c 1 --launch-skip 1 {k_flag} {binary_path} 2>&1")
        stdout, _, _ = await self.sh(cmd)
        return stdout
