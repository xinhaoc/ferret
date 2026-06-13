"""nvcc wrapper — compile kernel, parse errors."""

import asyncio
from dataclasses import dataclass


@dataclass
class CompileResult:
    ok: bool
    error: str = ""
    binary_path: str = ""


class Compiler:
    def __init__(self, sh_fn, arch: str = "sm_100a", includes: str = "",
                 agent_root: str = "", cwd: str = ""):
        """
        Args:
            sh_fn: async function that runs a shell command and returns (stdout, stderr, exit_code)
            arch: GPU architecture (sm_100a for B200, sm_90a for H100)
            includes: extra -I flags
            agent_root: path to cuda-agent root (for pick_gpu.sh)
            cwd: absolute path to workspace directory. All compile commands
                will run from here, so kernel.cu / kernel binary paths are
                resolved relative to the workspace regardless of the Python
                process's own cwd.
        """
        self.sh = sh_fn
        self.arch = arch
        self.includes = includes
        self.cwd_prefix = f"cd {cwd} && " if cwd else ""
        self.gpu_prefix = f"eval $({agent_root}/pick_gpu.sh) && " if agent_root else ""

    async def compile(self, kernel_path: str, output_path: str = "kernel") -> CompileResult:
        cmd = (
            f"{self.cwd_prefix}"
            f"{self.gpu_prefix}"
            f"nvcc -gencode arch=compute_{self.arch[3:]},code={self.arch} "
            f"-O3 -lineinfo {self.includes} "
            f"-o {output_path} {kernel_path} -lcuda 2>&1"
        )
        stdout, stderr, code = await self.sh(cmd)
        output = stdout + stderr
        if code == 0:
            return CompileResult(ok=True, binary_path=output_path)
        return CompileResult(ok=False, error=output)

    async def compile_cubin(self, kernel_path: str) -> CompileResult:
        """Compile to cubin for SASS analysis."""
        cmd = (
            f"{self.cwd_prefix}"
            f"{self.gpu_prefix}"
            f"nvcc --cubin -arch={self.arch} -O3 -lineinfo "
            f"{self.includes} {kernel_path} 2>&1"
        )
        stdout, stderr, code = await self.sh(cmd)
        output = stdout + stderr
        if code == 0:
            return CompileResult(ok=True, binary_path=kernel_path.replace(".cu", ".cubin"))
        return CompileResult(ok=False, error=output)
