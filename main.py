"""Entry point for ferret — autonomous CUDA kernel optimization agent.

Usage:
    cd ~/repos
    python -m ferret.main "Write MLA decode for B200, beat CUTLASS by 10%"
    python -m ferret.main --resume
"""

import argparse
import asyncio
import logging
import os
from pathlib import Path

# Paths
AGENT_ROOT = Path(__file__).parent.resolve()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ferret")


async def shell_local(cmd: str) -> tuple[str, str, int]:
    env_file = AGENT_ROOT / ".env"
    prefix = f"source {env_file} 2>/dev/null; " if env_file.exists() else ""
    wrapped = prefix + cmd
    proc = await asyncio.create_subprocess_exec(
        "bash", "-c", wrapped,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)
    except asyncio.TimeoutError:
        import signal
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            proc.kill()
        await proc.communicate()
        return "", "TIMEOUT: command exceeded 300s limit", 1
    return stdout.decode(), stderr.decode(), proc.returncode


async def main():
    parser = argparse.ArgumentParser(description="ferret — autonomous CUDA kernel optimization agent")
    parser.add_argument("task", nargs="?", default="")
    parser.add_argument("--workspace", default="workspace")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--model", default=None)
    parser.add_argument("--arch", default=None)
    parser.add_argument("--baseline-source", required=True,
                        help="Path to baseline source code in resources/ (e.g. resources/flashinfer-0.6.7/)")
    parser.add_argument("--baseline-tflops", type=float, default=0.0,
                        help="Baseline TFLOPS fallback (agent writes workspace/.baseline_tflops automatically)")
    args = parser.parse_args()

    model = args.model or os.environ.get("FERRET_MODEL", "claude-opus-4-6")
    arch = args.arch or os.environ.get("FERRET_ARCH", "sm_100a")

    ws_path = AGENT_ROOT / args.workspace

    # Validate baseline source path
    baseline_path = AGENT_ROOT / args.baseline_source
    if not baseline_path.exists():
        parser.error(f"Baseline source not found: {baseline_path}")

    logger.info(f"Agent root: {AGENT_ROOT}")
    logger.info(f"Workspace: {ws_path}")
    logger.info(f"Baseline: {args.baseline_source} @ {args.baseline_tflops} TFLOPS")

    sh = shell_local

    if not args.task and not args.resume:
        parser.error("Provide a task or use --resume")

    task = args.task
    if args.resume and not task:
        spec_path = ws_path / "spec.yaml"
        if spec_path.exists():
            task = f"Resume from {ws_path}"
        else:
            parser.error("No spec.yaml found. Provide a task.")

    from lithos.models import AnthropicChatClient
    client = AnthropicChatClient()

    from .orchestrator import CudaOrchestratorV2
    orchestrator = CudaOrchestratorV2(
        client=client,
        model_name=model,
        workspace_path=str(ws_path),
        sh_fn=sh,
        agent_root=AGENT_ROOT,
        baseline_source=args.baseline_source,
        baseline_tflops=args.baseline_tflops,
        arch=arch,
        max_iterations=args.max_iterations,
    )

    await orchestrator.run(task)


if __name__ == "__main__":
    asyncio.run(main())
