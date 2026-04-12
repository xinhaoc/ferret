"""Entry point for ferret — autonomous CUDA kernel optimization agent.

Usage:
    cd ~/repos
    python -m ferret.main tasks/mla-mtp-decode-q1to4-kv4096.yaml

The task.yaml file is the sole source of truth for what the agent should
optimize: problem description, shapes, per-config baselines, constraints,
hints, budget. See tasks/template.yaml for the schema.

Resume is automatic: if the workspace directory contains a .git with tags,
ferret picks up from the latest tagged kernel. No --resume flag needed.
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from .task_spec import load_task_spec

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
    parser = argparse.ArgumentParser(
        description="ferret — autonomous CUDA kernel optimization agent",
    )
    parser.add_argument(
        "task_yaml",
        help="Path to the task spec YAML (see tasks/template.yaml for schema)",
    )
    parser.add_argument(
        "--workspace",
        default="workspace",
        help="Workspace directory for kernel.cu, progress.md, .git (default: ./workspace)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Override spec.budget.max_iterations",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override env FERRET_MODEL (default: claude-opus-4-6)",
    )
    parser.add_argument(
        "--arch",
        default=None,
        help="Override env FERRET_ARCH (default: sm_100a)",
    )
    args = parser.parse_args()

    # Load + validate the spec up front — fail fast before burning API tokens
    # on a typo.
    task_yaml_path = Path(args.task_yaml).resolve()
    if not task_yaml_path.exists():
        parser.error(f"task spec not found: {task_yaml_path}")
    try:
        spec = load_task_spec(task_yaml_path)
    except ValueError as e:
        parser.error(f"invalid task spec: {e}")

    # Validate the baseline source path referenced by the spec exists, relative
    # to the agent root (where resources/ lives). Agent will read this during
    # the run, so catch the typo now.
    baseline_path = AGENT_ROOT / spec.baseline.source
    if not baseline_path.exists():
        parser.error(
            f"baseline source not found: {baseline_path} "
            f"(referenced from {task_yaml_path})"
        )

    # Workspace dir
    ws_path = Path(args.workspace)
    if not ws_path.is_absolute():
        ws_path = AGENT_ROOT / ws_path

    model = args.model or os.environ.get("FERRET_MODEL", "claude-opus-4-6")
    arch = args.arch or os.environ.get("FERRET_ARCH", "sm_100a")
    max_iterations = args.max_iterations or spec.budget.max_iterations

    logger.info(f"Agent root   : {AGENT_ROOT}")
    logger.info(f"Workspace    : {ws_path}")
    logger.info(f"Task spec    : {task_yaml_path}")
    logger.info(f"Task name    : {spec.name}")
    logger.info(f"GPU / arch   : {spec.gpu} / {arch}")
    logger.info(f"Configs      : {[c.name for c in spec.configs]} (scoring={spec.scoring})")
    logger.info(f"Budget       : {max_iterations} iter / {spec.budget.max_wall_minutes}min / {spec.budget.max_tokens:,} tokens")

    from lithos.models import AnthropicChatClient
    client = AnthropicChatClient()

    from .orchestrator import CudaOrchestratorV2
    orchestrator = CudaOrchestratorV2(
        client=client,
        model_name=model,
        workspace_path=str(ws_path),
        sh_fn=shell_local,
        agent_root=AGENT_ROOT,
        task_yaml=task_yaml_path,
        arch=arch,
        max_iterations=max_iterations,
    )

    # Structured mode — task parameter is informational only (spec is authority).
    await orchestrator.run(task=spec.name)


if __name__ == "__main__":
    asyncio.run(main())
