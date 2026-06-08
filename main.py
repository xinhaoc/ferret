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


# shell_remote (ssh + rsync wrapper for routing GPU work to a remote host)
# lives in remote.py — imported lazily only when --remote-host is set.


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
    parser.add_argument(
        "--remote-host",
        default=None,
        help=("If set, ssh every shell command to this host and rsync the "
              "workspace before/after each call. Use an SSH config alias "
              "(e.g. 'nebius-b200') with ControlMaster for speed. The remote "
              "must have the agent root at the same absolute path as local "
              "and resources/ already staged. Default: run all shells locally."),
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

    # Validate reference paths exist. `baseline.source` is a name (e.g. "cuBLAS",
    # "trtllm-gen") — the scoring target, not a readable path. `references` is
    # the REPRODUCE reading list and each entry MUST resolve to a file/dir.
    for ref in spec.references:
        ref_path = AGENT_ROOT / ref
        if not ref_path.exists():
            parser.error(
                f"reference path not found: {ref_path} "
                f"(from references[] in {task_yaml_path})"
            )

    # Workspace dir
    ws_path = Path(args.workspace)
    if not ws_path.is_absolute():
        ws_path = AGENT_ROOT / ws_path

    model = args.model or os.environ.get("FERRET_MODEL", "claude-opus-4-6")
    arch = args.arch or os.environ.get("FERRET_ARCH", "sm_100a")
    max_iterations = args.max_iterations or spec.budget.max_iterations

    # Pick which shell to use. Default = local (unchanged behavior). With
    # --remote-host, every shell command goes through ssh + rsync. Selection
    # happens BEFORE the orchestrator is constructed so the tool layer
    # doesn't need to know which one it got.
    if args.remote_host:
        from .remote import make_shell_remote
        sh_fn = make_shell_remote(args.remote_host, ws_path, env_file=AGENT_ROOT / ".env")
        logger.info(f"Remote host  : {args.remote_host} (workspace mirrored to same path)")
    else:
        sh_fn = shell_local

    logger.info(f"Agent root   : {AGENT_ROOT}")
    logger.info(f"Workspace    : {ws_path}")
    logger.info(f"Task spec    : {task_yaml_path}")
    logger.info(f"Task name    : {spec.name}")
    logger.info(f"GPU / arch   : {spec.gpu} / {arch}")
    logger.info(f"Configs      : {[c.name for c in spec.configs]} (scoring={spec.scoring})")
    logger.info(f"Budget       : {max_iterations} iter / {spec.budget.max_wall_minutes}min")

    from motus.models import AnthropicChatClient
    client = AnthropicChatClient()

    from .orchestrator import CudaOrchestratorV2
    orchestrator = CudaOrchestratorV2(
        client=client,
        model_name=model,
        workspace_path=str(ws_path),
        sh_fn=sh_fn,
        agent_root=AGENT_ROOT,
        task_yaml=task_yaml_path,
        arch=arch,
        max_iterations=max_iterations,
    )

    # Structured mode — task parameter is informational only (spec is authority).
    await orchestrator.run(task=spec.name)


if __name__ == "__main__":
    asyncio.run(main())
