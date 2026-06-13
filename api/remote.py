"""Remote-host shell wrapper.

Builds a `shell_remote(cmd)` async function that runs each command on a
remote machine via ssh, mirroring the workspace directory bidirectionally
around the call so the orchestrator's local file reads (kernel.cu, .git
state, output files) stay in sync with the remote where nvcc and the
binary actually run.

Used when ferret is launched with `--remote-host <alias>`. When the flag
is not set, main.py uses the local shell directly and this module is
never imported.
"""

import asyncio
import os
from pathlib import Path


def make_shell_remote(remote_host: str, workspace_local: Path, env_file: Path | None = None):
    """Build a shell_fn that ssh's each command to `remote_host` and rsyncs
    the workspace before/after.

    Prerequisites (one-time, performed by the user before launching):
      1. Passwordless SSH from this host to `remote_host` (e.g. via
         ~/.ssh/config alias with IdentityFile + ControlMaster).
      2. The remote machine has the agent root at the SAME absolute path
         as the local agent root (so 'cd <agent_root>' in tool commands
         resolves on both sides). Easiest: same user, same ~/repos layout.
      3. The remote has resources/ already staged (one-time
         `rsync -az resources/ <remote_host>:<agent_root>/resources/`).
      4. nvcc + CUDA driver + B200 visible to nvidia-smi on remote.

    The workspace is mirrored bidirectionally around each shell call:
      - PRE:  push local -> remote (agent's Python-side kernel.cu writes
              land on remote before nvcc sees them).
      - CMD:  ssh remote 'bash -c <cmd>'.
      - POST: pull remote -> local (binary, .git, output files come back
              so the orchestrator's local state reading stays accurate).
    """
    remote_ws = str(workspace_local.resolve())
    ssh_opts = [
        "-o", "ServerAliveInterval=30",
        "-o", "ControlMaster=auto",
        "-o", "ControlPath=~/.ssh/cm-%r@%h:%p",
        "-o", "ControlPersist=10m",
    ]
    rsync_opts = [
        "-az", "--delete",
        "-e", "ssh " + " ".join(ssh_opts),
    ]

    async def _rsync(src: str, dst: str) -> tuple[str, int]:
        proc = await asyncio.create_subprocess_exec(
            "rsync", *rsync_opts, src.rstrip("/") + "/", dst.rstrip("/") + "/",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return (stdout + stderr).decode(), proc.returncode

    async def shell_remote(cmd: str) -> tuple[str, str, int]:
        # PRE-sync: push local workspace -> remote
        _, rc = await _rsync(str(workspace_local), f"{remote_host}:{remote_ws}")
        if rc != 0:
            return "", f"rsync pre-sync to {remote_host} failed (rc={rc})", 1

        prefix = f"source {env_file} 2>/dev/null; " if env_file and env_file.exists() else ""
        wrapped = prefix + cmd

        proc = await asyncio.create_subprocess_exec(
            "ssh", *ssh_opts, remote_host, "--", "bash", "-c", wrapped,
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

        # POST-sync: pull remote workspace -> local
        _, rc_back = await _rsync(f"{remote_host}:{remote_ws}", str(workspace_local))
        # Don't fail the command on post-sync errors — surface the cmd output.
        # A post-sync warning gets stitched into stderr so the user can see it.
        post_warn = b"" if rc_back == 0 else f"\n[shell_remote] post-sync rsync rc={rc_back}\n".encode()
        return stdout.decode(), (stderr + post_warn).decode(), proc.returncode

    return shell_remote
