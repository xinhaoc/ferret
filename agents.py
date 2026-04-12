"""Agent creation for ferret — matches the two-stage REPRODUCE/OPTIMIZE design in prompts.py.

- Uses prompts.OPTIMIZER_PROMPT
- No save_version/save_attempt tools (agent uses git via run_command)
- No research subagents (agent reads references directly)
"""

from pathlib import Path

from lithos.agent import ReActAgent
from lithos.models.base import CachePolicy
from lithos.tools import FunctionTool

from .prompts import OPTIMIZER_PROMPT
from .tools.profiler import extract_kernel_names


def create_optimizer(
    client,
    model_name: str,
    workspace,
    doc_loader,
    compiler,
    tester,
    sh_fn,
    agent_root: Path,
    kernel_read_flag: dict,
) -> ReActAgent:
    """Create the main optimizer agent with all tools."""
    ws = workspace
    doc = doc_loader
    sh = sh_fn

    _kernel_cache = {"content": None}

    async def _read_kernel_from_disk() -> str:
        content = ws.current_kernel
        _kernel_cache["content"] = content
        return content

    async def _write_kernel_to_disk(code: str) -> None:
        ws.save_kernel(code)
        _kernel_cache["content"] = code

    # -- Tools --

    async def write_kernel(kernel_code: str) -> str:
        """Write full kernel.cu from scratch. Use ONLY for the initial write.
        For changes, use edit_kernel() instead."""
        await _write_kernel_to_disk(kernel_code)
        kernel_read_flag["read"] = False
        return f"Written {len(kernel_code.splitlines())} lines to kernel.cu."

    async def edit_kernel(old_string: str, new_string: str) -> str:
        """Edit kernel.cu by replacing old_string with new_string.
        You MUST call read_kernel() first before editing.
        old_string must match exactly one location in the file."""
        if not kernel_read_flag["read"]:
            return "Error: you must call read_kernel() before editing."
        kernel = _kernel_cache["content"] or await _read_kernel_from_disk()
        if not kernel:
            return "Error: no kernel.cu to edit."
        if old_string not in kernel:
            return f"Error: old_string not found in kernel.cu. Read the file again.\nSearched for: {old_string[:200]}"
        count = kernel.count(old_string)
        if count > 1:
            return f"Error: old_string matches {count} locations. Provide more context."
        if old_string == new_string:
            return "Error: old_string and new_string are identical."
        new_kernel = kernel.replace(old_string, new_string, 1)
        await _write_kernel_to_disk(new_kernel)
        kernel_read_flag["read"] = False
        changed = len(new_string.splitlines()) - len(old_string.splitlines())
        return f"Edited kernel.cu: {changed:+d} lines. Total: {len(new_kernel.splitlines())} lines."

    async def read_kernel() -> str:
        """Read the current kernel.cu source code. Must call before edit_kernel()."""
        kernel_read_flag["read"] = True
        content = await _read_kernel_from_disk()
        return content or "No kernel.cu found."

    ws_abs = str(workspace.path.resolve())

    async def run_ncu(kernel_name: str = "") -> str:
        """Run full ncu profiling on current compiled kernel binary."""
        if not kernel_name:
            src = _kernel_cache["content"] or ""
            names = extract_kernel_names(src) if src else []
            kernel_name = names[0] if names else ""
        k_flag = f"-k {kernel_name}" if kernel_name else ""
        cmd = f"cd {ws_abs} && ncu --set full -c 1 --launch-skip 1 {k_flag} ./kernel 2>&1"
        stdout, stderr, code = await sh(cmd)
        return stdout + stderr

    agent_root_abs = str(agent_root.resolve())

    async def run_command(command: str) -> str:
        """Run a shell command (compile, benchmark, git, etc.).
        Runs in the agent root directory (where baselines/, examples/, docs/, workspace/ are).
        kernel.cu and git repo are in workspace/. Use 'cd workspace && git ...' for git commands."""
        stdout, stderr, code = await sh(f"cd {agent_root_abs} && {command}")
        return f"exit_code={code}\n{stdout + stderr}"

    async def read_docs(path: str) -> str:
        """Read a documentation file. Example: read_docs('docs/architecture/blackwell-b200.md')"""
        return doc.read_doc(path)

    async def read_reference(path: str, offset: int = 0) -> str:
        """Read a reference implementation file."""
        return doc.read_reference(path, offset)

    async def grep_reference(pattern: str, path: str = "") -> str:
        """Search for a pattern in reference code."""
        return doc.grep_reference(pattern, path)

    async def read_mapping() -> str:
        """Read docs/MAPPING.md — maps topics to reference files."""
        return doc.read_mapping()

    async def read_sass() -> str:
        """Analyze SASS of current kernel. Returns instruction counts, register usage, spills — not raw dump.
        Use run_command('cuobjdump --dump-sass workspace/kernel.cubin') if you need the full raw SASS."""
        await compiler.compile_cubin("kernel.cu")
        results = []

        # Instruction counts
        out, _, _ = await sh(f"cd {ws_abs} && cuobjdump --dump-sass kernel.cubin | grep -oP '^\\s+/\\*[^*]+\\*/\\s+\\K\\S+' | sort | uniq -c | sort -rn | head -25")
        results.append("=== Instruction Counts ===\n" + out)

        # Register spills
        out, _, _ = await sh(f"cd {ws_abs} && cuobjdump --dump-sass kernel.cubin | grep -c 'STL\\|LDL'")
        results.append(f"=== Spills (STL/LDL) === {out.strip()}")

        # Resource usage
        out, _, _ = await sh(f"cd {ws_abs} && cuobjdump --dump-resource-usage kernel.cubin")
        results.append("=== Resource Usage ===\n" + out)

        return "\n".join(results)

    async def list_files(path: str = "") -> str:
        """List files in a directory relative to agent root."""
        target = str(agent_root / path) if path else str(agent_root)
        stdout, stderr, _ = await sh(f"ls -la {target} 2>&1")
        return stdout + stderr

    async def glob_files(pattern: str) -> str:
        """Find files matching a glob pattern."""
        import glob as g
        matches = sorted(g.glob(str(agent_root / pattern), recursive=True))
        return "\n".join(str(Path(m).relative_to(agent_root)) for m in matches) or "No matches."

    async def think(thought: str) -> str:
        """Think through a problem. Use for complex reasoning before deciding."""
        return "Thought recorded."

    return ReActAgent(
        name="ferret_optimizer",
        client=client,
        model_name=model_name,
        system_prompt=OPTIMIZER_PROMPT,
        tools={
            "write_kernel": FunctionTool(write_kernel),
            "edit_kernel": FunctionTool(edit_kernel),
            "read_kernel": FunctionTool(read_kernel),
            "run_ncu": FunctionTool(run_ncu),
            "run_command": FunctionTool(run_command),
            "read_docs": FunctionTool(read_docs),
            "read_reference": FunctionTool(read_reference),
            "grep_reference": FunctionTool(grep_reference),
            "read_mapping": FunctionTool(read_mapping),
            "read_sass": FunctionTool(read_sass),
            "list_files": FunctionTool(list_files),
            "glob_files": FunctionTool(glob_files),
            "think": FunctionTool(think),
        },
        max_steps=80,
        cache_policy=CachePolicy.STATIC_LONG,
    )
