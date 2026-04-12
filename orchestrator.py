"""ferret CUDA Kernel Optimization Orchestrator.

Two-stage design matching prompts.py:
  Stage 1: REPRODUCE — agent reproduces baseline architecture (best < 90% baseline)
  Stage 2: OPTIMIZE — agent profiles and micro-optimizes (best >= 90% baseline)

Design notes:
- No multi-turn forced sequence (no plan-docs / plan-select / plan-baselines / plan-design / seed)
- No workflow file reads injected into prompts
- No research subagents (agent reads references directly)
- Stage-aware iteration prompts
- Git-based version tracking (agent runs git commands)
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from .workspace import Workspace
from .cost_tracker import CostTracker
from .tools.compiler import Compiler
from .tools.tester import Tester
from .tools.doc_loader import DocLoader

logger = logging.getLogger("ferret")


@dataclass
class IterResult:
    improved: bool
    goal_reached: bool = False
    score: float = 0.0
    description: str = ""


class CudaOrchestratorV2:
    """Two-stage orchestrator: REPRODUCE then OPTIMIZE."""

    def __init__(
        self,
        client,
        model_name: str,
        workspace_path: str | Path,
        sh_fn,
        agent_root: str | Path,
        baseline_source: str = "",
        baseline_tflops: float = 0.0,
        arch: str = "sm_100a",
        max_iterations: int = 100,
        stall_threshold: int = 4,
        log_dir: str | Path | None = None,
    ):
        # Minimal workspace — just ensure the directory exists.
        # Don't use Workspace() constructor which creates lineage/ and attempts/ (V1 leftovers).
        ws_path = Path(workspace_path)
        ws_path.mkdir(parents=True, exist_ok=True)

        # Still use Workspace for spec/notes/kernel path helpers
        self.workspace = Workspace.__new__(Workspace)
        self.workspace.path = ws_path
        self.agent_root = Path(agent_root)
        self.sh = sh_fn
        self.arch = arch
        self.max_iterations = max_iterations
        self.stall_threshold = stall_threshold
        self._client = client
        self._model_name = model_name

        # Infrastructure
        self.doc_loader = DocLoader(self.agent_root, workspace_path=self.workspace.path)
        ws_abs = str(self.workspace.path.resolve())
        self.compiler = Compiler(sh_fn, arch=arch, agent_root=str(self.agent_root), cwd=ws_abs)
        gpu_prefix = f"eval $({self.agent_root}/pick_gpu.sh) && " if (self.agent_root / "pick_gpu.sh").exists() else ""
        self.tester = Tester(sh_fn, gpu_prefix=gpu_prefix, cwd=ws_abs)

        # Cost tracking
        log_path = Path(log_dir or workspace_path) / "cache_trace.jsonl"
        self.cost = CostTracker(log_path, model=model_name)

        # Kernel read tracking
        self._kernel_read_flag = {"read": False}

        # Agent
        self.main_agent = self._make_optimizer()

        # Logging
        self._tool_log_path = Path(log_dir or workspace_path) / "tool_calls.jsonl"
        self._conversation_log = Path(log_dir or workspace_path) / "conversation.jsonl"

        async def _log_step(content, tool_calls):
            if content:
                entry = {"timestamp": time.time(), "type": "reasoning", "content": content}
                with open(self._conversation_log, "a") as f:
                    f.write(json.dumps(entry) + "\n")
            for tc in tool_calls:
                args = tc.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {"raw": args[:500]}
                entry = {"timestamp": time.time(), "type": "tool_call", "tool": tc["name"], "args": args}
                with open(self._tool_log_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")

        self._log_step_fn = _log_step
        self.main_agent.step_callback = _log_step

        # Baseline config (from CLI args — legacy, will be removed in step 7
        # when main.py takes task.yaml as the sole source of truth)
        self._baseline_source = baseline_source
        self._baseline_tflops = baseline_tflops

        # Structured task spec — loaded if workspace/task.yaml is present.
        # None in legacy mode. Not yet consumed by any other code path;
        # steps 5/6 will wire it into prompt rendering, step 7 will make it
        # the sole source (dropping --baseline-* CLI args).
        self.spec = None
        task_yaml = ws_path / "task.yaml"
        if task_yaml.exists():
            from .task_spec import load_task_spec
            self.spec = load_task_spec(task_yaml)
            logger.info(
                f"Loaded task spec: {self.spec.name} "
                f"({len(self.spec.configs)} configs, scoring={self.spec.scoring})"
            )

        # State
        self.consecutive_failures = 0

    def _make_optimizer(self):
        from .agents import create_optimizer
        return create_optimizer(
            self._client, self._model_name,
            self.workspace, self.doc_loader, self.compiler, self.tester,
            self.sh, self.agent_root, self._kernel_read_flag,
        )

    # ── Scores from git ──

    async def _get_best_tflops(self) -> float:
        """Get best TFLOPS from git tags. Parses commit messages for TFLOPS: lines."""
        ws_abs = str(self.workspace.path.resolve())
        stdout, _, code = await self.sh(f"cd {ws_abs} && git tag 2>/dev/null")
        if code != 0 or not stdout.strip():
            return 0.0
        best = 0.0
        for tag in stdout.strip().split("\n"):
            tag = tag.strip()
            if not tag:
                continue
            msg_out, _, _ = await self.sh(f"cd {ws_abs} && git log -1 --format=%b {tag} 2>/dev/null")
            for line in msg_out.split("\n"):
                if line.startswith("TFLOPS:"):
                    try:
                        import re
                        after = line.split(":", 1)[1]
                        # Try Q\d= format first (e.g. Q1=23.8)
                        nums = re.findall(r'Q\d+=([\d.]+)', after)
                        # Fallback: standalone floats (e.g. "235.6")
                        if not nums:
                            nums = re.findall(r'(?<![=\w])(\d+\.\d+)', after)
                        for n in nums:
                            try:
                                val = float(n)
                                if val > best:
                                    best = val
                            except ValueError:
                                pass
                    except ValueError:
                        pass
        return best

    def _get_baseline_tflops(self) -> float:
        """Read baseline TFLOPS from workspace/.baseline_tflops (written by agent)."""
        f = self.workspace.path / ".baseline_tflops"
        if f.exists():
            try:
                return float(f.read_text().strip())
            except ValueError:
                pass
        return self._baseline_tflops  # fallback to CLI arg

    async def _current_stage(self) -> str:
        """Determine current stage: REPRODUCE or OPTIMIZE."""
        baseline = self._get_baseline_tflops()
        if baseline <= 0:
            return "REPRODUCE"
        best = await self._get_best_tflops()
        if best >= 0.9 * baseline:
            return "OPTIMIZE"
        return "REPRODUCE"

    # ── Main Loop ──

    async def run(self, task: str):
        logger.info(f"Starting CUDA optimization: {task}")
        start = time.time()

        has_kernel = self.workspace.kernel_path.exists()
        best = await self._get_best_tflops()
        stage = await self._current_stage()

        if has_kernel and best > 0:
            logger.info(f"Resuming ({best:.1f} TFLOPS) — stage: {stage}")
        else:
            logger.info(f"New task — starting in {stage}")

        # First turn: give context
        await self._first_turn(task, has_kernel)

        empty_iterations = 0

        for iteration in range(1, self.max_iterations + 1):
            try:
                result = await self._iterate(iteration)
                if not result.improved and not result.description.strip():
                    empty_iterations += 1
                    if empty_iterations >= 5:
                        logger.warning("5 consecutive empty iterations. Stopping.")
                        break
                    continue
                empty_iterations = 0
            except Exception as e:
                err_str = str(e)
                if "too long" in err_str or "too many tokens" in err_str.lower():
                    logger.warning(f"Context limit hit. Resetting agent...")
                    self.main_agent = self._make_optimizer()
                    self.main_agent.step_callback = self._log_step_fn
                    await self._first_turn(task, has_kernel=True)
                    empty_iterations = 0
                    continue
                logger.error(f"Iteration {iteration} failed: {e}")
                empty_iterations += 1
                if empty_iterations >= 5:
                    break
                continue

            best_now = await self._get_best_tflops()
            logger.info(f"Iter {iteration}: {'IMPROVED' if result.improved else 'no gain'} "
                        f"— {result.description} (best: {best_now:.1f})")

            if result.goal_reached:
                logger.info(f"GOAL REACHED!")
                break
            if (self.workspace.path / "STOP").exists():
                logger.info("STOP file detected.")
                (self.workspace.path / "STOP").unlink()
                break

            if result.improved:
                self.consecutive_failures = 0
            else:
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.stall_threshold:
                    await self._handle_stall()

            # Context check
            usage = getattr(self.main_agent, 'usage', {})
            total_input = sum(usage.get(k, 0) for k in
                              ['prompt_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens'])
            if total_input > 500_000:
                logger.warning(f"Context at {total_input:,} tokens. Stopping.")
                break

        elapsed = time.time() - start
        final_best = await self._get_best_tflops()
        logger.info(f"Done in {elapsed/60:.1f} min. Best: {final_best:.1f} TFLOPS")

    # ── Turns ──

    async def _first_turn(self, task: str, has_kernel: bool):
        """Single turn to load context. No forced multi-turn sequence."""
        ws = self.workspace
        stage = await self._current_stage()
        baseline = self._get_baseline_tflops()
        best = await self._get_best_tflops()

        # Machine environment
        env_file = ws.path.parent / "env.md"
        env_section = ""
        if env_file.exists():
            env_section = f"\n## Machine Environment\n{env_file.read_text()[:2000]}\n"

        if has_kernel:
            # Inject actual git log so agent sees what was tried
            ws_abs = str(ws.path.resolve())
            git_log_out, _, _ = await self.sh(f"cd {ws_abs} && git log --format='%s%n%b' 2>/dev/null")
            git_history = git_log_out[:4000] if git_log_out else "No git history yet."

            baseline_src = getattr(self, '_baseline_source', '')
            baseline_hint = f"Baseline source: `{baseline_src}` — you are at {best/baseline*100:.0f}% of baseline, read it to close the gap.\n" if baseline_src and baseline > 0 and stage == "REPRODUCE" else ""

            # If user provided a task on resume, inject it (overrides default focus)
            task_section = f"## Task\n{task}\n\n" if task and not task.startswith("Resume from") else ""

            prompt = (
                f"## Resuming\n\n"
                f"**Stage: {stage}**"
                f"{' — your best is < 90% of baseline, focus on reproducing baseline architecture' if stage == 'REPRODUCE' else ''}\n\n"
                f"{env_section}"
                f"{task_section}"
                f"## Spec\n{ws.spec}\n\n"
                f"{baseline_hint}"
                f"## Git History (what was tried — read carefully before starting)\n{git_history}\n\n"
                f"Best: {best:.1f} TFLOPS"
                f"{f' (baseline: {baseline:.1f}, {best/baseline*100:.0f}%)' if baseline > 0 else ''}\n\n"
                f"Read the git history above. "
                f"{'Your job is to reproduce the baseline. Try your best to read the baseline source, read the current kernel, find the differences, fix them.' if stage == 'REPRODUCE' else 'Now your job is to go beyond the baseline. Do not just copy the baseline approach — explore different architectures, instruction sets, and techniques. Read workspace/progress.md for previous plans and untried ideas. Read docs and references you have not read before (e.g. docs/patterns/, docs/ptx-isa-9.2/, or other codebases in resources/ like ThunderKittens, DeepGemm, CUTLASS).'}"
            )
        else:
            # New task
            prompt = (
                f"## New Task\n\n{task}\n\n"
                f"**Stage: REPRODUCE**\n\n"
                f"{env_section}"
                f"Follow the 'Getting started' steps in your system prompt:\n"
                f"1. Read the architecture doc for the target GPU\n"
                f"2. Measure baselines\n"
                f"3. Write spec.yaml (see examples/specs/ for format)\n"
                f"4. Study references and write your first kernel\n\n"
                f"Work through these steps. Save your first correct kernel with git."
            )

        self._log_conversation("prompt", prompt, "first-turn")
        await self._retry_agent(self.main_agent, prompt, label="first-turn")

    async def _iterate(self, iteration: int) -> IterResult:
        """One iteration. Prompt depends on stage."""
        stage = await self._current_stage()
        baseline = self._get_baseline_tflops()
        best = await self._get_best_tflops()
        pct = f"{best/baseline*100:.0f}% of baseline" if baseline > 0 else "unknown"
        baseline_hint = f"Baseline source: `{self._baseline_source}` — you are at {pct}, read it to close the gap.\n" if getattr(self, '_baseline_source', '') else ""

        if stage == "REPRODUCE":
            prompt = (
                f"## Iteration {iteration} — REPRODUCE stage ({pct})\n\n"
                f"Best: {best:.1f} TFLOPS. Baseline: {baseline:.1f} TFLOPS. At {pct}.\n\n"
                f"{baseline_hint}"
                f"You are NOT optimizing yet. Do NOT profile with ncu.\n"
                f"Focus on reproducing the baseline's architecture:\n"
                f"- If your attempt fails, compare to the baseline source line by line and fix the difference\n\n"
                f"Save improvements with git commit + tag. Save failed attempts with git commit (no tag).\n"
                f"Revert after failure: `cd workspace && git checkout $(git describe --tags --abbrev=0) -- kernel.cu`\n"
            )
        else:
            prompt = (
                f"## Iteration {iteration} — OPTIMIZE stage ({pct} of baseline)\n\n"
                f"Best: {best:.1f} TFLOPS. Baseline: {baseline:.1f} TFLOPS.\n\n"
                f"Read workspace/progress.md first.\n"
                f"- If an idea keeps appearing in 'Untried (Hard)', stop avoiding it. Save a checkpoint version, then commit to it. Debugging a hard approach for many iterations is fine.\n"
                f"- If an approach keeps appearing in 'Tried', stop repeating it. You are stuck in a loop. Try something fundamentally different.\n"
                f"- Read a doc or reference you haven't read before (e.g. `docs/patterns/`, `docs/ptx-isa-9.2/`, or other codebases in `resources/` like ThunderKittens, DeepGemm, CUTLASS).\n\n"
                f"Save improvements with git commit + tag. Save failed attempts with git commit (no tag).\n"
                f"Revert after failure: `cd workspace && git checkout $(git describe --tags --abbrev=0) -- kernel.cu`\n"
            )

        best_before = await self._get_best_tflops()
        self._log_conversation("prompt", prompt, f"iter-{iteration}")
        result_text = await self._retry_agent(self.main_agent, prompt, label=f"iter-{iteration}")

        best_after = await self._get_best_tflops()
        improved = best_after > best_before

        return IterResult(
            improved=improved,
            score=best_after,
            description=str(result_text)[:100] if result_text else "",
        )

    async def _handle_stall(self):
        """When stuck — force research before next iteration."""
        stage = await self._current_stage()

        if stage == "REPRODUCE":
            prompt = (
                f"## STALLED in REPRODUCE stage\n\n"
                f"Your recent attempts are not making progress toward the baseline.\n\n"
                f"1. Study a DIFFERENT reference implementation than what you've been reading\n"
                f"2. Compare your kernel structure to the reference — what's different?\n"
                f"3. The gap is structural, not a tuning issue. What architectural decision is wrong?\n"
                f"4. Update progress.md: add to Tried and Untried (Hard) sections\n\n"
                f"Do NOT edit kernel.cu in this turn. Only research and plan."
            )
        else:
            prompt = (
                f"## STALLED in OPTIMIZE stage\n\n"
                f"Recent optimizations are not producing gains.\n\n"
                f"1. Deep profile: run_ncu(), read_sass()\n"
                f"2. Compare your SASS to reference — what instructions differ?\n"
                f"3. Review git log — what's been tried and why it didn't help\n"
                f"4. Consider reverting to an earlier version for a different direction\n"
                f"5. Update progress.md: add to Tried and Untried (Hard) sections\n\n"
                f"Do NOT edit kernel.cu in this turn. Only research and plan."
            )

        self._log_conversation("prompt", prompt, "stall-research")
        await self._retry_agent(self.main_agent, prompt, label="stall-research")
        self.consecutive_failures = 0

    # ── Helpers ──

    def _log_conversation(self, entry_type: str, content: str, label: str = ""):
        entry = {"timestamp": time.time(), "type": entry_type, "label": label, "content": content}
        with open(self._conversation_log, "a") as f:
            f.write(json.dumps(entry) + "\n")

    async def _retry_agent(self, agent, prompt, label="agent"):
        self._log_conversation("prompt", prompt, label)
        attempt = 0
        while True:
            try:
                result = await agent(prompt)
                result_str = str(result) if result else ""
                self._log_conversation("response", result_str, label)

                usage = getattr(agent, 'usage', {})
                mem = getattr(agent, '_memory', None)
                msg_count = len(mem._messages) if mem and hasattr(mem, '_messages') else '?'
                logger.info(
                    f"[{label}] result_len={len(result_str)} "
                    f"msgs={msg_count} "
                    f"prompt_tokens={usage.get('prompt_tokens', 0):,} "
                    f"completion_tokens={usage.get('completion_tokens', 0):,}"
                )
                return result
            except Exception as e:
                err = str(e).lower()
                if "overloaded" in err or "timeout" in err or "connect" in err:
                    wait = min(30 * (2 ** attempt), 600)
                    attempt += 1
                    logger.warning(f"{label}: {err[:80]}, retry #{attempt} in {wait}s...")
                    await asyncio.sleep(wait)
                elif "max steps" in err:
                    logger.warning(f"{label}: hit max steps, returning partial")
                    return "Agent reached step limit."
                else:
                    raise
