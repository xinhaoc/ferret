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

import yaml

from .cost_tracker import CostTracker
from .state import RunState, compute_state
from .tools.compiler import Compiler
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
        task_yaml: str | Path | None = None,
        baseline_source: str = "",
        baseline_tflops: float = 0.0,
        arch: str = "sm_100a",
        max_iterations: int = 100,
        stall_threshold: int = 4,
        log_dir: str | Path | None = None,
    ):
        # Workspace paths — plain Path attributes, no class wrapper.
        # (v3 used Workspace.__new__ to bypass V1's lineage/attempts dirs.)
        self.workspace_path = Path(workspace_path)
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        self.kernel_path = self.workspace_path / "kernel.cu"
        self.agent_root = Path(agent_root)
        self.sh = sh_fn
        self.arch = arch
        self.max_iterations = max_iterations
        self.stall_threshold = stall_threshold
        self._client = client
        self._model_name = model_name

        # Infrastructure
        self.doc_loader = DocLoader(self.agent_root, workspace_path=self.workspace_path)
        ws_abs = str(self.workspace_path.resolve())
        self.compiler = Compiler(sh_fn, arch=arch, agent_root=str(self.agent_root), cwd=ws_abs)

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

        # Structured task spec — explicit task_yaml path wins, otherwise default
        # to workspace/task.yaml. None (legacy mode) when neither exists.
        self.spec = None
        resolved_task_yaml = (
            Path(task_yaml) if task_yaml is not None else ws_path / "task.yaml"
        )
        if resolved_task_yaml.exists():
            from .task_spec import load_task_spec
            self.spec = load_task_spec(resolved_task_yaml)
            logger.info(
                f"Loaded task spec: {self.spec.name} "
                f"({len(self.spec.configs)} configs, scoring={self.spec.scoring}) "
                f"from {resolved_task_yaml}"
            )

        # State
        self.consecutive_failures = 0

    def _make_optimizer(self):
        from .agents import create_optimizer
        return create_optimizer(
            self._client, self._model_name,
            self.workspace_path, self.kernel_path,
            self.doc_loader, self.compiler,
            self.sh, self.agent_root, self._kernel_read_flag,
        )

    # ── Scores from git ──

    async def _get_best_tflops(self) -> float:
        """Get best TFLOPS from git tags. Parses commit messages for TFLOPS: lines."""
        ws_abs = str(self.workspace_path.resolve())
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
        f = self.workspace_path / ".baseline_tflops"
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

        has_kernel = self.kernel_path.exists()

        if self.spec is not None:
            # Structured mode — describe state via compute_state (not the
            # misleading single-float max). Shows per-config picture at startup.
            state = compute_state(self.workspace_path, self.spec)
            if state.has_any_kernel:
                ratios_str = ", ".join(
                    f"{k}={v*100:.0f}%" for k, v in state.ratios.items()
                )
                logger.info(
                    f"Resuming — stage: {state.stage}, score: {state.score:.3f} "
                    f"via {self.spec.scoring} ({ratios_str})"
                )
            else:
                logger.info(
                    f"New task ({self.spec.name}) — starting in {state.stage}"
                )
        else:
            # Legacy mode — unchanged from v3
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
                err_lower = err_str.lower()
                # Reset on: (a) context overflow, (b) the known malformed-message
                # 400 that hits when memory contains an empty-content assistant
                # turn at high message counts. v3 lithos had this same bug; the
                # fix is identical — discard corrupted memory, replay first-turn.
                is_context = "too long" in err_str or "too many tokens" in err_lower
                is_malformed = (
                    "invalid_request_error" in err_lower
                    and "messages." in err_lower
                    and "content" in err_lower
                )
                if is_context or is_malformed:
                    reason = "context limit" if is_context else "malformed message 400"
                    logger.warning(f"{reason} — resetting agent and replaying first-turn")
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
            if (self.workspace_path / "STOP").exists():
                logger.info("STOP file detected.")
                (self.workspace_path / "STOP").unlink()
                break

            if result.improved:
                self.consecutive_failures = 0
            else:
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.stall_threshold:
                    await self._handle_stall()

            # Budget checks — spec.budget in structured mode, fixed fallback otherwise
            if self.spec is not None:
                max_tokens = self.spec.budget.max_tokens
                max_wall_seconds = self.spec.budget.max_wall_minutes * 60
            else:
                max_tokens = 500_000
                max_wall_seconds = 0  # 0 = disabled

            usage = getattr(self.main_agent, 'usage', {})
            total_input = sum(usage.get(k, 0) for k in
                              ['prompt_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens'])
            if total_input > max_tokens:
                logger.warning(
                    f"Token budget exceeded: {total_input:,} > {max_tokens:,}. Stopping."
                )
                break
            if max_wall_seconds and (time.time() - start) > max_wall_seconds:
                elapsed_min = (time.time() - start) / 60
                logger.warning(
                    f"Wall-time budget exceeded: {elapsed_min:.1f} min > "
                    f"{self.spec.budget.max_wall_minutes} min. Stopping."
                )
                break

        elapsed = time.time() - start
        final_best = await self._get_best_tflops()
        logger.info(f"Done in {elapsed/60:.1f} min. Best: {final_best:.1f} TFLOPS")

        # Dump motus execution trace so we can diagnose any crashes offline.
        # Purely observational — doesn't change agent behavior.
        try:
            trace_path = self.workspace_path / "motus_trace.json"
            trace = self.main_agent.get_execution_trace()
            trace_path.write_text(json.dumps(trace, default=str, indent=2))
            logger.info(f"motus trace → {trace_path}")
        except Exception as e:
            logger.warning(f"could not dump motus trace: {e}")

    # ── Turns ──

    def _render_hints_block(self) -> str:
        """Build the '## Hints (read once, then forget)' block.

        Injected into the first-turn prompt only. On subsequent iterations the
        agent no longer sees these — prevents bias from hints rotting over long
        sessions. Returns '' if no spec or no hints.
        """
        if self.spec is None or not self.spec.hints:
            return ""
        lines = ["## Hints (read once, then forget)"]
        for h in self.spec.hints:
            lines.append(f"- {h}")
        return "\n".join(lines) + "\n\n"

    async def _read_git_history(self, max_chars: int = 4000) -> str:
        """Return commit subject+body dump from workspace/.git, truncated."""
        ws_abs = str(self.workspace_path.resolve())
        out, _, _ = await self.sh(
            f"cd {ws_abs} && git log --format='%s%n%b' 2>/dev/null"
        )
        return out[:max_chars] if out else ""

    async def _first_turn(self, task: str, has_kernel: bool):
        """Single turn to load context. No forced multi-turn sequence.

        Structured mode (self.spec set): builds one template from spec fields +
        per-config header + constraints + hints + (optional) git history.
        Legacy mode: unchanged from v3 — two templates switched on has_kernel.
        """
        # Machine environment (both modes)
        env_file = self.workspace_path.parent / "env.md"
        env_section = ""
        if env_file.exists():
            env_section = f"\n## Machine Environment\n{env_file.read_text()[:2000]}\n"

        if self.spec is not None:
            # Structured path — single template, spec-driven
            state = compute_state(self.workspace_path, self.spec)
            git_history = await self._read_git_history()
            shapes_block = yaml.dump(self.spec.shapes, default_flow_style=False).rstrip()

            parts = [
                f"# Task: {self.spec.name}",
                f"GPU: {self.spec.gpu} ({self.spec.arch}) | "
                f"Precision: {self.spec.precision}",
                "",
                "## Description",
                self.spec.description.rstrip(),
                "",
                "## Shapes",
                shapes_block,
                "",
                f"## Baseline reference",
                f"`{self.spec.baseline.source}`",
                "",
                env_section.rstrip(),
                "",
                self._render_state_header(0, state),
                "",
                self._render_constraints_block().rstrip(),
                "",
                self._render_hints_block().rstrip(),
            ]

            if git_history:
                parts.extend([
                    "",
                    "## Git history (what was tried — read carefully before starting)",
                    git_history.rstrip(),
                    "",
                ])
                if state.stage == "REPRODUCE":
                    parts.append(
                        "You are resuming. Read the git history, read the current "
                        "kernel, find the architectural gap with the baseline "
                        f"source (`{self.spec.baseline.source}`), and close it."
                    )
                else:
                    parts.append(
                        "You are resuming in OPTIMIZE. Read workspace/progress.md "
                        "for plans and untried ideas. Then profile (run_ncu, "
                        "read_sass) and attack the score bottleneck. Read docs "
                        "and references you have not read before."
                    )
            else:
                # Fresh workspace, no kernel yet
                parts.extend([
                    "",
                    "New task — no prior kernel. Follow the 'Getting started' "
                    "steps in your system prompt. Spec is THIS file "
                    "(workspace/task.yaml) — do not modify it. Save your first "
                    "correct kernel with git.",
                ])

            prompt = "\n".join(p for p in parts if p is not None)
        else:
            # Legacy path — unchanged from v3 (preserved until step 7)
            stage = await self._current_stage()
            baseline = self._get_baseline_tflops()
            best = await self._get_best_tflops()

            if has_kernel:
                ws_abs = str(self.workspace_path.resolve())
                git_log_out, _, _ = await self.sh(f"cd {ws_abs} && git log --format='%s%n%b' 2>/dev/null")
                git_history = git_log_out[:4000] if git_log_out else "No git history yet."

                baseline_src = getattr(self, '_baseline_source', '')
                baseline_hint = f"Baseline source: `{baseline_src}` — you are at {best/baseline*100:.0f}% of baseline, read it to close the gap.\n" if baseline_src and baseline > 0 and stage == "REPRODUCE" else ""

                task_section = f"## Task\n{task}\n\n" if task and not task.startswith("Resume from") else ""

                # Inline spec.yaml read (v3's Workspace.spec property)
                spec_yaml_file = self.workspace_path / "spec.yaml"
                legacy_spec = spec_yaml_file.read_text() if spec_yaml_file.exists() else ""

                prompt = (
                    f"## Resuming\n\n"
                    f"**Stage: {stage}**"
                    f"{' — your best is < 90% of baseline, focus on reproducing baseline architecture' if stage == 'REPRODUCE' else ''}\n\n"
                    f"{env_section}"
                    f"{task_section}"
                    f"## Spec\n{legacy_spec}\n\n"
                    f"{baseline_hint}"
                    f"## Git History (what was tried — read carefully before starting)\n{git_history}\n\n"
                    f"Best: {best:.1f} TFLOPS"
                    f"{f' (baseline: {baseline:.1f}, {best/baseline*100:.0f}%)' if baseline > 0 else ''}\n\n"
                    f"Read the git history above. "
                    f"{'Your job is to reproduce the baseline. Try your best to read the baseline source, read the current kernel, find the differences, fix them.' if stage == 'REPRODUCE' else 'Now your job is to go beyond the baseline. Do not just copy the baseline approach — explore different architectures, instruction sets, and techniques. Read workspace/progress.md for previous plans and untried ideas. Read docs and references you have not read before (e.g. docs/patterns/, docs/ptx-isa-9.2/, or other codebases in resources/ like ThunderKittens, DeepGemm, CUTLASS).'}"
                )
            else:
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

    # ── Prompt rendering helpers (structured mode) ──

    def _render_state_header(self, iteration: int, state) -> str:
        """Build the 'Iteration N — stage / per-config table / worst' text block.

        Used when self.spec is set (structured mode). Shows every config's
        current TFLOPS vs its baseline, with ✓ marker for configs above their
        target_ratio and ← WORST marker for the score bottleneck.
        """
        # Fresh workspace (no tagged kernel yet) — don't render a table of zeros.
        # Show the baseline targets instead so the agent knows what to aim for.
        if not state.has_any_kernel:
            lines = [
                f"## Iteration {iteration} — {state.stage} stage (no kernel yet)",
                "",
                "No tagged improvement in workspace/.git. Targets to reach:",
            ]
            for cfg in self.spec.configs:
                lines.append(
                    f"  {cfg.name}: baseline {cfg.baseline_tflops:.1f} TFLOPS "
                    f"(target {cfg.target_ratio*100:.0f}%)"
                )
            return "\n".join(lines)

        lines = [
            f"## Iteration {iteration} — {state.stage} stage "
            f"(score: {state.score:.3f} via {self.spec.scoring})",
            "",
            "Per-config status:",
        ]
        for cfg in self.spec.configs:
            tflops = state.results.get(cfg.name, 0.0)
            ratio = state.ratios.get(cfg.name, 0.0)
            if ratio >= cfg.target_ratio:
                marker = " ✓"
            elif cfg.name == state.worst_config:
                marker = "  ← WORST"
            else:
                marker = ""
            lines.append(
                f"  {cfg.name}: {tflops:6.1f} / {cfg.baseline_tflops:6.1f} "
                f"= {ratio*100:5.1f}% (target {cfg.target_ratio*100:.0f}%){marker}"
            )
        if state.worst_config and state.ratios.get(state.worst_config, 0.0) < 1.0:
            lines.append("")
            lines.append(f"Focus on {state.worst_config} — it is the score bottleneck.")
        return "\n".join(lines)

    def _render_constraints_block(self) -> str:
        """Build the 'Constraints (enforced every iteration)' block.

        Re-injected every iteration so constraints cannot be forgotten in long
        sessions. Returns '' if no constraints or no spec.
        """
        if self.spec is None or not self.spec.constraints:
            return ""
        lines = ["## Constraints (enforced every iteration)"]
        for c in self.spec.constraints:
            lines.append(f"- {c}")
        return "\n".join(lines) + "\n\n"

    async def _iterate(self, iteration: int) -> IterResult:
        """One iteration. Prompt depends on stage.

        Structured mode (self.spec set): uses compute_state for a per-config
        view and re-injects constraints. Legacy mode: unchanged from v3.
        """
        if self.spec is not None:
            # Structured path — use compute_state + per-config rendering
            state = compute_state(self.workspace_path, self.spec)
            header = self._render_state_header(iteration, state)
            constraints = self._render_constraints_block()

            if state.stage == "REPRODUCE":
                advice = (
                    "You are NOT optimizing yet. Do NOT profile with ncu.\n"
                    "Focus on reproducing the baseline's architecture:\n"
                    "- If your attempt fails, compare to the baseline source "
                    "line by line and fix the difference\n"
                )
            else:
                advice = (
                    "Read workspace/progress.md first.\n"
                    "- If an idea keeps appearing in 'Untried (Hard)', stop "
                    "avoiding it. Save a checkpoint version, then commit to it. "
                    "Debugging a hard approach for many iterations is fine.\n"
                    "- If an approach keeps appearing in 'Tried', stop "
                    "repeating it. You are stuck in a loop. Try something "
                    "fundamentally different.\n"
                    "- Read a doc or reference you haven't read before "
                    "(e.g. `docs/patterns/`, `docs/ptx-isa-9.2/`, or other "
                    "codebases in `resources/` like ThunderKittens, DeepGemm, "
                    "CUTLASS).\n"
                )
            footer = (
                "\nSave improvements with git commit + tag. Save failed "
                "attempts with git commit (no tag).\n"
                "Revert after failure: "
                "`cd workspace && git checkout $(git describe --tags --abbrev=0) -- kernel.cu`\n"
            )
            prompt = f"{header}\n\n{constraints}{advice}{footer}"
        else:
            # Legacy path — unchanged from v3 (preserved until step 7)
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
        """When stuck — force research before next iteration.

        Structured mode: stage comes from compute_state (correct), constraints
        block is re-injected (same as _iterate), and the stall prompt includes
        the worst-config focus signal. Legacy mode: unchanged from v3.
        """
        if self.spec is not None:
            state = compute_state(self.workspace_path, self.spec)
            stage = state.stage
            constraints = self._render_constraints_block()
            focus = (
                f"\nCurrent bottleneck: {state.worst_config} "
                f"(ratio {state.ratios.get(state.worst_config, 0.0)*100:.1f}%). "
                f"Research should target this config first.\n"
                if state.worst_config else ""
            )
        else:
            # Legacy — unchanged from v3
            stage = await self._current_stage()
            constraints = ""
            focus = ""

        if stage == "REPRODUCE":
            body = (
                f"## STALLED in REPRODUCE stage\n\n"
                f"Your recent attempts are not making progress toward the baseline.\n\n"
                f"1. Study a DIFFERENT reference implementation than what you've been reading\n"
                f"2. Compare your kernel structure to the reference — what's different?\n"
                f"3. The gap is structural, not a tuning issue. What architectural decision is wrong?\n"
                f"4. Update progress.md: add to Tried and Untried (Hard) sections\n"
                f"{focus}\n"
                f"Do NOT edit kernel.cu in this turn. Only research and plan."
            )
        else:
            body = (
                f"## STALLED in OPTIMIZE stage\n\n"
                f"Recent optimizations are not producing gains.\n\n"
                f"1. Deep profile: run_ncu(), read_sass()\n"
                f"2. Compare your SASS to reference — what instructions differ?\n"
                f"3. Review git log — what's been tried and why it didn't help\n"
                f"4. Consider reverting to an earlier version for a different direction\n"
                f"5. Update progress.md: add to Tried and Untried (Hard) sections\n"
                f"{focus}\n"
                f"Do NOT edit kernel.cu in this turn. Only research and plan."
            )

        prompt = f"{constraints}{body}" if constraints else body

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
