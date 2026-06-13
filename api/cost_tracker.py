"""Cost tracking and cache logging for the CUDA agent.

Logs every LLM call with full cache breakdown to a per-session JSONL file.
Tracks cumulative cost per agent and per phase.
Supports post-hoc cache policy analysis via cache_what_if.py.
"""

import json
import os
import time
from collections import defaultdict
from pathlib import Path

# Pricing per MTok (Opus 4.6 default)
PRICING = {
    "claude-opus-4-6": {"input": 5.0, "output": 25.0, "write_5m": 6.25, "write_1h": 10.0, "read": 0.50},
    "claude-opus-4-5": {"input": 5.0, "output": 25.0, "write_5m": 6.25, "write_1h": 10.0, "read": 0.50},
    "claude-sonnet-4": {"input": 3.0, "output": 15.0, "write_5m": 3.75, "write_1h": 6.0, "read": 0.30},
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0, "write_5m": 1.25, "write_1h": 2.0, "read": 0.10},
}


def _get_pricing(model: str) -> dict:
    for key, p in PRICING.items():
        if key in model:
            return p
    return PRICING["claude-opus-4-6"]


class CostTracker:
    """Tracks cost and logs every LLM call for cache analysis."""

    def __init__(self, log_path: str | Path, model: str = "claude-opus-4-6"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.pricing = _get_pricing(model)

        # Cumulative tracking
        self.total_cost = 0.0
        self.total_no_cache = 0.0
        self.per_agent: dict[str, float] = defaultdict(float)
        self.per_phase: dict[str, float] = defaultdict(float)
        self.call_count = 0

    def record(self, agent: str, phase: str, iteration: int, usage: dict):
        """Record one LLM call.

        Args:
            agent: agent name (cuda_analyzer, cuda_coder, etc.)
            phase: phase name (plan, seed, decide, implement, debug)
            iteration: current iteration number
            usage: Anthropic usage dict with input_tokens, output_tokens,
                   cache_read_input_tokens, cache_creation_input_tokens, etc.
        """
        self.call_count += 1

        inp = usage.get("input_tokens", usage.get("prompt_tokens", 0))
        out = usage.get("output_tokens", usage.get("completion_tokens", 0))
        cache_read = usage.get("cache_read_input_tokens", 0)
        cache_write = usage.get("cache_creation_input_tokens", 0)
        cache_creation = usage.get("cache_creation", {})
        write_5m = cache_creation.get("ephemeral_5m_input_tokens", 0)
        write_1h = cache_creation.get("ephemeral_1h_input_tokens", 0)

        # If no 5m/1h breakdown, default to 1h (main agent uses 1h TTL)
        if cache_write > 0 and write_5m == 0 and write_1h == 0:
            write_1h = cache_write  # Main agent uses 1h TTL

        total_input = inp + cache_read + cache_write

        # Compute actual cost
        cost = (
            inp * self.pricing["input"]
            + cache_read * self.pricing["read"]
            + write_5m * self.pricing["write_5m"]
            + write_1h * self.pricing["write_1h"]
            + out * self.pricing["output"]
        ) / 1e6

        # Compute no-cache cost
        no_cache = (total_input * self.pricing["input"] + out * self.pricing["output"]) / 1e6

        self.total_cost += cost
        self.total_no_cache += no_cache
        self.per_agent[agent] += cost
        self.per_phase[phase] += cost

        # Write to JSONL
        entry = {
            "timestamp": time.time(),
            "call_number": self.call_count,
            "agent": agent,
            "phase": phase,
            "iteration": iteration,
            "model": self.model,
            "input_tokens": inp,
            "cache_read_tokens": cache_read,
            "cache_write_tokens": cache_write,
            "cache_write_5m": write_5m,
            "cache_write_1h": write_1h,
            "output_tokens": out,
            "total_input": total_input,
            "cost_actual": round(cost, 6),
            "cost_no_cache": round(no_cache, 6),
            "cumulative_cost": round(self.total_cost, 4),
            "cumulative_savings_pct": round(
                (1 - self.total_cost / self.total_no_cache) * 100, 1
            ) if self.total_no_cache > 0 else 0,
        }

        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def report(self) -> str:
        """Print cost summary."""
        savings = self.total_no_cache - self.total_cost
        pct = (savings / self.total_no_cache * 100) if self.total_no_cache > 0 else 0

        lines = [
            f"{'='*50}",
            f"  Cost Summary ({self.call_count} LLM calls)",
            f"{'='*50}",
            f"  Actual:    ${self.total_cost:.4f}",
            f"  No cache:  ${self.total_no_cache:.4f}",
            f"  Savings:   ${savings:.4f} ({pct:.1f}%)",
            f"",
            f"  By agent:",
        ]
        for agent, cost in sorted(self.per_agent.items(), key=lambda x: -x[1]):
            lines.append(f"    {agent:<20} ${cost:.4f}")

        lines.append(f"\n  By phase:")
        for phase, cost in sorted(self.per_phase.items(), key=lambda x: -x[1]):
            lines.append(f"    {phase:<20} ${cost:.4f}")

        return "\n".join(lines)

    def iteration_summary(self, iteration: int) -> str:
        """One-line summary for printing after each iteration."""
        savings_pct = (
            (1 - self.total_cost / self.total_no_cache) * 100
            if self.total_no_cache > 0 else 0
        )
        return (
            f"[Cost] ${self.total_cost:.4f} total "
            f"({self.call_count} calls, {savings_pct:.0f}% cache savings)"
        )
