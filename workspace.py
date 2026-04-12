"""Workspace state management — same format as mirage-cuda-agent/workspace/.

Manages spec.yaml, kernel.cu, lineage/lineage.json, attempts/log.json, notes.md.
Interchangeable with Claude Code sessions on the same workspace.
"""

import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


class Score:
    """Result of testing + benchmarking a kernel."""

    def __init__(self, correct: bool, per_config: list[dict],
                 geomean_tflops: float, max_error: float):
        self.correct = correct
        self.per_config = per_config
        self.geomean_tflops = geomean_tflops
        self.max_error = max_error

    def __ge__(self, other):
        if isinstance(other, (int, float)):
            return self.geomean_tflops >= other
        return self.geomean_tflops >= other.geomean_tflops


class Decision:
    """Output of the analyzer agent."""

    def __init__(self, technique: str, category: str, why: str,
                 description: str, docs_read: list[str] | None = None):
        self.technique = technique
        self.category = category
        self.why = why
        self.description = description
        self.docs_read = docs_read or []


class Workspace:
    """Same format as mirage-cuda-agent/workspace/. See workflow/lineage.md."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "lineage").mkdir(exist_ok=True)
        (self.path / "attempts").mkdir(exist_ok=True)

    # -- Paths --
    @property
    def spec_path(self) -> Path:
        return self.path / "spec.yaml"

    @property
    def kernel_path(self) -> Path:
        return self.path / "kernel.cu"

    @property
    def notes_path(self) -> Path:
        return self.path / "notes.md"

    @property
    def lineage_path(self) -> Path:
        return self.path / "lineage" / "lineage.json"

    @property
    def attempts_path(self) -> Path:
        return self.path / "attempts" / "log.json"

    # -- Reads --
    @property
    def spec(self) -> str:
        return self.spec_path.read_text() if self.spec_path.exists() else ""

    @property
    def current_kernel(self) -> str:
        return self.kernel_path.read_text() if self.kernel_path.exists() else ""

    @property
    def notes(self) -> str:
        return self.notes_path.read_text() if self.notes_path.exists() else ""

    @property
    def lineage(self) -> list[dict]:
        if not self.lineage_path.exists():
            return []
        data = json.loads(self.lineage_path.read_text())
        # Handle agent writing {"versions": [...]} instead of [...]
        if isinstance(data, dict):
            return data.get("versions", data.get("entries", []))
        return data

    @property
    def attempts(self) -> list[dict]:
        if not self.attempts_path.exists():
            return []
        return json.loads(self.attempts_path.read_text())

    @property
    def best_score(self) -> float:
        entries = self.lineage
        if not entries:
            return 0.0
        def _get_tflops(e):
            return e.get("geomean_tflops", e.get("tflops_geomean", e.get("tflops", 0.0)))
        return max(_get_tflops(e) for e in entries)

    @property
    def latest_version(self) -> int:
        entries = self.lineage
        if not entries:
            return 0
        last = entries[-1]
        return last.get("version", len(entries))

    @property
    def next_id(self) -> int:
        """Next sequential ID across lineage + attempts."""
        return len(self.lineage) + len(self.attempts) + 1

    def lineage_summary(self) -> str:
        """Compact lineage for LLM context."""
        entries = self.lineage
        if not entries:
            return "No versions yet."
        lines = []
        for e in entries:
            scores = e.get("scores", [])
            latencies = ""
            if scores:
                parts = []
                for s in scores:
                    cfg = s.get("config", {})
                    cfg_str = ",".join(f"{k}={v}" for k, v in cfg.items())
                    ms = s.get("median_ms", 0)
                    tf = s.get("tflops", 0)
                    if ms > 0:
                        parts.append(f"{cfg_str}: {ms:.3f}ms/{tf:.1f}TF")
                    elif tf > 0:
                        parts.append(f"{cfg_str}: {tf:.1f}TF")
                if parts:
                    latencies = f" ({', '.join(parts)})"
            tf_str = f"{e['geomean_tflops']:.3f}" if e['geomean_tflops'] < 1.0 else f"{e['geomean_tflops']:.1f}"
            lines.append(
                f"v{e['version']:03d}: {tf_str} TFLOPS{latencies} "
                f"— {e['description']} [{e['category']}]"
            )
        return "\n".join(lines)

    def attempts_summary(self) -> str:
        """Compact attempts for LLM context."""
        entries = self.attempts
        if not entries:
            return "No failed attempts."
        lines = []
        for e in entries:
            scores = e.get("scores", [])
            latencies = ""
            if scores:
                parts = []
                for s in scores:
                    cfg = s.get("config", {})
                    cfg_str = ",".join(f"{k}={v}" for k, v in cfg.items())
                    ms = s.get("median_ms", 0)
                    tf = s.get("tflops", 0)
                    if ms > 0:
                        parts.append(f"{cfg_str}: {ms:.3f}ms/{tf:.1f}TF")
                if parts:
                    latencies = f" ({', '.join(parts)})"
            lines.append(
                f"a{e['id']:03d}: {e['description']}{latencies} [{e['category']}] "
                f"({e['status']}) — {e['notes'][:80]}"
            )
        return "\n".join(lines)

    # -- Writes --
    def save_spec(self, spec_yaml: str):
        self.spec_path.write_text(spec_yaml)

    def save_kernel(self, kernel_code: str):
        self.kernel_path.write_text(kernel_code)

    def save_notes(self, notes: str):
        self.notes_path.write_text(notes)

    def append_notes(self, text: str):
        current = self.notes
        self.notes_path.write_text(current + "\n" + text)

    def save_version(self, version: int, kernel: str, score: Score,
                     category: str, description: str, notes: str = ""):
        """Save improvement to lineage. Equivalent to save.sh improvement."""
        filename = f"v{version:03d}_{_safe_name(description)}.cu"
        (self.path / "lineage" / filename).write_text(kernel)

        entry = {
            "version": version,
            "file": filename,
            "timestamp": datetime.now().isoformat(),
            "geomean_tflops": score.geomean_tflops,
            "scores": score.per_config,
            "max_abs_error": score.max_error,
            "category": category,
            "description": description,
            "notes": notes,
        }
        entries = self.lineage
        entries.append(entry)
        self.lineage_path.write_text(json.dumps(entries, indent=2))
        self.kernel_path.write_text(kernel)

    def save_attempt(self, kernel: str, score: Score | None,
                     decision: Decision, status: str):
        """Save failed/no-gain attempt. Equivalent to save.sh attempt."""
        aid = self.next_id
        suffix = "FAIL" if status == "failed" else "NO_GAIN"
        filename = f"a{aid:03d}_{_safe_name(decision.description)}_{suffix}.cu"
        (self.path / "attempts" / filename).write_text(kernel)

        entry = {
            "id": aid,
            "file": filename,
            "status": status,
            "base_version": self.latest_version,
            "geomean_tflops": score.geomean_tflops if score else 0,
            "scores": score.per_config if score else [],
            "max_abs_error": score.max_error if score else 0,
            "category": decision.category,
            "description": decision.description,
            "notes": decision.why,
        }
        entries = self.attempts
        entries.append(entry)
        self.attempts_path.write_text(json.dumps(entries, indent=2))

    def revert_kernel(self):
        """Revert kernel.cu to latest lineage version."""
        entries = self.lineage
        if entries:
            best_file = entries[-1]["file"]
            src = self.path / "lineage" / best_file
            self.kernel_path.write_text(src.read_text())

    def init_empty(self):
        """Initialize empty workspace for a new task."""
        if not self.lineage_path.exists():
            self.lineage_path.write_text("[]")
        if not self.attempts_path.exists():
            self.attempts_path.write_text("[]")
        if not self.notes_path.exists():
            self.notes_path.write_text("# Optimization Notes\n\n")


def _safe_name(s: str, max_len: int = 30) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)[:max_len]
