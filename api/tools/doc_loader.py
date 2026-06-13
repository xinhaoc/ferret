"""Documentation loader — provides tools for LLM agents to read docs and references."""

import json
from pathlib import Path


class DocLoader:
    """Loads docs, patterns, architecture specs, and reference code.

    Provides tool functions that LLM agents can call to read specific files
    from the knowledge base without the LLM needing to know the file structure.

    Tracks read counts across sessions in workspace/file_reads.json.
    """

    def __init__(self, agent_root: Path, workspace_path: Path | None = None):
        self.root = agent_root
        self.docs_dir = agent_root / "docs"
        self.resources_dir = agent_root / "resources"
        self.workflow_dir = agent_root / "workflow"
        self._reads_file = workspace_path / "file_reads.json" if workspace_path else None
        self._reads_cache = self._load_reads()

    def _load_reads(self) -> dict:
        if self._reads_file and self._reads_file.exists():
            try:
                return json.loads(self._reads_file.read_text())
            except Exception:
                pass
        return {}

    def _record_read(self, path: str) -> int:
        """Increment read count, persist, return new count."""
        self._reads_cache[path] = self._reads_cache.get(path, 0) + 1
        count = self._reads_cache[path]
        if self._reads_file:
            try:
                self._reads_file.write_text(json.dumps(self._reads_cache, indent=2))
            except Exception:
                pass
        return count

    def _read_count_footer(self, path: str, count: int) -> str:
        return f"\n\n---\nRead count: {count} (across all sessions)"

    def read_doc(self, path: str) -> str:
        """Read a doc file. Path relative to agent root.

        Examples:
            read_doc("docs/architecture/blackwell-b200.md")
            read_doc("docs/patterns/memory-access.md")
            read_doc("docs/profiling/ncu-metrics.md")
        """
        full = self.root / path
        if not full.exists():
            return f"File not found: {path}"
        count = self._record_read(path)
        content = full.read_text()
        if full.stat().st_size > 2_000_000:
            content = content[:2_000_000] + "\n\n[TRUNCATED — file too large]"
        return content + self._read_count_footer(path, count)

    def read_reference(self, path: str, offset: int = 0) -> str:
        """Read a reference implementation file in full (from offset to end).

        Examples:
            read_reference("resources/flashmla-main/csrc/sm100/decode/head64/kernel.cuh")
            read_reference("resources/cutlass-4.4.2/include/cute/arch/mma_sm100_umma.hpp")
        """
        full = self.root / path
        if not full.exists():
            return f"File not found: {path}"
        count = self._record_read(path)
        lines = full.read_text().split("\n")
        selected = lines[offset:]
        header = f"[{path} lines {offset+1}-{offset+len(selected)} of {len(lines)}]\n"
        content = header + "\n".join(f"{i+offset+1}\t{line}" for i, line in enumerate(selected))
        return content + self._read_count_footer(path, count)

    def grep_reference(self, pattern: str, path: str = "", max_results: int = 100) -> str:
        """Search for a pattern in reference code.

        Examples:
            grep_reference("tcgen05.mma", "resources/cutlass-4.4.2/include/cute/arch")
            grep_reference("blockscaled", "resources/deepgemm-2.1.1.post3")
        """
        import subprocess
        search_path = str(self.root / path) if path else str(self.resources_dir)
        try:
            result = subprocess.run(
                ["grep", "-rn", pattern, search_path],
                capture_output=True, text=True, timeout=10)
            lines = result.stdout.strip().split("\n")[:max_results]
            return "\n".join(lines) if lines[0] else "No matches found."
        except Exception as e:
            return f"Search failed: {e}"

    def list_docs(self, subdir: str = "") -> str:
        """List available docs in a directory.

        Examples:
            list_docs("docs/patterns")
            list_docs("docs/architecture")
        """
        target = self.root / subdir if subdir else self.docs_dir
        if not target.exists():
            return f"Directory not found: {subdir}"
        files = sorted(target.rglob("*.md"))
        return "\n".join(str(f.relative_to(self.root)) for f in files)

    def read_mapping(self) -> str:
        """Read docs/MAPPING.md — maps topics to reference files."""
        return self.read_doc("docs/MAPPING.md")

    def read_index(self) -> str:
        """Read docs/INDEX.md — master index of all knowledge."""
        return self.read_doc("docs/INDEX.md")
