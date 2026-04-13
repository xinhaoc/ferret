#!/usr/bin/env python3
"""Verify every reference to a resources/<submodule>/<path> in the ferret
source tree resolves to an existing file or directory in the pinned
submodule.

Run this:
  - After bumping a submodule version
  - After renaming a submodule directory (e.g. version suffix change)
  - After moving files around in docs/
  - As a sanity check before pushing

Exit code 0 on clean, 1 if any ref is broken. Stdlib only, no deps.

Usage:
  python scripts/check_resource_refs.py                 # check everything
  python scripts/check_resource_refs.py --verbose       # show per-submodule counts
  python scripts/check_resource_refs.py --root /path    # check a different tree

What it scans:
  All *.md and *.py files under --root, excluding anything inside
  resources/ itself and inside .git/. Captures path expressions of the
  form `[resources/]<submodule-name>[/<path>]` where <submodule-name>
  is a directory under resources/ (including version suffixes).

What it reports:
  - Total references scanned
  - Number resolved / missing
  - For each missing ref: the expected path + the deepest existing parent
    + up to 3 referencing files

Typical workflow after a submodule bump:
  1. Update .gitmodules + bump the directory suffix (git mv)
  2. Run this script → expect failures pointing at old version suffix
  3. Use the failures to drive sed/python replacements across docs + tools
  4. Run again → expect PASS
  5. Commit
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path


def find_submodules(root: Path) -> tuple[list[str], set[str], set[str]]:
    """Return (names, on_disk, declared).

    names: union of on-disk dirs and `.gitmodules` path entries, sorted.
    on_disk: submodule directory names that exist under resources/.
    declared: submodule directory names declared in .gitmodules.

    Using the union catches drift in both directions: a source ref that
    points at an old directory name that's been renamed, and a
    .gitmodules entry whose directory was deleted by accident.
    """
    resources = root / "resources"
    on_disk: set[str] = set()
    if resources.is_dir():
        on_disk = {p.name for p in resources.iterdir() if p.is_dir()}

    declared: set[str] = set()
    gitmodules = root / ".gitmodules"
    if gitmodules.is_file():
        for line in gitmodules.read_text().splitlines():
            line = line.strip()
            if line.startswith("path =") or line.startswith("path="):
                path_val = line.split("=", 1)[1].strip()
                # Only care about entries under resources/
                if path_val.startswith("resources/"):
                    declared.add(path_val.split("/", 1)[1])

    names = sorted(on_disk | declared)
    return names, on_disk, declared


def build_ref_regex(submodule_names: list[str]) -> re.Pattern:
    """Regex that captures (submodule_name, optional /path).

    Longest names first so `flash-attention-fa4-v4.0.0.beta8` is preferred
    over `flash-attention` (if both were ever present).

    Negative lookbehind on `[A-Za-z0-9._\\-/]` prevents matching inside
    a longer word like `my-deepgemm-2.1.1` (would match deepgemm-2.1.1
    otherwise and falsely flag it). Allows an optional `resources/`
    prefix since some docs write bare submodule-name paths.
    """
    names_alt = "|".join(re.escape(s) for s in sorted(submodule_names, key=len, reverse=True))
    return re.compile(
        rf'(?<![A-Za-z0-9._\-/])(?:resources/)?({names_alt})(/[A-Za-z0-9._\-/]*)?'
    )


def collect_refs(
    root: Path, ref_re: re.Pattern
) -> dict[str, set[tuple[str, str]]]:
    """Walk the tree, return {submodule: {(subpath, source-file)}}."""
    refs: dict[str, set[tuple[str, str]]] = defaultdict(set)
    extensions = ("*.md", "*.py", "*.yaml", "*.yml")
    files: list[Path] = []
    for ext in extensions:
        files.extend(root.rglob(ext))
    for f in files:
        # Skip anything inside resources/ itself or inside git metadata
        rel = f.relative_to(root)
        if rel.parts and (rel.parts[0] == "resources" or rel.parts[0] == ".git"):
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for m in ref_re.finditer(text):
            submodule = m.group(1)
            subpath = (m.group(2) or "").lstrip("/").rstrip("/.,;)")
            refs[submodule].add((subpath, str(rel)))
    return refs


def check(root: Path, verbose: bool = False) -> int:
    resources = root / "resources"
    names, on_disk, declared = find_submodules(root)
    if not names:
        print(f"ERROR: no submodules found under {resources} or in .gitmodules",
              file=sys.stderr)
        return 1

    # Drift checks between .gitmodules and the filesystem
    drift_errors = 0
    declared_missing_disk = declared - on_disk
    disk_missing_declared = on_disk - declared
    if declared_missing_disk:
        drift_errors += len(declared_missing_disk)
        print(f"FAIL — {len(declared_missing_disk)} submodule(s) declared "
              f"in .gitmodules but missing on disk:")
        for n in sorted(declared_missing_disk):
            print(f"  ✗ resources/{n}  (run `git submodule update --init resources/{n}`)")
        print()
    if disk_missing_declared:
        drift_errors += len(disk_missing_declared)
        print(f"FAIL — {len(disk_missing_declared)} directory/ies under "
              f"resources/ not declared in .gitmodules:")
        for n in sorted(disk_missing_declared):
            print(f"  ✗ resources/{n}  (orphaned — add to .gitmodules or remove)")
        print()

    ref_re = build_ref_regex(names)
    refs = collect_refs(root, ref_re)

    # Verify each (submodule, subpath)
    ok = 0
    missing: list[tuple[str, str, str, str]] = []  # (sm, sp, where, hint)
    per_submodule_ok: defaultdict[str, int] = defaultdict(int)

    for submodule, path_refs in refs.items():
        sub_dir = resources / submodule
        for subpath, where in path_refs:
            if not sub_dir.exists():
                missing.append((submodule, subpath, where,
                                "submodule directory does not exist"))
                continue
            if subpath == "":
                ok += 1
                per_submodule_ok[submodule] += 1
                continue
            target = sub_dir / subpath
            if target.exists():
                ok += 1
                per_submodule_ok[submodule] += 1
            else:
                parent = target.parent
                while parent != sub_dir and not parent.exists():
                    parent = parent.parent
                hint = ""
                if parent != sub_dir and parent.exists():
                    hint = f"parent exists: {parent.relative_to(resources)}"
                missing.append((submodule, subpath, where, hint))

    total = ok + len(missing)

    if verbose:
        print(f"scanned {total} unique (submodule, path) references")
        print(f"{'submodule':42} ok  status")
        print(f"{'-'*42} ---  ------")
        for sm in names:
            c = per_submodule_ok.get(sm, 0)
            status = []
            if sm in on_disk and sm in declared:
                status.append("tracked")
            elif sm in on_disk:
                status.append("orphan dir")
            elif sm in declared:
                status.append("not cloned")
            if c == 0:
                status.append("unused")
            print(f"{sm:42} {c:3}  {', '.join(status)}")
        print()

    if missing:
        # Group by unique (submodule, subpath) — a single missing path can
        # be referenced from several source files.
        grouped: defaultdict[tuple[str, str, str], list[str]] = defaultdict(list)
        for sm, sp, where, hint in missing:
            grouped[(sm, sp, hint)].append(where)

        print(f"FAIL — {len(missing)} broken ref(s), {len(grouped)} unique path(s):")
        print()
        for (sm, sp, hint), wheres in sorted(grouped.items()):
            print(f"  ✗ resources/{sm}/{sp}" if sp else f"  ✗ resources/{sm}")
            if hint:
                print(f"      ({hint})")
            shown = wheres[:3]
            for w in shown:
                print(f"      referenced in: {w}")
            if len(wheres) > len(shown):
                print(f"      ... and {len(wheres) - len(shown)} more")
            print()
        print(f"summary: {ok}/{total} refs resolve, {len(missing)} broken, "
              f"{drift_errors} submodule drift issue(s)")
        return 1

    if drift_errors:
        print(f"FAIL — all refs resolve but {drift_errors} submodule drift issue(s)")
        return 1

    print(f"PASS — all {ok} refs resolve across {len(on_disk)} submodule(s), "
          f"no .gitmodules drift")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parent.parent),
        help="Tree to scan (default: the ferret repo root)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show per-submodule reference count",
    )
    args = parser.parse_args()
    return check(Path(args.root).resolve(), verbose=args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
