#!/usr/bin/env python3
"""Audit user-facing docs for unfinished placeholder markers."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path


MARKERS = ("TODO", "TBD", "待补充")
DEFAULT_TARGETS = ("docs", "README.md", "mkdocs.yml")
EXCLUDED_NAMES = {"codex-handoff.md"}


@dataclass
class Finding:
    file: Path
    line: int
    marker: str


def iter_targets(root: Path, targets: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for target in targets:
        path = root / target
        if path.is_dir():
            files.extend(
                p
                for p in sorted(path.rglob("*"))
                if p.is_file() and p.suffix in {".md", ".yml", ".yaml"}
            )
        elif path.is_file():
            files.append(path)

    return [p for p in files if p.name not in EXCLUDED_NAMES]


def audit_file(path: Path) -> list[Finding]:
    findings: list[Finding] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        for marker in MARKERS:
            if marker in line:
                findings.append(Finding(path, line_number, marker))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--target", action="append", help="path to scan; can be passed multiple times")
    args = parser.parse_args()

    targets = tuple(args.target) if args.target else DEFAULT_TARGETS
    files = iter_targets(args.root, targets)
    findings: list[Finding] = []
    for path in files:
        findings.extend(audit_file(path))

    print(f"audited_files={len(files)}")
    print(f"findings={len(findings)}")
    for finding in findings:
        print(f"{finding.file}:{finding.line}: placeholder marker: {finding.marker}")

    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
