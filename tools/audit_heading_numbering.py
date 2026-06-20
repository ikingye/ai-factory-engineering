#!/usr/bin/env python3
"""Audit chapter-level H2 heading numbering policy.

Every H2 in a chapter must be numbered consecutively with the local chapter
prefix, for example "21.1 本章回答的问题" and "21.16 NVIDIA GPU Container 原理".
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


CHAPTER_RE = re.compile(r"chapter-(\d+)-")
H2_RE = re.compile(r"^##\s+(.+?)\s*$")
NUMBERED_H2_RE = re.compile(r"^(\d+)\.(\d+)\s+.+")


@dataclass
class Finding:
    file: Path
    line: int
    message: str


def expected_chapter_number(path: Path) -> int | None:
    match = CHAPTER_RE.search(path.name)
    if not match:
        return None
    return int(match.group(1))


def audit_file(path: Path) -> list[Finding]:
    chapter = expected_chapter_number(path)
    findings: list[Finding] = []
    expected_index = 1

    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        match = H2_RE.match(line)
        if not match:
            continue

        heading = match.group(1).strip()
        numbered = NUMBERED_H2_RE.match(heading)
        if not numbered:
            findings.append(Finding(path, line_number, f"unnumbered H2: {heading}"))
            continue

        if chapter is not None and int(numbered.group(1)) != chapter:
            findings.append(
                Finding(
                    path,
                    line_number,
                    f"wrong chapter prefix: {heading}; expected {chapter}.x",
                )
            )

        actual_index = int(numbered.group(2))
        if actual_index != expected_index:
            findings.append(
                Finding(
                    path,
                    line_number,
                    f"non-consecutive H2: {heading}; expected {chapter}.{expected_index}",
                )
            )

        expected_index += 1

    return findings


def chapter_files(root: Path) -> list[Path]:
    return sorted(root.glob("docs/part-*/*.md"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--file",
        action="append",
        type=Path,
        help="audit a specific chapter file; can be passed multiple times",
    )
    args = parser.parse_args()

    files = args.file if args.file else chapter_files(args.root)
    findings: list[Finding] = []
    for path in files:
        findings.extend(audit_file(path))

    print(f"audited_files={len(files)}")
    print(f"findings={len(findings)}")
    for finding in findings:
        print(f"{finding.file}:{finding.line}: {finding.message}")

    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
