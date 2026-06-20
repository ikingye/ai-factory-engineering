#!/usr/bin/env python3
"""Audit heading numbering policy for all published docs.

Chapter pages use a book-style hierarchy:

- H2 headings are the five stable chapter groups, for example "21.1 导读".
- H3 headings are the concrete sections, for example
  "21.3.10 NVIDIA GPU Container 原理".

Non-chapter pages keep page-local H2 numbering, for example
"1. 这本书解决什么问题". If they use H3 headings, H3 headings are numbered
under the current H2, for example "1.1 AI Factory".
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


CHAPTER_RE = re.compile(r"chapter-(\d+)-")
H2_RE = re.compile(r"^##\s+(.+?)\s*$")
H3_RE = re.compile(r"^###\s+(.+?)\s*$")
CHAPTER_H2_RE = re.compile(r"^(\d+)\.(\d+)\s+(.+)")
CHAPTER_H3_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)\s+.+")
PAGE_H2_RE = re.compile(r"^(\d+)\.\s+.+")
PAGE_H3_RE = re.compile(r"^(\d+)\.(\d+)\s+.+")
EXCLUDED_FILES = {"codex-handoff.md"}
CHAPTER_GROUPS = [
    "导读",
    "基础模型",
    "关键技术",
    "工程落地",
    "小结与延伸阅读",
]


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
    expected_h2_index = 1
    expected_h3_index = 1
    current_h2_index: int | None = None
    in_fence = False

    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        h2_match = H2_RE.match(line)
        h3_match = H3_RE.match(line)
        if not h2_match and not h3_match:
            continue

        heading = (h2_match or h3_match).group(1).strip()
        if chapter is None:
            if h3_match:
                numbered = PAGE_H3_RE.match(heading)
                if not numbered:
                    findings.append(Finding(path, line_number, f"unnumbered H3: {heading}"))
                    continue

                if current_h2_index is None:
                    findings.append(Finding(path, line_number, f"H3 before valid H2: {heading}"))
                elif int(numbered.group(1)) != current_h2_index:
                    findings.append(
                        Finding(
                            path,
                            line_number,
                            f"H3 does not belong to current H2: {heading}; expected {current_h2_index}.x",
                        )
                    )

                actual_h3_index = int(numbered.group(2))
                expected_label = f"{current_h2_index}.{expected_h3_index}"
                if actual_h3_index != expected_h3_index:
                    findings.append(
                        Finding(
                            path,
                            line_number,
                            f"non-consecutive H3: {heading}; expected {expected_label}",
                        )
                    )
                expected_h3_index += 1
                continue

            numbered = PAGE_H2_RE.match(heading)
            if not numbered:
                findings.append(Finding(path, line_number, f"unnumbered H2: {heading}"))
                continue

            actual_index = int(numbered.group(1))
            expected_label = f"{expected_h2_index}."
            if actual_index != expected_h2_index:
                findings.append(
                    Finding(
                        path,
                        line_number,
                        f"non-consecutive H2: {heading}; expected {expected_label}",
                    )
                )
            current_h2_index = actual_index
            expected_h2_index += 1
            expected_h3_index = 1
            continue

        if h2_match:
            numbered = CHAPTER_H2_RE.match(heading)
            if not numbered:
                findings.append(Finding(path, line_number, f"unnumbered chapter H2: {heading}"))
                current_h2_index = None
                expected_h3_index = 1
                continue

            if int(numbered.group(1)) != chapter:
                findings.append(
                    Finding(
                        path,
                        line_number,
                        f"wrong chapter prefix: {heading}; expected {chapter}.x",
                    )
                )

            actual_index = int(numbered.group(2))
            expected_label = f"{chapter}.{expected_h2_index}"
            if actual_index != expected_h2_index:
                findings.append(
                    Finding(
                        path,
                        line_number,
                        f"non-consecutive chapter H2: {heading}; expected {expected_label}",
                    )
                )

            expected_title = (
                CHAPTER_GROUPS[expected_h2_index - 1]
                if 1 <= expected_h2_index <= len(CHAPTER_GROUPS)
                else None
            )
            actual_title = numbered.group(3).strip()
            if expected_title and actual_title != expected_title:
                findings.append(
                    Finding(
                        path,
                        line_number,
                        f"wrong chapter H2 group title: {heading}; expected {expected_label} {expected_title}",
                    )
                )

            current_h2_index = actual_index
            expected_h2_index += 1
            expected_h3_index = 1
        else:
            numbered = CHAPTER_H3_RE.match(heading)
            if not numbered:
                findings.append(Finding(path, line_number, f"unnumbered chapter H3: {heading}"))
                continue

            if int(numbered.group(1)) != chapter:
                findings.append(
                    Finding(
                        path,
                        line_number,
                        f"wrong chapter prefix: {heading}; expected {chapter}.x.y",
                    )
                )

            actual_h2_index = int(numbered.group(2))
            if current_h2_index is None:
                findings.append(Finding(path, line_number, f"chapter H3 before valid H2: {heading}"))
            elif actual_h2_index != current_h2_index:
                findings.append(
                    Finding(
                        path,
                        line_number,
                        f"chapter H3 does not belong to current H2: {heading}; expected {chapter}.{current_h2_index}.x",
                    )
                )

            actual_h3_index = int(numbered.group(3))
            expected_label = f"{chapter}.{current_h2_index}.{expected_h3_index}"
            if actual_h3_index != expected_h3_index:
                findings.append(
                    Finding(
                        path,
                        line_number,
                        f"non-consecutive chapter H3: {heading}; expected {expected_label}",
                    )
                )
            expected_h3_index += 1

    if chapter is not None and expected_h2_index - 1 != len(CHAPTER_GROUPS):
        findings.append(
            Finding(
                path,
                0,
                f"chapter has {expected_h2_index - 1} H2 groups; expected {len(CHAPTER_GROUPS)}",
            )
        )

    return findings


def markdown_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.glob("docs/**/*.md")
        if path.is_file() and path.name not in EXCLUDED_FILES
    )


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

    files = args.file if args.file else markdown_files(args.root)
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
