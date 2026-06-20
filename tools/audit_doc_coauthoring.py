#!/usr/bin/env python3
"""Audit doc-coauthoring reader-test structure for every chapter."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


CONTEXT_SECTION = "本章上下文"
READER_SECTION = "读者测试"
CONTEXT_MARKERS = ("层级定位", "前置依赖", "后续关联", "读完能力")
READER_MARKERS = ("机制题", "边界题", "路径题", "排障题")
HEADING_RE = re.compile(r"^(#{2,3})\s+(.+?)\s*$", re.MULTILINE)
HEADING_NUMBER_RE = re.compile(r"^(?:\d+\.\d+\.\d+|\d+\.\d+|\d+\.)\s+")


@dataclass
class Finding:
    file: Path
    message: str


def section_body(text: str, heading: str) -> str | None:
    matches = list(HEADING_RE.finditer(text))
    for index, match in enumerate(matches):
        level = len(match.group(1))
        current = HEADING_NUMBER_RE.sub("", match.group(2).strip())
        if current != heading:
            continue
        start = match.end()
        end = len(text)
        for next_match in matches[index + 1 :]:
            if len(next_match.group(1)) <= level:
                end = next_match.start()
                break
        return text[start:end].strip()
    return None


def audit_file(path: Path) -> list[Finding]:
    text = path.read_text(encoding="utf-8")
    findings: list[Finding] = []

    context = section_body(text, CONTEXT_SECTION)
    if context is None:
        findings.append(Finding(path, f"missing section: {CONTEXT_SECTION}"))
    else:
        for marker in CONTEXT_MARKERS:
            if marker not in context:
                findings.append(Finding(path, f"{CONTEXT_SECTION} missing marker: {marker}"))

    body = section_body(text, READER_SECTION)
    if body is None:
        findings.append(Finding(path, f"missing section: {READER_SECTION}"))
        return findings

    for marker in READER_MARKERS:
        if marker not in body:
            findings.append(Finding(path, f"{READER_SECTION} missing marker: {marker}"))

    bullet_count = len(re.findall(r"(?m)^-\s+", body))
    if bullet_count < 4:
        findings.append(
            Finding(path, f"{READER_SECTION} has too few reader questions: {bullet_count}")
        )

    if "TODO" in body or "TBD" in body or "待补充" in body:
        findings.append(Finding(path, f"{READER_SECTION} contains placeholder marker"))

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
        print(f"{finding.file}: {finding.message}")

    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
