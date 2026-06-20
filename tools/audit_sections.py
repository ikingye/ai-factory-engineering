#!/usr/bin/env python3
"""Audit AI Factory Engineering chapter section lengths.

The book is written in Chinese with embedded English technical terms.  For this
project, "word count" is treated as a Chinese-friendly content length:

- each CJK character counts as one unit;
- each contiguous ASCII word/number counts as one unit;
- code fences, Mermaid blocks, chapter orientation sections, summaries and
  reading placeholders are excluded from the hard 500-800 section target.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


EXCLUDED_HEADINGS = {
    "本章回答的问题",
    "本章上下文",
    "读者测试",
    "小结",
    "延伸阅读",
}

DEFAULT_INCLUDED_HEADINGS = {
    "一个真实场景",
    "核心概念",
    "系统架构",
    "工程实现",
    "常见故障",
    "性能指标",
    "设计取舍",
}

NUMBERED_HEADING_RE = re.compile(r"^\d+(?:\.\d+)+\s+")
HEADING_NUMBER_RE = re.compile(r"^\d+\.\d+\s+")
H2_RE = re.compile(r"^##\s+(.+?)\s*$")
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
ASCII_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./:+-]*")


@dataclass
class Section:
    file: Path
    heading: str
    count: int


def strip_code_fences(markdown: str) -> str:
    lines: list[str] = []
    in_fence = False
    for line in markdown.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            lines.append(line)
    return "\n".join(lines)


def content_units(text: str) -> int:
    text = strip_code_fences(text)
    # Drop Markdown punctuation that should not inflate content length.
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    cjk = len(CJK_RE.findall(text))
    without_cjk = CJK_RE.sub(" ", text)
    ascii_words = len(ASCII_WORD_RE.findall(without_cjk))
    return cjk + ascii_words


def should_audit_heading(heading: str) -> bool:
    clean = heading.strip()
    normalized = HEADING_NUMBER_RE.sub("", clean)
    if normalized in EXCLUDED_HEADINGS:
        return False
    if normalized in DEFAULT_INCLUDED_HEADINGS:
        return True
    return bool(NUMBERED_HEADING_RE.match(clean))


def iter_sections(path: Path) -> list[Section]:
    sections: list[Section] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        match = H2_RE.match(line)
        if match:
            if current_heading and should_audit_heading(current_heading):
                sections.append(
                    Section(path, current_heading, content_units("\n".join(current_lines)))
                )
            current_heading = match.group(1)
            current_lines = []
        elif current_heading is not None:
            current_lines.append(line)

    if current_heading and should_audit_heading(current_heading):
        sections.append(Section(path, current_heading, content_units("\n".join(current_lines))))
    return sections


def chapter_files(root: Path) -> list[Path]:
    return sorted(root.glob("docs/part-*/*.md"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min", type=int, default=500, dest="min_count")
    parser.add_argument("--max", type=int, default=800, dest="max_count")
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--all", action="store_true", help="print all audited sections")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--file",
        action="append",
        type=Path,
        help="audit a specific Markdown file; can be passed multiple times",
    )
    args = parser.parse_args()

    sections: list[Section] = []
    files = args.file if args.file else chapter_files(args.root)
    for path in files:
        sections.extend(iter_sections(path))

    low = [s for s in sections if s.count < args.min_count]
    high = [s for s in sections if s.count > args.max_count]

    print(f"audited_sections={len(sections)}")
    print(f"below_min={len(low)} min={args.min_count}")
    print(f"above_max={len(high)} max={args.max_count}")

    if args.all:
        for section in sections:
            print(f"{section.count:4d} {section.file}:{section.heading}")
        return 0

    if low:
        print("\nLOW:")
        for section in sorted(low, key=lambda s: (s.count, str(s.file), s.heading))[: args.limit]:
            print(f"{section.count:4d} {section.file}:{section.heading}")

    if high:
        print("\nHIGH:")
        for section in sorted(high, key=lambda s: (-s.count, str(s.file), s.heading))[: args.limit]:
            print(f"{section.count:4d} {section.file}:{section.heading}")

    return 1 if low or high else 0


if __name__ == "__main__":
    sys.exit(main())
