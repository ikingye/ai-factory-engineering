#!/usr/bin/env python3
"""Audit structural depth signals for AI Factory Engineering chapters."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


REQUIRED_HEADINGS = {
    "一个真实场景",
    "本章上下文",
    "读者测试",
    "核心概念",
    "系统架构",
    "工程实现",
    "常见故障",
    "性能指标",
    "设计取舍",
    "小结",
    "延伸阅读",
}

H2_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
HEADING_NUMBER_RE = re.compile(r"^(?:\d+\.\d+|\d+\.)\s+")


@dataclass
class Finding:
    file: Path
    message: str


def audit_file(path: Path) -> list[Finding]:
    text = path.read_text(encoding="utf-8")
    findings: list[Finding] = []
    headings = {
        HEADING_NUMBER_RE.sub("", match.group(1).strip())
        for match in H2_RE.finditer(text)
    }

    for heading in REQUIRED_HEADINGS:
        if heading not in headings:
            findings.append(Finding(path, f"missing heading: {heading}"))

    if "```mermaid" not in text:
        findings.append(Finding(path, "missing Mermaid diagram"))

    if "```yaml" not in text and "```bash" not in text and "```python" not in text:
        findings.append(Finding(path, "missing executable/config code block"))

    if "TODO" in text or "TBD" in text or "待补充" in text:
        findings.append(Finding(path, "contains placeholder marker"))

    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--file", action="append", type=Path)
    args = parser.parse_args()

    files = args.file if args.file else sorted(args.root.glob("docs/part-*/*.md"))
    findings: list[Finding] = []
    for path in files:
        findings.extend(audit_file(path))

    print(f"audited_files={len(files)}")
    print(f"findings={len(findings)}")
    for finding in findings[: args.limit]:
        print(f"{finding.file}: {finding.message}")

    if len(findings) > args.limit:
        print(f"... {len(findings) - args.limit} more")

    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
