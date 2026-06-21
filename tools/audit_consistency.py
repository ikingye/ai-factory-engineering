#!/usr/bin/env python3
"""Audit cross-book consistency for AI Factory Engineering.

This script catches high-risk drift that ordinary Markdown/build checks do not
see: stale terminology, obsolete structure rules, exact long duplicate
paragraphs, and public site URL drift.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


DOC_TARGETS = ("docs", "README.md", "mkdocs.yml")
EXCLUDED_NAMES = {"codex-handoff.md"}
MIN_DUPLICATE_PARAGRAPH_CHARS = 120
MIN_DUPLICATE_SENTENCE_CHARS = 35

FORBIDDEN_PATTERNS = {
    r"AI Factory 的七层模型": "use `AI Factory 的分层模型`; the book uses eight subject layers plus cross-cutting capabilities",
    r"七层模型": "avoid seven-layer wording; use the book-wide layered model wording",
    r"严格说这是八个层次": "remove self-contradictory seven-vs-eight explanation",
    r"用户提供的": "remove conversation-process wording from public docs",
    r"原文链接": "do not expose source clipping metadata in public docs",
    r"mp\.weixin": "do not expose WeChat source URLs in public docs",
    r"超级助理": "do not expose clipping-tool metadata in public docs",
    r"参考某篇文章": "avoid article-reference process wording in public docs",
    r"NVIDIA GPU Container": "use neutral GPU container runtime wording",
    r"Nvidia GPU Container": "use neutral GPU container runtime wording",
    r"GPU Container": "use neutral GPU container runtime wording",
    r"GPU Container 原理": "use neutral GPU container runtime wording",
    r"下一轮需要": "avoid stale process roadmap wording; describe the current update loop or next durable topic",
    r"第一条深度化链路": "avoid stale roadmap wording after multiple subject loops have been deepened",
    r"本书不依赖某个固定数值": "use verified official specs plus local validation wording for hardware claims",
    r"PCIe Gen4 x16 64GB/s(?! bidirectional|\))": "state PCIe aggregate bandwidth direction explicitly",
    r"PCIe Gen5 128GB/s(?! bidirectional|\))": "state PCIe aggregate bandwidth direction explicitly",
    r"H100 SXM \| Hopper；80GB HBM3；3\.35TB/s；FP8 Transformer Engine；NVLink 900GB/s；PCIe Gen5": "do not mix SXM NVLink and PCIe shape specs in one unqualified H100 row",
    r"21\.22 延伸阅读": "old flat chapter numbering; use H2 groups and H3 sections",
    r"所有二级标题按章号连续编号": "old flat H2 rule; use H2 groups plus H3 sections",
    r"所有 H2 都连续编号": "old flat H2 rule; use H2 groups plus H3 sections",
    r"章号\.序号": "old flat chapter-number wording; use 章号.组号.小节号 for chapter H3",
}

REQUIRED_SNIPPETS = {
    "mkdocs.yml": ["site_url: https://kingye.me/ai-factory-engineering/"],
    "docs/glossary.md": [
        "AI Factory 是把模型、数据、算力、平台和运维流程组织起来",
        "Token Factory 是观察 AI Factory 产出的经济性视角",
        "TokenFoundry 更适合作为产业组织或案例名词",
        "Kubernetes 是容器编排系统",
    ],
    "docs/style-guide.md": [
        "不把 AI Factory 等同于 GPU 集群",
        "不把 Token Factory 等同于 AI Factory",
        "Kubernetes、Slurm、容器和 GPU 调度",
        "H2 是稳定大组，H3 是可引用小节",
    ],
    "docs/depth-standard.md": [
        "GPU 虚拟化与资源管理链路覆盖矩阵",
        "NVIDIA GPU 代际、官方规格来源、SKU/形态、HBM、互联、功耗、精度能力和本地验收口径",
        "IB、RoCEv1/v2、iWARP、UEC/UET 类新型传输",
        "`prefix_cache_accounting_record`",
    ],
    "docs/system-map.md": [
        "`gpu_generation_spec_record`",
        "`prefix_cache_accounting_record`",
        "`gpu_slicing_policy`",
        "`rdma_transport_profile`",
        "`numa_topology_evidence`",
    ],
    "docs/part-02-platform/chapter-07-metering-billing.md": [
        "`cache_read_tokens`",
        "`cache_write_tokens`",
        "输入 token 是本次请求完整 prompt 的 token 数，不会因为 cache 命中而扣减",
    ],
    "docs/part-04-runtime/chapter-15-inference-engines.md": [
        "cache-aware 调度",
        "缓存读 token",
        "缓存写 token",
    ],
    "docs/part-07-network-storage/chapter-31-scale-up-networking.md": [
        "PCIe 3.0/4.0/5.0 x16 单向约为 15.75/31.5/63GB/s",
        "NVLink/NVSwitch 代际演进",
        "rack-scale domain",
    ],
    "docs/part-00-introduction/chapter-00-from-datacenter-to-ai-factory.md": [
        "### 0.3.5 AI Factory 的分层模型",
        "这是八个主体层次，外加可靠性、可观测性、准入验收、故障诊断、安全、成本和 Token Factory 经济性等横切能力",
    ],
}


@dataclass
class Finding:
    file: Path
    line: int
    message: str


def iter_files(root: Path, targets: tuple[str, ...]) -> list[Path]:
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


def strip_code_fences(text: str) -> str:
    return re.sub(r"```.*?```", "", text, flags=re.S)


def normalized_paragraphs(path: Path) -> list[tuple[int, str]]:
    text = strip_code_fences(path.read_text(encoding="utf-8"))
    paragraphs: list[tuple[int, str]] = []
    offset = 0
    for raw in re.split(r"\n\s*\n", text):
        line = text.count("\n", 0, offset) + 1
        offset += len(raw) + 2
        lines = [
            item.strip()
            for item in raw.splitlines()
            if item.strip()
            and not item.lstrip().startswith("#")
            and not item.lstrip().startswith("|")
            and not item.lstrip().startswith("- ")
        ]
        paragraph = re.sub(r"\s+", " ", " ".join(lines)).strip()
        if len(paragraph) >= MIN_DUPLICATE_PARAGRAPH_CHARS:
            paragraphs.append((line, paragraph))
    return paragraphs


def audit_forbidden_patterns(files: list[Path]) -> list[Finding]:
    findings: list[Finding] = []
    compiled = [(re.compile(pattern), message) for pattern, message in FORBIDDEN_PATTERNS.items()]
    for path in files:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for pattern, message in compiled:
                if pattern.search(line):
                    findings.append(Finding(path, line_number, message))
    return findings


def audit_required_snippets(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for relative_path, snippets in REQUIRED_SNIPPETS.items():
        path = root / relative_path
        if not path.exists():
            findings.append(Finding(path, 0, "required consistency file is missing"))
            continue
        text = path.read_text(encoding="utf-8")
        for snippet in snippets:
            if snippet not in text:
                findings.append(Finding(path, 0, f"missing required consistency snippet: {snippet}"))
    return findings


def audit_duplicate_paragraphs(files: list[Path]) -> list[Finding]:
    findings: list[Finding] = []
    seen: dict[str, tuple[Path, int]] = {}
    for path in files:
        for line_number, paragraph in normalized_paragraphs(path):
            previous = seen.get(paragraph)
            if previous is not None:
                previous_path, previous_line = previous
                findings.append(
                    Finding(
                        path,
                        line_number,
                        f"duplicate long paragraph also at {previous_path}:{previous_line}",
                    )
                )
            else:
                seen[paragraph] = (path, line_number)
    return findings


def normalized_sentences(path: Path) -> list[tuple[int, str]]:
    text = strip_code_fences(path.read_text(encoding="utf-8"))
    # Drop Markdown tables because repeated matrix row fragments are often
    # structural labels, not prose duplication.
    text = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("|"))
    sentences: list[tuple[int, str]] = []
    for match in re.finditer(r"[^。！？\n]{35,}[。！？]", text):
        sentence = re.sub(r"\s+", " ", match.group(0)).strip()
        if len(sentence) < MIN_DUPLICATE_SENTENCE_CHARS:
            continue
        if sentence.startswith("#") or sentence.startswith("-"):
            continue
        line_number = text.count("\n", 0, match.start()) + 1
        sentences.append((line_number, sentence))
    return sentences


def audit_duplicate_sentences(files: list[Path]) -> list[Finding]:
    findings: list[Finding] = []
    seen: dict[str, tuple[Path, int]] = {}
    for path in files:
        for line_number, sentence in normalized_sentences(path):
            previous = seen.get(sentence)
            if previous is not None:
                previous_path, previous_line = previous
                findings.append(
                    Finding(
                        path,
                        line_number,
                        f"duplicate long sentence also at {previous_path}:{previous_line}",
                    )
                )
            else:
                seen[sentence] = (path, line_number)
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()

    files = iter_files(args.root, DOC_TARGETS)
    findings: list[Finding] = []
    findings.extend(audit_forbidden_patterns(files))
    findings.extend(audit_required_snippets(args.root))
    findings.extend(audit_duplicate_paragraphs(files))
    findings.extend(audit_duplicate_sentences(files))

    print(f"audited_files={len(files)}")
    print(f"findings={len(findings)}")
    for finding in findings:
        location = f"{finding.file}:{finding.line}" if finding.line else str(finding.file)
        print(f"{location}: {finding.message}")

    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
