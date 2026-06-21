#!/usr/bin/env python3
"""Audit generated diagram assets and Markdown references."""

from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
MANIFEST = DOCS / "assets" / "diagrams" / "manifest.json"
MERMAID_RE = re.compile(r"```mermaid\b")
IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
DIAGRAM_REF_RE = re.compile(r"(?:\.\./)*assets/diagrams/[^)#?]+\.svg$")


@dataclass
class Finding:
    path: Path
    line: int
    message: str


def iter_markdown_files() -> list[Path]:
    return [
        path
        for path in sorted(DOCS.rglob("*.md"))
        if path.name != "codex-handoff.md"
    ]


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def audit_no_mermaid() -> list[Finding]:
    findings: list[Finding] = []
    for path in iter_markdown_files():
        text = path.read_text(encoding="utf-8")
        for match in MERMAID_RE.finditer(text):
            findings.append(Finding(path, line_number(text, match.start()), "inline Mermaid block remains; use generated SVG asset"))
    return findings


def audit_image_references() -> list[Finding]:
    findings: list[Finding] = []
    for path in iter_markdown_files():
        text = path.read_text(encoding="utf-8")
        for match in IMAGE_RE.finditer(text):
            target = match.group(1).split("#", 1)[0]
            if target.startswith(("http://", "https://", "/")):
                continue
            resolved = (path.parent / target).resolve()
            try:
                resolved.relative_to(DOCS.resolve())
            except ValueError:
                continue
            if not resolved.exists():
                findings.append(Finding(path, line_number(text, match.start()), f"image target missing: {target}"))
    return findings


def diagram_reference(path: Path, target: str) -> str | None:
    clean_target = target.split("#", 1)[0].split("?", 1)[0]
    if clean_target.startswith(("http://", "https://")):
        return None
    if clean_target.startswith("/"):
        resolved = (DOCS / clean_target.lstrip("/")).resolve()
    else:
        resolved = (path.parent / clean_target).resolve()
    try:
        rel = resolved.relative_to(DOCS.resolve()).as_posix()
    except ValueError:
        return None
    if DIAGRAM_REF_RE.fullmatch(rel):
        return rel
    return None


def collect_markdown_diagram_refs() -> dict[str, tuple[Path, int]]:
    refs: dict[str, tuple[Path, int]] = {}
    for path in iter_markdown_files():
        text = path.read_text(encoding="utf-8")
        for match in IMAGE_RE.finditer(text):
            ref = diagram_reference(path, match.group(1))
            if ref:
                refs[ref] = (path, line_number(text, match.start()))
    return refs


def audit_manifest() -> list[Finding]:
    findings: list[Finding] = []
    if not MANIFEST.exists():
        return [Finding(MANIFEST, 0, "diagram manifest is missing")]
    try:
        records = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [Finding(MANIFEST, exc.lineno, f"manifest JSON invalid: {exc}")]
    if not isinstance(records, list):
        return [Finding(MANIFEST, 0, "manifest must be a list")]
    if len(records) == 0:
        findings.append(Finding(MANIFEST, 0, "manifest is empty"))

    seen_svg: set[str] = set()
    for idx, record in enumerate(records, start=1):
        for key in ("source", "svg", "png", "source_mmd", "title", "hash"):
            if key not in record:
                findings.append(Finding(MANIFEST, idx, f"manifest record missing `{key}`"))
        for key in ("source", "svg", "png", "source_mmd"):
            if key not in record:
                continue
            target = DOCS / str(record[key])
            if not target.exists():
                findings.append(Finding(MANIFEST, idx, f"{key} target missing: {record[key]}"))
        svg_rel = str(record.get("svg", ""))
        if svg_rel in seen_svg:
            findings.append(Finding(MANIFEST, idx, f"duplicate svg record: {svg_rel}"))
        seen_svg.add(svg_rel)
        svg_path = DOCS / svg_rel
        if svg_path.exists():
            try:
                ET.parse(svg_path)
            except ET.ParseError as exc:
                findings.append(Finding(svg_path, exc.position[0], f"SVG XML invalid: {exc}"))

    markdown_refs = collect_markdown_diagram_refs()
    manifest_refs = {
        str(record.get("svg", ""))
        for record in records
        if isinstance(record, dict) and record.get("svg")
    }
    for ref, (path, line) in sorted(markdown_refs.items()):
        if ref not in manifest_refs:
            findings.append(Finding(path, line, f"diagram reference missing from manifest: {ref}"))
    for ref in sorted(manifest_refs - set(markdown_refs)):
        findings.append(Finding(MANIFEST, 0, f"manifest diagram is not referenced by Markdown: {ref}"))
    return findings


def main() -> int:
    findings = audit_no_mermaid()
    findings.extend(audit_image_references())
    findings.extend(audit_manifest())
    for finding in findings:
        print(f"{finding.path}:{finding.line}: {finding.message}")
    print(f"diagram_findings={len(findings)}")
    if MANIFEST.exists():
        try:
            print(f"manifest_records={len(json.loads(MANIFEST.read_text(encoding='utf-8')))}")
        except json.JSONDecodeError:
            print("manifest_records=invalid")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
