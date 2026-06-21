#!/usr/bin/env python3
"""Render all book diagrams as static SVG/PNG assets.

On the first migration pass this script can convert inline Mermaid blocks from
Markdown. After migration, the stable source of truth is the saved `.mmd` files
under `docs/assets/diagrams/sources/`; public Markdown references generated SVG
assets only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ASSET_DIR = DOCS / "assets" / "diagrams"
SOURCE_DIR = ASSET_DIR / "sources"
MANIFEST = ASSET_DIR / "manifest.json"
SKILL_RENDERER = (
    Path.home()
    / ".agents"
    / "skills"
    / "fireworks-tech-graph"
    / "scripts"
    / "generate-from-template.py"
)
MERMAID_RE = re.compile(r"```mermaid\n(.*?)\n```", re.S)
NODE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_.-]*|\[\*\])\s*(.*)$", re.S)
PARTICIPANT_RE = re.compile(r"^participant\s+(\w+)(?:\s+as\s+(.+))?$")


@dataclass
class ParsedNode:
    node_id: str
    label: str
    raw_shape: str = "rect"
    container: str | None = None


@dataclass
class ParsedEdge:
    source: str
    target: str
    label: str = ""
    dashed: bool = False


@dataclass
class ParsedDiagram:
    kind: str
    direction: str
    nodes: dict[str, ParsedNode] = field(default_factory=dict)
    edges: list[ParsedEdge] = field(default_factory=list)
    containers: dict[str, str] = field(default_factory=dict)
    node_order: list[str] = field(default_factory=list)

    def add_node(self, node_id: str, label: str | None = None, raw_shape: str = "rect", container: str | None = None) -> str:
        normalized = normalize_node_id(node_id)
        if normalized not in self.nodes:
            self.nodes[normalized] = ParsedNode(normalized, label or normalized, raw_shape, container)
            self.node_order.append(normalized)
        else:
            node = self.nodes[normalized]
            if label and node.label == node.node_id:
                node.label = label
            if raw_shape != "rect":
                node.raw_shape = raw_shape
            if container and not node.container:
                node.container = container
        return normalized


def iter_markdown_files() -> list[Path]:
    return [
        path
        for path in sorted(DOCS.rglob("*.md"))
        if path.name != "codex-handoff.md"
    ]


def normalize_node_id(raw: str) -> str:
    raw = raw.strip()
    if raw == "[*]":
        return "__state_marker__"
    raw = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)
    if not raw:
        raw = "node"
    if raw[0].isdigit():
        raw = f"n_{raw}"
    return raw


def clean_label(raw: str) -> str:
    label = raw.strip()
    if (label.startswith('"') and label.endswith('"')) or (label.startswith("'") and label.endswith("'")):
        label = label[1:-1]
    label = label.replace("\\n", "\n")
    label = re.sub(r"\s*\n\s*", "\n", label)
    label = re.sub(r"[ \t]+", " ", label)
    return label.strip()


def split_long_token(text: str, limit: int = 24) -> tuple[str, str]:
    if len(text) <= limit:
        return text, ""
    separators = ["_", " / ", "/", " ", "-"]
    for separator in separators:
        parts = text.split(separator)
        if len(parts) <= 1:
            continue
        left: list[str] = []
        right: list[str] = []
        cursor = 0
        for idx, part in enumerate(parts):
            trial = separator.join(left + [part]) if left else part
            if len(trial) <= limit or not left:
                left.append(part)
                cursor = idx + 1
            else:
                break
        right = parts[cursor:]
        if left and right:
            return separator.join(left), separator.join(right)
    return text[:limit].rstrip(), text[limit:].lstrip()


def split_label(label: str) -> tuple[str, str]:
    parts = [part.strip() for part in re.split(r"\n+", label) if part.strip()]
    if not parts:
        return "", ""
    title, extra = split_long_token(parts[0], 26)
    subtitle_parts = []
    if extra:
        subtitle_parts.append(extra)
    subtitle_parts.extend(parts[1:])
    subtitle = " / ".join(subtitle_parts)
    if len(subtitle) > 58:
        subtitle = subtitle[:55].rstrip() + "..."
    return title, subtitle


def text_width(label: str, subtitle: str = "") -> int:
    longest = max([len(x) for x in [label, subtitle] if x] or [8])
    return max(164, min(380, longest * 7 + 54))


def statement_lines(code: str) -> list[str]:
    statements: list[str] = []
    current: list[str] = []
    quote: str | None = None
    depth = 0
    for raw_line in code.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("%%"):
            continue
        current.append(stripped)
        escaped = False
        for ch in stripped:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if quote:
                if ch == quote:
                    quote = None
                continue
            if ch in {"'", '"'}:
                quote = ch
            elif ch in "[{(":
                depth += 1
            elif ch in "]})":
                depth = max(0, depth - 1)
        if not quote and depth == 0:
            statements.append("\n".join(current))
            current = []
    if current:
        statements.append("\n".join(current))
    return statements


def parse_node_expr(expr: str, diagram: ParsedDiagram, container: str | None = None, state_role: str | None = None) -> str:
    expr = expr.strip().rstrip(";")
    if not expr:
        return ""
    if expr == "[*]":
        node_id = "__start__" if state_role == "source" else "__end__"
        label = "Start" if state_role == "source" else "End"
        return diagram.add_node(node_id, label, "state", container)

    match = NODE_RE.match(expr)
    if not match:
        return diagram.add_node(expr, expr, "rect", container)

    raw_id, rest = match.groups()
    rest = rest.strip()
    label = raw_id
    shape = "rect"
    if rest:
        opener = rest[0]
        closer = {"[": "]", "{": "}", "(": ")"}.get(opener)
        if closer:
            shape = "decision" if opener == "{" else "rect"
            body = extract_balanced(rest, opener, closer)
            if body:
                label = clean_label(body)
    return diagram.add_node(raw_id, label, shape, container)


def extract_balanced(text: str, opener: str, closer: str) -> str:
    if not text.startswith(opener):
        return ""
    quote: str | None = None
    depth = 0
    escaped = False
    start = 1
    for idx, ch in enumerate(text):
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in {"'", '"'}:
            quote = ch
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start:idx]
    return text[start:].strip()


def split_edge_chain(statement: str) -> tuple[list[str], list[str]]:
    parts: list[str] = []
    labels: list[str] = []
    quote: str | None = None
    depth = 0
    escaped = False
    cursor = 0
    idx = 0
    while idx < len(statement):
        ch = statement[idx]
        if escaped:
            escaped = False
            idx += 1
            continue
        if ch == "\\":
            escaped = True
            idx += 1
            continue
        if quote:
            if ch == quote:
                quote = None
            idx += 1
            continue
        if ch in {"'", '"'}:
            quote = ch
            idx += 1
            continue
        if ch in "[{(":
            depth += 1
            idx += 1
            continue
        if ch in "]})":
            depth = max(0, depth - 1)
            idx += 1
            continue
        if depth == 0 and statement.startswith("-->", idx):
            parts.append(statement[cursor:idx].strip())
            idx += 3
            label = ""
            if idx < len(statement) and statement[idx] == "|":
                end = statement.find("|", idx + 1)
                if end != -1:
                    label = clean_label(statement[idx + 1 : end])
                    idx = end + 1
            labels.append(label)
            cursor = idx
            continue
        idx += 1
    parts.append(statement[cursor:].strip())
    return parts, labels


def parse_flowchart(code: str) -> ParsedDiagram:
    first = code.strip().splitlines()[0].strip()
    direction = first.split()[1] if len(first.split()) > 1 else "TB"
    diagram = ParsedDiagram("flowchart", direction)
    container_stack: list[str] = []

    for statement in statement_lines("\n".join(code.splitlines()[1:])):
        stripped = statement.strip()
        if not stripped or stripped.startswith(("classDef", "class ", "style ")):
            continue
        if stripped.startswith("subgraph "):
            ident_label = stripped[len("subgraph ") :].strip()
            container_id = ident_label.split("[", 1)[0].strip() or f"container_{len(diagram.containers) + 1}"
            label = container_id
            if "[" in ident_label:
                label = clean_label(extract_balanced(ident_label[ident_label.find("[") :], "[", "]"))
            normalized = normalize_node_id(container_id)
            diagram.containers[normalized] = label
            container_stack.append(normalized)
            continue
        if stripped == "end":
            if container_stack:
                container_stack.pop()
            continue
        if "-->" not in stripped:
            parse_node_expr(stripped, diagram, container_stack[-1] if container_stack else None)
            continue

        parts, labels = split_edge_chain(stripped)
        if len(parts) < 2:
            continue
        current_container = container_stack[-1] if container_stack else None
        source = parse_node_expr(parts[0], diagram, current_container)
        for offset, target_expr in enumerate(parts[1:]):
            target = parse_node_expr(target_expr, diagram, current_container)
            if source and target:
                diagram.edges.append(ParsedEdge(source, target, labels[offset] if offset < len(labels) else ""))
            source = target
    return diagram


def parse_state_diagram(code: str) -> ParsedDiagram:
    diagram = ParsedDiagram("state", "TB")
    for statement in statement_lines("\n".join(code.splitlines()[1:])):
        stripped = statement.strip()
        if not stripped or stripped.startswith("state "):
            continue
        if "-->" not in stripped:
            parse_node_expr(stripped, diagram)
            continue
        left_right, _, transition = stripped.partition(":")
        parts, labels = split_edge_chain(left_right)
        if len(parts) < 2:
            continue
        source = parse_node_expr(parts[0], diagram, state_role="source")
        target = parse_node_expr(parts[1], diagram, state_role="target")
        label = clean_label(transition) or (labels[0] if labels else "")
        diagram.edges.append(ParsedEdge(source, target, label))
    return diagram


def parse_sequence_diagram(code: str) -> ParsedDiagram:
    diagram = ParsedDiagram("sequence", "LR")
    participants: dict[str, str] = {}
    loop_prefix = ""
    for raw in code.splitlines()[1:]:
        stripped = raw.strip()
        if not stripped:
            continue
        participant = PARTICIPANT_RE.match(stripped)
        if participant:
            alias, label = participant.groups()
            participants[alias] = clean_label(label or alias)
            diagram.add_node(alias, participants[alias], "participant")
            continue
        if stripped.startswith("loop "):
            loop_prefix = f"{clean_label(stripped[5:])}: "
            continue
        if stripped == "end":
            loop_prefix = ""
            continue
        match = re.match(r"(\w+)\s*(-+>>|-->>|->>|-->)\s*(\w+)\s*:\s*(.+)$", stripped)
        if not match:
            continue
        source, op, target, label = match.groups()
        diagram.add_node(source, participants.get(source, source), "participant")
        diagram.add_node(target, participants.get(target, target), "participant")
        diagram.edges.append(ParsedEdge(source, target, f"{loop_prefix}{clean_label(label)}", dashed=op.startswith("--")))
    return diagram


def parse_diagram(code: str) -> ParsedDiagram:
    first = code.strip().splitlines()[0].strip()
    if first.startswith("stateDiagram"):
        return parse_state_diagram(code)
    if first.startswith("sequenceDiagram"):
        return parse_sequence_diagram(code)
    return parse_flowchart(code)


def assign_levels(diagram: ParsedDiagram) -> dict[str, int]:
    levels = {node_id: 0 for node_id in diagram.nodes}
    order = {node_id: index for index, node_id in enumerate(diagram.node_order)}
    max_level = max(1, len(diagram.nodes) - 1)
    for _ in range(len(diagram.nodes)):
        changed = False
        for edge in diagram.edges:
            if edge.source not in diagram.nodes or edge.target not in diagram.nodes or edge.source == edge.target:
                continue
            # Mermaid diagrams in this book are authored in reading order. Back
            # edges express retry/feedback loops, so they should not collapse a
            # downstream chain into one giant cyclic level.
            if order.get(edge.target, 0) <= order.get(edge.source, 0):
                continue
            candidate = min(max_level, levels[edge.source] + 1)
            if candidate > levels[edge.target]:
                levels[edge.target] = candidate
                changed = True
        if not changed:
            break
    return levels


def node_style(node: ParsedNode, index: int) -> dict[str, object]:
    title, subtitle = split_label(node.label)
    palette = [
        ("#eff6ff", "#bfdbfe"),
        ("#f0fdf4", "#bbf7d0"),
        ("#fff7ed", "#fed7aa"),
        ("#faf5ff", "#e9d5ff"),
        ("#f0fdfa", "#99f6e4"),
        ("#f8fafc", "#cbd5e1"),
    ]
    fill, stroke = palette[index % len(palette)]
    if node.raw_shape == "decision":
        fill, stroke = "#fff7ed", "#fdba74"
    if node.raw_shape == "state":
        fill, stroke = "#f8fafc", "#94a3b8"
    return {
        "kind": "rect" if node.raw_shape != "decision" else "double_rect",
        "label": title or node.node_id,
        "sublabel": subtitle,
        "fill": fill,
        "stroke": stroke,
        "flat": True,
        "width": text_width(title or node.node_id, subtitle),
        "height": 70 if subtitle else 58,
        "type_label": "DECISION" if node.raw_shape == "decision" else "",
    }


def layout_diagram(diagram: ParsedDiagram, title: str, source_ref: str) -> dict[str, object]:
    levels = assign_levels(diagram)
    grouped: dict[int, list[str]] = defaultdict(list)
    for node_id in diagram.node_order:
        grouped[levels.get(node_id, 0)].append(node_id)
    max_level = max(grouped.keys(), default=0)
    max_rows = max((len(items) for items in grouped.values()), default=1)
    lr = diagram.direction in {"LR", "RL"} or diagram.kind == "sequence"

    node_specs: dict[str, dict[str, object]] = {
        node_id: node_style(node, idx)
        for idx, node_id in enumerate(diagram.node_order)
        for node in [diagram.nodes[node_id]]
    }
    if lr:
        col_widths = {
            level: max(int(node_specs[node_id]["width"]) for node_id in items) + 72
            for level, items in grouped.items()
        }
        width = max(960, 120 + sum(col_widths.get(level, 230) for level in range(max_level + 1)))
        min_height = 360 if max_rows <= 2 else 600
        height = max(min_height, 180 + max_rows * 112 + (70 if diagram.edges and max_rows > 2 else 0))
        x_cursor = 60
        nodes = []
        for level in range(max_level + 1):
            items = grouped.get(level, [])
            col_width = col_widths.get(level, 230)
            y_start = 126 + max(0, (max_rows - len(items)) * 56)
            for row, node_id in enumerate(items):
                spec = dict(node_specs[node_id])
                spec.update({"id": node_id, "x": x_cursor + (col_width - int(spec["width"])) / 2, "y": y_start + row * 112})
                nodes.append(spec)
            x_cursor += col_width
    else:
        row_heights = {
            level: max(int(node_specs[node_id]["height"]) for node_id in items) + 72
            for level, items in grouped.items()
        }
        row_widths = {
            level: sum(int(node_specs[node_id]["width"]) for node_id in items) + max(0, len(items) - 1) * 46
            for level, items in grouped.items()
        }
        width = max(960, 120 + max(row_widths.values(), default=840))
        height = max(600, 128 + sum(row_heights.get(level, 130) for level in range(max_level + 1)) + 90)
        nodes = []
        y_cursor = 118
        for level in range(max_level + 1):
            items = grouped.get(level, [])
            row_height = row_heights.get(level, 130)
            row_width = row_widths.get(level, 0)
            x_cursor = max(60, (width - row_width) / 2)
            for col, node_id in enumerate(items):
                spec = dict(node_specs[node_id])
                spec.update({"id": node_id, "x": x_cursor, "y": y_cursor + (row_height - int(spec["height"])) / 2})
                nodes.append(spec)
                x_cursor += int(spec["width"]) + 46
            y_cursor += row_height

    node_lookup = {node["id"]: node for node in nodes}
    edge_pairs = [
        (edge.source, edge.target)
        for edge in diagram.edges
        if edge.source in node_lookup and edge.target in node_lookup
    ]
    reciprocal_pairs = {
        tuple(sorted((source, target)))
        for source, target in edge_pairs
        if source != target and (target, source) in edge_pairs
    }
    arrows = []
    for edge in diagram.edges:
        if edge.source not in node_lookup or edge.target not in node_lookup:
            continue
        source_node = node_lookup[edge.source]
        target_node = node_lookup[edge.target]
        source_cx = float(source_node["x"]) + float(source_node["width"]) / 2
        target_cx = float(target_node["x"]) + float(target_node["width"]) / 2
        source_cy = float(source_node["y"]) + float(source_node["height"]) / 2
        target_cy = float(target_node["y"]) + float(target_node["height"]) / 2
        same_level = levels.get(edge.source, 0) == levels.get(edge.target, 0)
        reciprocal = tuple(sorted((edge.source, edge.target))) in reciprocal_pairs
        arrow: dict[str, object] = {
            "source": edge.source,
            "target": edge.target,
            "label": edge.label,
            "flow": "control",
        }
        if edge.dashed:
            arrow["dashed"] = True
            arrow["flow"] = "async"
        if lr:
            arrow["source_port"] = "right"
            arrow["target_port"] = "left"
            if same_level:
                if reciprocal:
                    lane_y = min(float(source_node["y"]), float(target_node["y"])) - 34
                    port = "top"
                    if diagram.node_order.index(edge.source) > diagram.node_order.index(edge.target):
                        lane_y = max(
                            float(source_node["y"]) + float(source_node["height"]),
                            float(target_node["y"]) + float(target_node["height"]),
                        ) + 34
                        port = "bottom"
                    arrow["source_port"] = port
                    arrow["target_port"] = port
                    arrow["route_points"] = [[source_cx, lane_y], [target_cx, lane_y]]
                elif source_cx <= target_cx:
                    arrow["source_port"] = "right"
                    arrow["target_port"] = "left"
                else:
                    arrow["source_port"] = "left"
                    arrow["target_port"] = "right"
            elif levels.get(edge.source, 0) > levels.get(edge.target, 0):
                arrow["source_port"] = "bottom"
                arrow["target_port"] = "bottom"
                y_lane = max(node_lookup[edge.source]["y"], node_lookup[edge.target]["y"]) + 92
                arrow["route_points"] = [[source_cx, y_lane], [target_cx, y_lane]]
        else:
            arrow["source_port"] = "bottom"
            arrow["target_port"] = "top"
            if same_level:
                if reciprocal:
                    lane_y = min(float(source_node["y"]), float(target_node["y"])) - 34
                    port = "top"
                    if diagram.node_order.index(edge.source) > diagram.node_order.index(edge.target):
                        lane_y = max(
                            float(source_node["y"]) + float(source_node["height"]),
                            float(target_node["y"]) + float(target_node["height"]),
                        ) + 34
                        port = "bottom"
                    arrow["source_port"] = port
                    arrow["target_port"] = port
                    arrow["route_points"] = [[source_cx, lane_y], [target_cx, lane_y]]
                elif source_cx <= target_cx:
                    arrow["source_port"] = "right"
                    arrow["target_port"] = "left"
                else:
                    arrow["source_port"] = "left"
                    arrow["target_port"] = "right"
            elif levels.get(edge.source, 0) > levels.get(edge.target, 0):
                if source_cx <= target_cx:
                    arrow["source_port"] = "right"
                    arrow["target_port"] = "right"
                else:
                    arrow["source_port"] = "left"
                    arrow["target_port"] = "left"
        arrows.append(arrow)

    containers = []
    for container_id, label in diagram.containers.items():
        members = [node_lookup[nid] for nid, node in diagram.nodes.items() if node.container == container_id and nid in node_lookup]
        if not members:
            continue
        left = min(node["x"] for node in members) - 24
        top = min(node["y"] for node in members) - 48
        right = max(node["x"] + node["width"] for node in members) + 24
        bottom = max(node["y"] + node["height"] for node in members) + 24
        containers.append({
            "x": max(24, left),
            "y": max(92, top),
            "width": right - max(24, left),
            "height": bottom - max(92, top),
            "label": label,
            "fill": "none",
        })

    return {
        "template_type": "architecture",
        "style": 1,
        "width": int(width),
        "height": int(height),
        "title": title,
        "subtitle": source_ref,
        "containers": containers,
        "nodes": nodes,
        "arrows": arrows,
        "legend": [
            {"flow": "control", "label": "主路径 / 状态转换"},
            {"flow": "async", "label": "异步或返回消息"},
        ] if any(edge.dashed for edge in diagram.edges) else [],
        "legend_position": "bottom-left",
        "legend_y": int(height - 38),
        "footer": "Rendered with fireworks-tech-graph",
        "footer_position": "bottom-right",
    }


def slugify(path: Path, index: int) -> str:
    stem = re.sub(r"[^A-Za-z0-9]+", "-", path.with_suffix("").as_posix())
    stem = stem.replace("docs-", "").strip("-").lower()
    return f"{stem}-{index:02d}"


def nearest_heading(text: str, offset: int) -> str:
    prefix = text[:offset]
    headings = re.findall(r"^(#{1,6})\s+(.+)$", prefix, flags=re.M)
    if not headings:
        return "AI Factory Engineering"
    heading = headings[-1][1].strip()
    heading = re.sub(r"\s*\{#.*?\}\s*$", "", heading)
    return heading[:96]


def relative_asset_path(markdown_path: Path, asset_path: Path) -> str:
    return Path(shutil.os.path.relpath(asset_path, start=markdown_path.parent)).as_posix()


def render_svg(data: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(SKILL_RENDERER), "architecture", str(output), json.dumps(data, ensure_ascii=False)]
    subprocess.run(cmd, check=True, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def render_sequence_svg(diagram: ParsedDiagram, title: str, source_ref: str, output: Path) -> None:
    participants = diagram.node_order
    margin_x = 72
    lane_width = 190
    header_y = 126
    header_width = 150
    header_height = 52
    message_gap = 56
    width = max(960, margin_x * 2 + max(0, len(participants) - 1) * lane_width + header_width)
    height = max(420, header_y + 90 + len(diagram.edges) * message_gap + 80)
    xs = {
        node_id: margin_x + header_width / 2 + index * lane_width
        for index, node_id in enumerate(participants)
    }
    lifeline_bottom = height - 58

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {int(width)} {int(height)}" width="{int(width)}" height="{int(height)}">',
        "  <defs>",
        '    <marker id="arrowA" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">',
        '      <polygon points="0 0, 10 3.5, 0 7" fill="#7c3aed"/>',
        "    </marker>",
        "    <style>",
        "      text { font-family: 'Helvetica Neue', Helvetica, Arial, 'PingFang SC', 'Microsoft YaHei', sans-serif; }",
        "      .title { font-size: 30px; font-weight: 700; fill: #111827; }",
        "      .subtitle { font-size: 14px; font-weight: 500; fill: #6b7280; }",
        "      .participant { font-size: 16px; font-weight: 700; fill: #111827; }",
        "      .message { font-size: 12px; font-weight: 600; fill: #4b5563; }",
        "      .footnote { font-size: 12px; font-weight: 500; fill: #94a3b8; }",
        "    </style>",
        "  </defs>",
        f'  <rect width="{int(width)}" height="{int(height)}" fill="#ffffff"/>',
        f'  <text x="{width / 2}" y="56" text-anchor="middle" class="title">{escape(title)}</text>',
        f'  <text x="{width / 2}" y="82" text-anchor="middle" class="subtitle">{escape(source_ref)}</text>',
    ]

    palette = [
        ("#eff6ff", "#bfdbfe"),
        ("#f0fdf4", "#bbf7d0"),
        ("#fff7ed", "#fed7aa"),
        ("#faf5ff", "#e9d5ff"),
        ("#f0fdfa", "#99f6e4"),
        ("#f8fafc", "#cbd5e1"),
    ]
    for index, node_id in enumerate(participants):
        node = diagram.nodes[node_id]
        fill, stroke = palette[index % len(palette)]
        x = xs[node_id] - header_width / 2
        lines.extend(
            [
                f'  <rect x="{x:.1f}" y="{header_y}" width="{header_width}" height="{header_height}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>',
                f'  <text x="{xs[node_id]:.1f}" y="{header_y + 32}" text-anchor="middle" class="participant">{escape(node.label)}</text>',
                f'  <line x1="{xs[node_id]:.1f}" y1="{header_y + header_height}" x2="{xs[node_id]:.1f}" y2="{lifeline_bottom}" stroke="#cbd5e1" stroke-width="1.5" stroke-dasharray="6 5"/>',
            ]
        )

    for index, edge in enumerate(diagram.edges):
        y = header_y + header_height + 42 + index * message_gap
        x1 = xs[edge.source]
        x2 = xs[edge.target]
        direction = 1 if x2 >= x1 else -1
        start = x1 + direction * 16
        end = x2 - direction * 16
        dash = ' stroke-dasharray="7 5"' if edge.dashed else ""
        lines.append(
            f'  <line x1="{start:.1f}" y1="{y}" x2="{end:.1f}" y2="{y}" stroke="#7c3aed" stroke-width="2.2" marker-end="url(#arrowA)"{dash}/>'
        )
        label = edge.label
        if label:
            label_x = (x1 + x2) / 2
            label_width = min(max(len(label) * 7 + 22, 90), abs(x2 - x1) + 110)
            lines.extend(
                [
                    f'  <rect x="{label_x - label_width / 2:.1f}" y="{y - 30}" width="{label_width:.1f}" height="22" rx="6" fill="#ffffff" opacity="0.96"/>',
                    f'  <text x="{label_x:.1f}" y="{y - 15}" text-anchor="middle" class="message">{escape(label)}</text>',
                ]
            )

    lines.extend(
        [
            f'  <text x="{width - 28}" y="{height - 26}" text-anchor="end" class="footnote">Rendered with fireworks-tech-graph</text>',
            "</svg>",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_png(svg: Path, png: Path) -> None:
    subprocess.run(["rsvg-convert", str(svg), "-o", str(png)], check=True, cwd=ROOT)


def collect_from_markdown() -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for markdown_path in iter_markdown_files():
        text = markdown_path.read_text(encoding="utf-8")
        for index, match in enumerate(MERMAID_RE.finditer(text), start=1):
            code = match.group(1).strip("\n")
            rel_doc = markdown_path.relative_to(DOCS).as_posix()
            name = slugify(markdown_path.relative_to(DOCS), index)
            items.append(
                {
                    "name": name,
                    "source": rel_doc,
                    "line": text[: match.start()].count("\n") + 1,
                    "index": index,
                    "title": nearest_heading(text, match.start()),
                    "code": code,
                    "range": (match.start(), match.end()),
                    "markdown_path": markdown_path,
                }
            )
    return items


def collect_from_sources() -> list[dict[str, object]]:
    if not MANIFEST.exists():
        return []
    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    items: list[dict[str, object]] = []
    for record in manifest:
        name = Path(str(record["svg"])).stem
        source_file = SOURCE_DIR / f"{name}.mmd"
        if not source_file.exists():
            continue
        items.append(
            {
                "name": name,
                "source": record["source"],
                "line": record["line"],
                "index": record["index"],
                "title": record["title"],
                "code": source_file.read_text(encoding="utf-8").strip("\n"),
                "range": None,
                "markdown_path": DOCS / str(record["source"]),
            }
        )
    return items


def render_all(write: bool) -> list[dict[str, object]]:
    if not SKILL_RENDERER.exists():
        raise SystemExit(f"fireworks-tech-graph renderer not found: {SKILL_RENDERER}")
    if shutil.which("rsvg-convert") is None:
        raise SystemExit("rsvg-convert is required to export PNG diagrams")

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    source_items = collect_from_markdown()
    replacing_markdown = bool(source_items)
    if not source_items:
        source_items = collect_from_sources()

    replacements_by_path: dict[Path, list[tuple[tuple[int, int], str]]] = defaultdict(list)
    for item in source_items:
        code = str(item["code"])
        parsed = parse_diagram(code)
        markdown_path = item["markdown_path"]
        title = str(item["title"])
        rel_doc = str(item["source"])
        source_ref = f"{rel_doc}:{item['line']}"
        name = str(item["name"])
        svg = ASSET_DIR / f"{name}.svg"
        png = ASSET_DIR / f"{name}.png"
        data = layout_diagram(parsed, title, source_ref)
        source_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]
        manifest.append(
            {
                "source": rel_doc,
                "line": item["line"],
                "index": item["index"],
                "kind": parsed.kind,
                "direction": parsed.direction,
                "nodes": len(parsed.nodes),
                "edges": len(parsed.edges),
                "svg": svg.relative_to(DOCS).as_posix(),
                "png": png.relative_to(DOCS).as_posix(),
                "source_mmd": (SOURCE_DIR / f"{name}.mmd").relative_to(DOCS).as_posix(),
                "hash": source_hash,
                "title": title,
            }
        )
        if write:
            (SOURCE_DIR / f"{name}.mmd").write_text(code + "\n", encoding="utf-8")
            if parsed.kind == "sequence":
                render_sequence_svg(parsed, title, source_ref, svg)
            else:
                render_svg(data, svg)
            export_png(svg, png)
        if write and replacing_markdown and item["range"]:
            alt = f"图：{title}"
            link = relative_asset_path(markdown_path, svg)
            replacements_by_path[markdown_path].append((item["range"], f"![{alt}]({link})"))

    if write and replacements_by_path:
        for markdown_path, replacements in replacements_by_path.items():
            text = markdown_path.read_text(encoding="utf-8")
            new_text_parts: list[str] = []
            cursor = 0
            for (start, end), replacement in replacements:
                new_text_parts.append(text[cursor:start])
                new_text_parts.append(replacement)
                cursor = end
            new_text_parts.append(text[cursor:])
            markdown_path.write_text("".join(new_text_parts), encoding="utf-8")

    if write:
        MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write diagrams and replace Mermaid blocks")
    args = parser.parse_args()
    manifest = render_all(args.write)
    print(f"diagrams={len(manifest)}")
    by_kind: dict[str, int] = defaultdict(int)
    for item in manifest:
        by_kind[str(item["kind"])] += 1
    for kind, count in sorted(by_kind.items()):
        print(f"{kind}={count}")


if __name__ == "__main__":
    main()
