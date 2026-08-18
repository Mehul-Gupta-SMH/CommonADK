"""Renders `interactions.yaml` to a Mermaid flowchart.

This is documentation only: `interactions.yaml` is the source of truth
(plan.md, "Interaction source of truth"), and `interaction-layer.md` is
regenerated from it so the diagram never drifts out of sync with the spec.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Union

from .models import InteractionGraph

_GENERATED_HEADER = (
    "<!-- GENERATED FILE -- do not edit by hand.\n"
    "     Regenerate with `commonadk.mermaid.write_interaction_layer`\n"
    "     (or `commonadk render`, once the CLI lands) from interactions.yaml. -->\n"
)


def _node_id(name: str) -> str:
    """Sanitize an agent name into a valid Mermaid node identifier."""
    return re.sub(r"[^0-9A-Za-z_]", "_", name)


def render_mermaid(graph: InteractionGraph) -> str:
    """Render an InteractionGraph as a `flowchart TD` Mermaid block (no fences)."""
    nodes: dict[str, str] = {}
    if graph.entry:
        nodes[graph.entry] = graph.entry
    for edge in graph.edges:
        nodes.setdefault(edge.from_, edge.from_)
        nodes.setdefault(edge.to, edge.to)

    lines = ["flowchart TD"]
    for name in sorted(nodes):
        node_id = _node_id(name)
        if graph.entry and name == graph.entry:
            lines.append(f'    {node_id}(["{name} (entry)"])')
        else:
            lines.append(f'    {node_id}["{name}"]')

    for edge in graph.edges:
        src, dst = _node_id(edge.from_), _node_id(edge.to)
        if edge.type == "delegate":
            lines.append(f"    {src} -- delegate --> {dst}")
        else:  # handoff
            lines.append(f"    {src} -. handoff .-> {dst}")

    return "\n".join(lines)


def write_interaction_layer(common_dir: Union[str, Path], graph: InteractionGraph) -> Path:
    """Write `common/interaction-layer.md` from `graph`. Returns the file path."""
    common_dir = Path(common_dir)
    out_path = common_dir / "interaction-layer.md"

    mermaid_block = render_mermaid(graph)
    content = (
        f"{_GENERATED_HEADER}\n"
        "# Interaction Layer\n\n"
        "```mermaid\n"
        f"{mermaid_block}\n"
        "```\n"
    )
    out_path.write_text(content)
    return out_path
