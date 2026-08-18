import commonadk
from commonadk.mermaid import render_mermaid, write_interaction_layer
from commonadk.models import InteractionEdge, InteractionGraph


def test_render_mermaid_contains_nodes_and_edges():
    graph = InteractionGraph(
        entry="coordinator",
        edges=[
            InteractionEdge(**{"from": "coordinator", "to": "researcher", "type": "delegate"}),
            InteractionEdge(**{"from": "researcher", "to": "writer", "type": "handoff"}),
        ],
    )
    out = render_mermaid(graph)

    assert out.startswith("flowchart TD")
    assert "coordinator" in out
    assert "researcher" in out
    assert "writer" in out
    assert "entry" in out  # entry node is visually marked


def test_render_mermaid_distinguishes_delegate_and_handoff():
    graph = InteractionGraph(
        entry="coordinator",
        edges=[
            InteractionEdge(**{"from": "coordinator", "to": "researcher", "type": "delegate"}),
            InteractionEdge(**{"from": "researcher", "to": "writer", "type": "handoff"}),
        ],
    )
    out = render_mermaid(graph)

    delegate_line = next(line for line in out.splitlines() if "delegate" in line)
    handoff_line = next(line for line in out.splitlines() if "handoff" in line)

    assert "-->" in delegate_line
    assert "-- delegate -->" in delegate_line
    # handoff uses a dashed arrow, visually distinct from delegate's solid one
    assert ".->" in handoff_line
    assert "-. handoff .->" in handoff_line
    assert delegate_line != handoff_line


def test_write_interaction_layer(tmp_project):
    project = commonadk.load(tmp_project)
    out_path = write_interaction_layer(tmp_project, project.graph)

    assert out_path == tmp_project / "interaction-layer.md"
    content = out_path.read_text()

    assert "GENERATED FILE" in content
    assert "```mermaid" in content
    assert "flowchart TD" in content
    assert "coordinator" in content
    assert "delegate" in content
    assert "handoff" in content


def test_example_interaction_layer_matches_current_graph(example_common_dir):
    """Guards against interaction-layer.md drifting from interactions.yaml."""
    project = commonadk.load(example_common_dir)
    expected_mermaid = render_mermaid(project.graph)

    committed = (example_common_dir / "interaction-layer.md").read_text()
    assert expected_mermaid in committed
