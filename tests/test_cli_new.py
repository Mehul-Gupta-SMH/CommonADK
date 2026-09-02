"""Tests for `commonadk new` (issue #12 -- CLI agent scaffolding).

Every test here invokes `cli.main(argv)` in-process (no subprocess, no
network), matching test_cli.py's style, and builds on the same `tmp_project`
fixture (a fresh, mutable copy of the shipped research-crew example) from
tests/conftest.py.
"""

from __future__ import annotations

import yaml
import pytest

from commonadk import cli
from commonadk.loader import load


# ---------------------------------------------------------------------------
# happy path (no --from)
# ---------------------------------------------------------------------------


def test_new_scaffolds_a_conforming_agent_folder(tmp_project, capsys):
    rc = cli.main(["new", str(tmp_project), "reviewer"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "reviewer" in captured.out

    agent_dir = tmp_project / "reviewer"
    assert (agent_dir / "skill.md").is_file()
    assert (agent_dir / "tools.py").is_file()
    assert (agent_dir / "agent-config.yaml").is_file()

    # name matches the folder, description is present, tools references a
    # real function, requires.env is empty -- per issue #12's ask.
    config = yaml.safe_load((agent_dir / "agent-config.yaml").read_text())
    assert config["name"] == "reviewer"
    assert config["description"]
    assert config["tools"] == ["example_tool"]
    assert config["requires"] == {"env": []}

    skill_text = (agent_dir / "skill.md").read_text()
    assert skill_text.strip() != ""

    # tools.py must actually define the referenced function, typed and
    # documented -- exactly what loader.py/validation.py require.
    tools_text = (agent_dir / "tools.py").read_text()
    assert "def example_tool(text: str) -> str:" in tools_text
    assert '"""' in tools_text


def test_new_output_passes_validate(tmp_project, capsys):
    rc_new = cli.main(["new", str(tmp_project), "reviewer"])
    assert rc_new == 0
    capsys.readouterr()

    rc_validate = cli.main(["validate", str(tmp_project)])
    captured = capsys.readouterr()

    assert rc_validate == 0
    assert "reviewer" in captured.out


def test_new_without_from_does_not_touch_interactions(tmp_project, capsys):
    interactions_before = (tmp_project / "interactions.yaml").read_text()
    layer_before = (tmp_project / "interaction-layer.md").read_text()

    rc = cli.main(["new", str(tmp_project), "reviewer"])
    capsys.readouterr()

    assert rc == 0
    assert (tmp_project / "interactions.yaml").read_text() == interactions_before
    assert (tmp_project / "interaction-layer.md").read_text() == layer_before


# ---------------------------------------------------------------------------
# refuse to overwrite
# ---------------------------------------------------------------------------


def test_new_refuses_to_overwrite_existing_agent(tmp_project, capsys):
    rc = cli.main(["new", str(tmp_project), "writer"])  # writer already exists
    captured = capsys.readouterr()

    assert rc != 0
    assert "writer" in captured.err
    assert "refusing to overwrite" in captured.err
    # the shipped writer/tools.py must survive untouched.
    assert "count_words" in (tmp_project / "writer" / "tools.py").read_text()


def test_new_refuses_to_overwrite_a_just_scaffolded_agent(tmp_project, capsys):
    rc_first = cli.main(["new", str(tmp_project), "reviewer"])
    assert rc_first == 0
    capsys.readouterr()

    config_before = (tmp_project / "reviewer" / "agent-config.yaml").read_text()

    rc_second = cli.main(["new", str(tmp_project), "reviewer"])
    captured = capsys.readouterr()

    assert rc_second != 0
    assert "reviewer" in captured.err
    assert "refusing to overwrite" in captured.err
    # the first scaffold's files must be untouched by the refused second call.
    assert (tmp_project / "reviewer" / "agent-config.yaml").read_text() == config_before


# ---------------------------------------------------------------------------
# --from / --type edge variant
# ---------------------------------------------------------------------------


def test_new_with_from_appends_edge_and_regenerates_interaction_layer(
    tmp_project, capsys
):
    rc = cli.main(
        ["new", str(tmp_project), "fact_checker", "--from", "writer", "--type", "handoff"]
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert str(tmp_project / "interactions.yaml") in captured.out
    assert str(tmp_project / "interaction-layer.md") in captured.out

    interactions = yaml.safe_load((tmp_project / "interactions.yaml").read_text())
    new_edges = [e for e in interactions["edges"] if e["to"] == "fact_checker"]
    assert new_edges == [{"from": "writer", "to": "fact_checker", "type": "handoff"}]
    # the pre-existing edges must survive untouched.
    assert {"from": "coordinator", "to": "researcher", "type": "delegate"} in interactions["edges"]
    assert {"from": "researcher", "to": "writer", "type": "handoff"} in interactions["edges"]

    layer = (tmp_project / "interaction-layer.md").read_text()
    assert "fact_checker" in layer
    assert "GENERATED FILE" in layer  # never hand-edited -- regenerated via the renderer

    project = load(tmp_project)
    assert any(
        e.from_ == "writer" and e.to == "fact_checker" and e.type == "handoff"
        for e in project.graph.edges
    )


def test_new_with_from_default_edge_type_is_delegate(tmp_project, capsys):
    rc = cli.main(["new", str(tmp_project), "archivist", "--from", "coordinator"])
    capsys.readouterr()

    assert rc == 0
    interactions = yaml.safe_load((tmp_project / "interactions.yaml").read_text())
    new_edges = [e for e in interactions["edges"] if e["to"] == "archivist"]
    assert new_edges == [{"from": "coordinator", "to": "archivist", "type": "delegate"}]


def test_new_with_from_output_passes_validate(tmp_project, capsys):
    rc_new = cli.main(
        ["new", str(tmp_project), "fact_checker", "--from", "writer", "--type", "handoff"]
    )
    assert rc_new == 0
    capsys.readouterr()

    rc_validate = cli.main(["validate", str(tmp_project)])
    captured = capsys.readouterr()

    assert rc_validate == 0
    assert "fact_checker" in captured.out


def test_new_type_without_from_is_rejected(tmp_project, capsys):
    rc = cli.main(["new", str(tmp_project), "reviewer", "--type", "handoff"])
    captured = capsys.readouterr()

    assert rc != 0
    assert "--type requires --from" in captured.err
    # nothing must have been created.
    assert not (tmp_project / "reviewer").exists()


def test_new_unknown_from_agent_is_rejected(tmp_project, capsys):
    rc = cli.main(["new", str(tmp_project), "reviewer", "--from", "nonexistent"])
    captured = capsys.readouterr()

    assert rc != 0
    assert "nonexistent" in captured.err
    assert "writer" in captured.err  # names the known agents
    # nothing must have been created.
    assert not (tmp_project / "reviewer").exists()


# ---------------------------------------------------------------------------
# error paths shared with the other commands
# ---------------------------------------------------------------------------


def test_new_broken_project_exits_1_before_scaffolding(tmp_project, capsys):
    researcher_cfg = tmp_project / "researcher" / "agent-config.yaml"
    data = yaml.safe_load(researcher_cfg.read_text())
    data["tools"] = ["search_wbe"]  # typo of search_web
    researcher_cfg.write_text(yaml.safe_dump(data))

    rc = cli.main(["new", str(tmp_project), "reviewer"])
    captured = capsys.readouterr()

    assert rc == 1
    assert "search_wbe" in captured.err
    assert not (tmp_project / "reviewer").exists()


def test_new_missing_project_folder_exits_1(tmp_path, capsys):
    rc = cli.main(["new", str(tmp_path / "does-not-exist"), "reviewer"])
    captured = capsys.readouterr()

    assert rc == 1
    assert "not found" in captured.err
