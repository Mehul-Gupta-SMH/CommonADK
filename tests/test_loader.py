import warnings

import yaml
import pytest

import commonadk
import commonadk.validation as validation


def test_load_example_project(example_common_dir):
    project = commonadk.load(example_common_dir)

    assert project.config.name == "research-crew"
    assert project.config.entry == "coordinator"
    assert set(project.agents) == {"coordinator", "researcher", "writer"}


def test_agents_have_instructions_and_tools(example_common_dir):
    project = commonadk.load(example_common_dir)

    coordinator = project.agents["coordinator"]
    assert "coordinator" in coordinator.instructions.lower()
    assert {t.name for t in coordinator.tools} == {
        "split_into_subtopics",
        "format_handoff_note",
    }

    researcher = project.agents["researcher"]
    assert {t.name for t in researcher.tools} == {"search_web", "fetch_page"}
    # tool functions are live and callable
    search_tool = next(t for t in researcher.tools if t.name == "search_web")
    assert callable(search_tool.func)
    assert "stub" in search_tool.func("electric vehicles").lower()

    writer = project.agents["writer"]
    assert {t.name for t in writer.tools} == {"count_words", "format_as_markdown"}


def test_tool_schema_metadata(example_common_dir):
    project = commonadk.load(example_common_dir)
    researcher = project.agents["researcher"]
    search_tool = next(t for t in researcher.tools if t.name == "search_web")

    assert search_tool.has_docstring
    assert search_tool.fully_typed
    assert search_tool.return_type == "str"
    assert [p.name for p in search_tool.parameters] == ["query"]
    assert search_tool.parameters[0].type == "str"
    assert search_tool.parameters[0].required is True


def test_edges_present(example_common_dir):
    project = commonadk.load(example_common_dir)
    edges = {(e.from_, e.to, e.type) for e in project.graph.edges}

    # Clean tree (plan.md v1 intersection rule / M3 hypothesis test):
    # coordinator -delegate-> researcher -handoff-> writer, with no direct
    # coordinator -> writer edge, so the reachable graph builds unmodified
    # as an ADK sub_agents tree.
    assert edges == {
        ("coordinator", "researcher", "delegate"),
        ("researcher", "writer", "handoff"),
    }
    assert project.graph.entry == "coordinator"


def test_frontmatter_stripped_from_skill(tmp_project):
    coordinator_skill = tmp_project / "coordinator" / "skill.md"
    coordinator_skill.write_text(
        "---\nrole: orchestrator\n---\n\n# Coordinator\n\nRoute work.\n"
    )
    project = commonadk.load(tmp_project)
    instructions = project.agents["coordinator"].instructions
    assert "role: orchestrator" not in instructions
    assert "Route work." in instructions


def test_load_missing_folder_raises(tmp_path):
    try:
        commonadk.load(tmp_path / "does-not-exist")
        assert False, "expected ValidationError"
    except validation.ValidationError as e:
        assert any("not found" in err for err in e.errors)


# ---------------------------------------------------------------------------
# skill.md frontmatter (feature 1): parsed and reconciled, not just stripped
# ---------------------------------------------------------------------------
#
# See docs/file-contracts.md, "skill.md -- frontmatter", for the full rule
# set these tests exercise.


def _set_agent_config(tmp_project, agent: str, **overrides) -> dict:
    path = tmp_project / agent / "agent-config.yaml"
    data = yaml.safe_load(path.read_text())
    data.update(overrides)
    path.write_text(yaml.safe_dump(data))
    return data


def test_skill_md_without_frontmatter_is_unchanged(tmp_project):
    """Compatibility contract: no frontmatter -> identical to pre-feature-1
    behavior. No warning is raised, and the config's description is left
    exactly as agent-config.yaml set it."""
    (tmp_project / "writer" / "skill.md").write_text("# Writer\n\nWrite things.\n")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        project = commonadk.load(tmp_project)

    assert project.agents["writer"].instructions == "# Writer\n\nWrite things."
    writer_warnings = [w for w in caught if "writer/skill.md" in str(w.message)]
    assert writer_warnings == []


def test_frontmatter_name_matching_config_name_is_accepted(tmp_project):
    (tmp_project / "writer" / "skill.md").write_text(
        "---\nname: writer\n---\n\n# Writer\n\nWrite things.\n"
    )
    project = commonadk.load(tmp_project)
    assert project.agents["writer"].instructions == "# Writer\n\nWrite things."


def test_frontmatter_name_mismatch_is_error(tmp_project):
    (tmp_project / "writer" / "skill.md").write_text(
        "---\nname: not-writer\n---\n\n# Writer\n"
    )
    with pytest.raises(validation.ValidationError) as excinfo:
        commonadk.load(tmp_project)
    assert any(
        "writer/skill.md" in e and "not-writer" in e for e in excinfo.value.errors
    )


def test_frontmatter_description_fills_when_agent_config_has_none(tmp_project):
    _set_agent_config(tmp_project, "writer", description="")
    (tmp_project / "writer" / "skill.md").write_text(
        "---\ndescription: From the skill's own frontmatter.\n---\n\n# Writer\n"
    )
    project = commonadk.load(tmp_project)
    assert project.agents["writer"].config.description == "From the skill's own frontmatter."


def test_frontmatter_description_conflict_config_wins_with_warning(tmp_project):
    _set_agent_config(tmp_project, "writer", description="Config wins.")
    (tmp_project / "writer" / "skill.md").write_text(
        "---\ndescription: Frontmatter loses.\n---\n\n# Writer\n"
    )
    with pytest.warns(UserWarning, match="frontmatter `description` differs"):
        project = commonadk.load(tmp_project)
    assert project.agents["writer"].config.description == "Config wins."


def test_frontmatter_unknown_key_is_warning_not_error(tmp_project):
    (tmp_project / "writer" / "skill.md").write_text(
        "---\ncustom_host_key: some-value\n---\n\n# Writer\n"
    )
    with pytest.warns(UserWarning, match="unrecognized frontmatter key 'custom_host_key'"):
        project = commonadk.load(tmp_project)
    # unlike commonadk's own YAML files (extra=\"forbid\"), an unknown
    # frontmatter key never blocks the load.
    assert "writer" in project.agents


def test_frontmatter_malformed_yaml_is_error(tmp_project):
    (tmp_project / "writer" / "skill.md").write_text(
        "---\nname: [unclosed\n---\n\n# Writer\n"
    )
    with pytest.raises(validation.ValidationError) as excinfo:
        commonadk.load(tmp_project)
    assert any(
        "writer/skill.md" in e and "invalid frontmatter YAML" in e
        for e in excinfo.value.errors
    )
