import commonadk


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

    assert ("coordinator", "researcher", "delegate") in edges
    assert ("coordinator", "writer", "delegate") in edges
    assert ("researcher", "writer", "handoff") in edges
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
    import commonadk.validation as validation

    try:
        commonadk.load(tmp_path / "does-not-exist")
        assert False, "expected ValidationError"
    except validation.ValidationError as e:
        assert any("not found" in err for err in e.errors)
