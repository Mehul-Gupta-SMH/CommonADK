import pytest
import yaml

import commonadk
from commonadk.validation import ValidationError


def _load_yaml(path):
    return yaml.safe_load(path.read_text())


def _dump_yaml(path, data):
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def test_unknown_tool_name_errors(tmp_project):
    cfg_path = tmp_project / "researcher" / "agent-config.yaml"
    data = _load_yaml(cfg_path)
    data["tools"].append("summon_dragon")
    _dump_yaml(cfg_path, data)

    with pytest.raises(ValidationError) as exc_info:
        commonadk.load(tmp_project)

    errors = exc_info.value.errors
    assert any("summon_dragon" in e for e in errors)


def test_tool_missing_type_hints_errors(tmp_project):
    tools_path = tmp_project / "researcher" / "tools.py"
    tools_path.write_text(
        tools_path.read_text()
        + "\n\ndef untyped_tool(x):\n    \"\"\"Missing type hints.\"\"\"\n    return x\n"
    )
    cfg_path = tmp_project / "researcher" / "agent-config.yaml"
    data = _load_yaml(cfg_path)
    data["tools"].append("untyped_tool")
    _dump_yaml(cfg_path, data)

    with pytest.raises(ValidationError) as exc_info:
        commonadk.load(tmp_project)

    errors = exc_info.value.errors
    assert any("untyped_tool" in e and "type hint" in e for e in errors)


def test_tool_missing_docstring_errors(tmp_project):
    tools_path = tmp_project / "researcher" / "tools.py"
    tools_path.write_text(
        tools_path.read_text() + "\n\ndef undocumented_tool(x: str) -> str:\n    return x\n"
    )
    cfg_path = tmp_project / "researcher" / "agent-config.yaml"
    data = _load_yaml(cfg_path)
    data["tools"].append("undocumented_tool")
    _dump_yaml(cfg_path, data)

    with pytest.raises(ValidationError) as exc_info:
        commonadk.load(tmp_project)

    errors = exc_info.value.errors
    assert any("undocumented_tool" in e and "docstring" in e for e in errors)


def test_edge_to_unknown_agent_errors(tmp_project):
    interactions_path = tmp_project / "interactions.yaml"
    data = _load_yaml(interactions_path)
    data["edges"].append({"from": "coordinator", "to": "ghostwriter", "type": "delegate"})
    _dump_yaml(interactions_path, data)

    with pytest.raises(ValidationError) as exc_info:
        commonadk.load(tmp_project)

    errors = exc_info.value.errors
    assert any("ghostwriter" in e for e in errors)


def test_bad_edge_type_errors(tmp_project):
    interactions_path = tmp_project / "interactions.yaml"
    data = _load_yaml(interactions_path)
    data["edges"].append({"from": "coordinator", "to": "researcher", "type": "teleport"})
    _dump_yaml(interactions_path, data)

    with pytest.raises(ValidationError) as exc_info:
        commonadk.load(tmp_project)

    errors = exc_info.value.errors
    assert any("teleport" in e for e in errors)


def test_folder_name_mismatch_errors(tmp_project):
    cfg_path = tmp_project / "researcher" / "agent-config.yaml"
    data = _load_yaml(cfg_path)
    data["name"] = "the_researcher"
    _dump_yaml(cfg_path, data)

    with pytest.raises(ValidationError) as exc_info:
        commonadk.load(tmp_project)

    errors = exc_info.value.errors
    assert any("researcher" in e and "the_researcher" in e for e in errors)


def test_unknown_model_alias_errors(tmp_project):
    cfg_path = tmp_project / "coordinator" / "agent-config.yaml"
    data = _load_yaml(cfg_path)
    data["model"] = "totally-made-up-alias"
    _dump_yaml(cfg_path, data)

    with pytest.raises(ValidationError) as exc_info:
        commonadk.load(tmp_project)

    errors = exc_info.value.errors
    assert any("totally-made-up-alias" in e for e in errors)


def test_entry_mismatch_errors(tmp_project):
    interactions_path = tmp_project / "interactions.yaml"
    data = _load_yaml(interactions_path)
    data["entry"] = "writer"
    _dump_yaml(interactions_path, data)

    with pytest.raises(ValidationError) as exc_info:
        commonadk.load(tmp_project)

    errors = exc_info.value.errors
    assert any("mismatch" in e for e in errors)


def test_missing_config_yaml_errors(tmp_project):
    (tmp_project / "config.yaml").unlink()

    with pytest.raises(ValidationError) as exc_info:
        commonadk.load(tmp_project)

    errors = exc_info.value.errors
    assert any("config.yaml" in e for e in errors)


def test_unknown_yaml_key_errors(tmp_project):
    cfg_path = tmp_project / "researcher" / "agent-config.yaml"
    data = _load_yaml(cfg_path)
    data["modle"] = "fast"  # typo of `model`
    _dump_yaml(cfg_path, data)

    with pytest.raises(ValidationError) as exc_info:
        commonadk.load(tmp_project)

    errors = exc_info.value.errors
    assert any("modle" in e for e in errors)


def test_runtime_key_warns_when_set(tmp_project):
    cfg_path = tmp_project / "researcher" / "agent-config.yaml"
    data = _load_yaml(cfg_path)
    data["runtime"] = "google-adk"
    _dump_yaml(cfg_path, data)

    with pytest.warns(UserWarning, match="runtime"):
        project = commonadk.load(tmp_project)

    assert project.agents["researcher"].config.runtime == "google-adk"


def test_no_runtime_warning_when_unset(tmp_project, recwarn):
    commonadk.load(tmp_project)
    assert not any("runtime" in str(w.message) for w in recwarn.list)


def test_all_errors_collected_in_one_raise(tmp_project):
    # Stack up multiple, unrelated problems at once.
    cfg_path = tmp_project / "researcher" / "agent-config.yaml"
    data = _load_yaml(cfg_path)
    data["tools"].append("summon_dragon")
    _dump_yaml(cfg_path, data)

    interactions_path = tmp_project / "interactions.yaml"
    idata = _load_yaml(interactions_path)
    idata["edges"].append({"from": "coordinator", "to": "ghostwriter", "type": "delegate"})
    _dump_yaml(interactions_path, idata)

    writer_cfg_path = tmp_project / "writer" / "agent-config.yaml"
    wdata = _load_yaml(writer_cfg_path)
    wdata["name"] = "the_writer"
    _dump_yaml(writer_cfg_path, wdata)

    with pytest.raises(ValidationError) as exc_info:
        commonadk.load(tmp_project)

    errors = exc_info.value.errors
    assert len(errors) >= 3
    assert any("summon_dragon" in e for e in errors)
    assert any("ghostwriter" in e for e in errors)
    assert any("the_writer" in e for e in errors)
