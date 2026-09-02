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


def test_runtime_key_with_installed_sdk_loads_silently(tmp_project, recwarn):
    # A `runtime:` naming a real, installed target is honored, not warned
    # about -- see docs/mixed-target-design.md, "What `runtime:` means now".
    pytest.importorskip("google.adk")

    cfg_path = tmp_project / "researcher" / "agent-config.yaml"
    data = _load_yaml(cfg_path)
    data["runtime"] = "google-adk"
    _dump_yaml(cfg_path, data)

    project = commonadk.load(tmp_project)

    assert project.agents["researcher"].config.runtime == "google-adk"
    assert not any("runtime" in str(w.message) for w in recwarn.list)


def test_no_runtime_check_when_unset(tmp_project, recwarn):
    # No agent sets `runtime:` -- the compatibility baseline this feature
    # must not disturb: no warning, no error, no adapter/SDK import at all.
    commonadk.load(tmp_project)
    assert not any("runtime" in str(w.message) for w in recwarn.list)


def test_unknown_runtime_name_errors(tmp_project):
    cfg_path = tmp_project / "researcher" / "agent-config.yaml"
    data = _load_yaml(cfg_path)
    data["runtime"] = "not-a-real-sdk"
    _dump_yaml(cfg_path, data)

    with pytest.raises(ValidationError) as exc_info:
        commonadk.load(tmp_project)

    errors = exc_info.value.errors
    assert any(
        "not-a-real-sdk" in e and "google-adk" in e for e in errors
    )  # known targets listed


def test_runtime_missing_sdk_gives_install_hint(tmp_project, monkeypatch):
    # Simulate google-adk not being installed by making the adapter
    # registry's import of it fail -- same technique as
    # test_adapter_google.py's test_get_adapter_missing_sdk_gives_install_hint.
    import importlib

    import commonadk.adapters as adapters_pkg

    real_import_module = importlib.import_module

    def fake_import_module(name, *args, **kwargs):
        if name == "commonadk.adapters.google_adk":
            raise ImportError("No module named 'google'")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(adapters_pkg, "import_module", fake_import_module)

    cfg_path = tmp_project / "researcher" / "agent-config.yaml"
    data = _load_yaml(cfg_path)
    data["runtime"] = "google-adk"
    _dump_yaml(cfg_path, data)

    with pytest.raises(ValidationError) as exc_info:
        commonadk.load(tmp_project)

    errors = exc_info.value.errors
    assert any(
        "researcher" in e and "google-adk" in e and "pip install" in e for e in errors
    )


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
