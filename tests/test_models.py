import pytest

import commonadk


def test_resolve_model_alias(example_common_dir):
    project = commonadk.load(example_common_dir)
    # coordinator's agent-config.yaml sets model: fast (an alias)
    assert project.resolve_model("coordinator") == "gemini/gemini-2.5-flash"


def test_resolve_model_explicit_string_passthrough(example_common_dir):
    project = commonadk.load(example_common_dir)
    # researcher's agent-config.yaml sets an explicit LiteLLM-format string
    assert project.resolve_model("researcher") == "gemini/gemini-2.5-pro"


def test_resolve_model_falls_back_to_default(example_common_dir):
    project = commonadk.load(example_common_dir)
    # writer sets model: fast explicitly, matching config.yaml's default_model
    assert project.resolve_model("writer") == project.config.model_aliases["fast"]


def test_resolve_model_unknown_alias_raises(example_common_dir):
    project = commonadk.load(example_common_dir)
    project.agents["coordinator"].config.model = "nonexistent-alias"
    with pytest.raises(ValueError, match="nonexistent-alias"):
        project.resolve_model("coordinator")


def test_resolve_model_unknown_agent_raises(example_common_dir):
    project = commonadk.load(example_common_dir)
    with pytest.raises(KeyError):
        project.resolve_model("nope")


def test_check_env_reports_missing_required(example_common_dir, monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("POSTGRES_DSN", raising=False)

    project = commonadk.load(example_common_dir)
    missing = project.check_env("researcher")

    assert missing == ["TAVILY_API_KEY"]  # POSTGRES_DSN is optional, not reported


def test_check_env_satisfied(example_common_dir, monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")

    project = commonadk.load(example_common_dir)
    assert project.check_env("researcher") == []


def test_check_env_no_requirements(example_common_dir):
    project = commonadk.load(example_common_dir)
    assert project.check_env("coordinator") == []
