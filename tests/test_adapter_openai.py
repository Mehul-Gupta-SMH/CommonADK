"""Tests for the OpenAI Agents adapter (M3).

Everything here is offline: constructing `agents.Agent` objects requires no
network access or API key -- `TAVILY_API_KEY` is only checked for
*presence*, via `monkeypatch.setenv`, never actually used to call anything.

`pytest.importorskip` at module scope means this whole file is skipped, not
failed, when `openai-agents` (imported as `agents`) isn't installed -- the
core suite must stay green either way.
"""

import yaml
import pytest

pytest.importorskip("agents")

import commonadk  # noqa: E402  (import after importorskip, deliberately)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tavily_env(monkeypatch):
    """Satisfy researcher's one *required* env var; POSTGRES_DSN stays unset
    on purpose -- it's declared optional and must never block a build."""
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.delenv("POSTGRES_DSN", raising=False)


@pytest.fixture()
def multi_parent_project(tmp_project, tavily_env):
    """The example, with a `coordinator -> writer` delegate edge added back
    on top of its shipped `coordinator -> researcher -> writer` tree, so
    `writer` becomes reachable from two parents. Unlike Google ADK
    (test_adapter_google.py's fixture of the same name), this must BUILD
    SUCCESSFULLY for OpenAI Agents -- handoffs are references, not a tree.
    """
    interactions_path = tmp_project / "interactions.yaml"
    data = yaml.safe_load(interactions_path.read_text())
    data["edges"].append({"from": "coordinator", "to": "writer", "type": "delegate"})
    interactions_path.write_text(yaml.safe_dump(data))
    return commonadk.load(tmp_project)


@pytest.fixture()
def cyclic_project(tmp_project, tavily_env):
    """A cycle in the reachable graph (writer -> coordinator, closing the
    loop). Unlike Google ADK, this must BUILD SUCCESSFULLY for OpenAI
    Agents -- see openai_agents.py's module docstring for why references
    (not a parent-tracked tree) make cycles constructible.
    """
    interactions_path = tmp_project / "interactions.yaml"
    data = yaml.safe_load(interactions_path.read_text())
    data["edges"] = [
        {"from": "coordinator", "to": "researcher", "type": "delegate"},
        {"from": "researcher", "to": "writer", "type": "handoff"},
        {"from": "writer", "to": "coordinator", "type": "handoff"},
    ]
    interactions_path.write_text(yaml.safe_dump(data))
    return commonadk.load(tmp_project)


# ---------------------------------------------------------------------------
# handoff graph construction
# ---------------------------------------------------------------------------


def test_coordinator_build_happy_path_on_example(example_common_dir, tavily_env):
    """The shipped research-crew example -- coordinator -delegate->
    researcher -handoff-> writer -- must build end-to-end unmodified on the
    OpenAI Agents target, exactly as it does on Google ADK (test_adapter_
    google.py). This is the M3 hypothesis test's entry point.
    """
    project = commonadk.load(example_common_dir)
    agent = project.build("coordinator", target="openai")

    assert agent.name == "coordinator"
    assert agent.instructions.strip() != ""
    assert [a.name for a in agent.handoffs] == ["researcher"]

    researcher = agent.handoffs[0]
    assert researcher.instructions.strip() != ""
    assert [a.name for a in researcher.handoffs] == ["writer"]

    writer = researcher.handoffs[0]
    assert writer.instructions.strip() != ""
    assert writer.handoffs == []


def test_multi_parent_graph_builds_with_shared_instance(multi_parent_project):
    """KEY DIFFERENCE from Google ADK: a multi-parent graph must build
    successfully here, and both parents must reference the *same* `writer`
    `Agent` instance (identity, not just equal names) -- `handoffs` is a
    list of references, not a tree of owned children.
    """
    coordinator = multi_parent_project.build("coordinator", target="openai")

    researcher = next(a for a in coordinator.handoffs if a.name == "researcher")
    writer_via_coordinator = next(a for a in coordinator.handoffs if a.name == "writer")
    writer_via_researcher = researcher.handoffs[0]

    assert writer_via_researcher.name == "writer"
    assert writer_via_coordinator is writer_via_researcher


def test_cyclic_graph_builds_with_wired_handoff_references(cyclic_project):
    """KEY DIFFERENCE from Google ADK: a cyclic graph must build
    successfully here (see cyclic_project fixture and openai_agents.py's
    module docstring) -- writer's handoffs wire back around to the same
    coordinator instance the build started from.
    """
    coordinator = cyclic_project.build("coordinator", target="openai")

    researcher = coordinator.handoffs[0]
    writer = researcher.handoffs[0]

    assert [a.name for a in writer.handoffs] == ["coordinator"]
    assert writer.handoffs[0] is coordinator


# ---------------------------------------------------------------------------
# model routing
# ---------------------------------------------------------------------------


def test_openai_model_is_passed_natively_bare(tmp_project, tavily_env):
    """An agent configured with an `openai/...` LiteLLM-format model must
    get the BARE model id passed natively (no LitellmModel wrapper).
    """
    writer_cfg = tmp_project / "writer" / "agent-config.yaml"
    data = yaml.safe_load(writer_cfg.read_text())
    data["model"] = "openai/gpt-4o"
    writer_cfg.write_text(yaml.safe_dump(data))

    project = commonadk.load(tmp_project)
    agent = project.build("writer", target="openai")

    assert agent.model == "gpt-4o"


@pytest.mark.parametrize(
    "model_alias_value",
    ["fast", "smart"],  # fast -> gemini/..., smart -> anthropic/...
)
def test_non_openai_model_is_litellm_wrapped(tmp_project, tavily_env, model_alias_value):
    """Non-openai providers (gemini and anthropic, via config.yaml aliases)
    must be wrapped in agents.extensions.models.litellm_model.LitellmModel,
    carrying the FULL LiteLLM-format string.
    """
    from agents.extensions.models.litellm_model import LitellmModel

    writer_cfg = tmp_project / "writer" / "agent-config.yaml"
    data = yaml.safe_load(writer_cfg.read_text())
    data["model"] = model_alias_value
    writer_cfg.write_text(yaml.safe_dump(data))

    project = commonadk.load(tmp_project)
    agent = project.build("writer", target="openai")

    assert isinstance(agent.model, LitellmModel)
    expected = project.config.model_aliases[model_alias_value]
    assert agent.model.model == expected


def test_per_target_override_wins(tmp_project, tavily_env):
    """researcher's base model is gemini/gemini-2.5-pro, but a
    `targets.openai.model` override must win and is passed through as-is
    (already SDK-native form).
    """
    researcher_cfg = tmp_project / "researcher" / "agent-config.yaml"
    data = yaml.safe_load(researcher_cfg.read_text())
    data["targets"]["openai"] = {"model": "gpt-4o-mini"}
    researcher_cfg.write_text(yaml.safe_dump(data))

    project = commonadk.load(tmp_project)
    agent = project.build("researcher", target="openai")

    assert agent.model == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# env preflight
# ---------------------------------------------------------------------------


def test_missing_required_env_var_blocks_build(example_common_dir, monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("POSTGRES_DSN", raising=False)
    project = commonadk.load(example_common_dir)

    with pytest.raises(OSError) as exc_info:
        project.build("researcher", target="openai")

    message = str(exc_info.value)
    assert "TAVILY_API_KEY" in message
    assert "Search API key used by search_web" in message
    # POSTGRES_DSN is declared `required: false` -- its absence must not be
    # reported as a blocking problem.
    assert "POSTGRES_DSN" not in message


def test_optional_env_var_absence_does_not_block(example_common_dir, tavily_env):
    project = commonadk.load(example_common_dir)

    agent = project.build("researcher", target="openai")  # must not raise
    assert agent.name == "researcher"


# ---------------------------------------------------------------------------
# the hypothesis test (plan.md "v1 success criterion")
# ---------------------------------------------------------------------------


def test_same_project_builds_on_both_targets(example_common_dir, tavily_env):
    # This is the v1 success criterion from plan.md ("Hypothesis"): the same
    # `common/` folder must build and run unmodified on both Google ADK and
    # the OpenAI Agents SDK. Load the project ONCE and build the same entry
    # agent under both targets from that single Project object.
    pytest.importorskip("google.adk")

    project = commonadk.load(example_common_dir)

    google_agent = project.build("coordinator", target="google-adk")
    openai_agent = project.build("coordinator", target="openai")

    assert google_agent.name == "coordinator"
    assert openai_agent.name == "coordinator"
