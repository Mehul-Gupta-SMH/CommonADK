"""Tests for the CrewAI adapter (M6).

Everything here is offline: constructing `crewai.Agent`/`crewai.LLM`/
`crewai.Crew` objects touches no network -- see crewai_adapter.py's module
docstring, "Telemetry / offline construction", for why `build()` itself
never emits a telemetry event either way. This module still sets
`CREWAI_DISABLE_TELEMETRY`/`OTEL_SDK_DISABLED` up front (belt-and-braces,
matching the module docstring's own caveat) so nothing here depends on that
being true only by accident, and so a future crewai version that *does*
touch the network at construction time fails loudly instead of hanging.

`pytest.importorskip` at module scope means this whole file is skipped, not
failed, when `crewai` isn't installed -- the core suite must stay green
either way.
"""

from __future__ import annotations

import os

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

import yaml
import pytest

pytest.importorskip("crewai")

import commonadk  # noqa: E402  (import after importorskip, deliberately)
from crewai import Crew, Process  # noqa: E402
from crewai.llms.base_llm import BaseLLM  # noqa: E402


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
    `writer` becomes reachable from two parents. Must BUILD SUCCESSFULLY
    here -- `crew.agents` is a flat list built once per logical agent name
    from `_reachable_agents`'s already-deduped result, so a name reachable
    by two paths is simply the same `Agent` instance appearing once.
    """
    interactions_path = tmp_project / "interactions.yaml"
    data = yaml.safe_load(interactions_path.read_text())
    data["edges"].append({"from": "coordinator", "to": "writer", "type": "delegate"})
    interactions_path.write_text(yaml.safe_dump(data))
    return commonadk.load(tmp_project)


@pytest.fixture()
def cyclic_project(tmp_project, tavily_env):
    """A cycle in the reachable graph (writer -> coordinator, closing the
    loop). Must BUILD SUCCESSFULLY here -- the build root is never added to
    its own `crew.agents` (it becomes `manager_agent` instead), so a cycle
    back to it is a no-op, not a hazard.
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
# crew construction / AgentSpec mapping / edge mapping
# ---------------------------------------------------------------------------


def test_coordinator_build_happy_path_on_example(example_common_dir, tavily_env):
    """The shipped research-crew example -- coordinator -delegate->
    researcher -handoff-> writer -- builds a hierarchical crew: coordinator
    becomes `manager_agent` (it has an outgoing edge), researcher and writer
    (both reachable, including writer which is only reachable via
    researcher -- the deep-edge case) become crew members.
    """
    project = commonadk.load(example_common_dir)

    with pytest.warns(UserWarning, match="manager role cannot hold tools"):
        crew = project.build("coordinator", target="crewai")

    assert isinstance(crew, Crew)
    assert crew.process == Process.hierarchical
    assert crew.tasks == []

    assert crew.manager_agent is not None
    manager = crew.manager_agent
    assert manager.role == "coordinator"
    assert manager.goal == project.agents["coordinator"].config.description
    assert manager.backstory == project.agents["coordinator"].instructions
    assert manager.allow_delegation is True
    # coordinator has its own tools declared, but the manager role can't
    # hold tools (kickoff() raises if it does) -- see module docstring.
    assert manager.tools == []

    # researcher and writer -- both reachable, deep-edge (writer) included.
    assert {a.role for a in crew.agents} == {"researcher", "writer"}

    researcher = next(a for a in crew.agents if a.role == "researcher")
    assert researcher.goal == project.agents["researcher"].config.description
    assert researcher.backstory == project.agents["researcher"].instructions
    assert {t.name for t in researcher.tools} == {"search_web", "fetch_page"}
    # researcher has its own outgoing edge (-> writer) -- it must be able
    # to delegate.
    assert researcher.allow_delegation is True

    writer = next(a for a in crew.agents if a.role == "writer")
    assert {t.name for t in writer.tools} == {"count_words", "format_as_markdown"}
    # writer has no outgoing edges -- it must NOT be able to delegate.
    assert writer.allow_delegation is False


def test_researcher_build_wires_writer_as_only_member(example_common_dir, tavily_env):
    """Building from `researcher` directly (not through `coordinator`) makes
    researcher the manager -- only `writer` (its one reachable agent) ends
    up as a crew member.
    """
    project = commonadk.load(example_common_dir)

    with pytest.warns(UserWarning, match="manager role cannot hold tools"):
        crew = project.build("researcher", target="crewai")

    assert crew.process == Process.hierarchical
    assert crew.manager_agent.role == "researcher"
    assert crew.manager_agent.tools == []  # researcher's own tools dropped
    assert {a.role for a in crew.agents} == {"writer"}


def test_writer_build_falls_back_to_solo_sequential_crew(example_common_dir, tavily_env):
    """`writer` has no outgoing edges -- nothing to manage, so this must
    fall back to a solo-member sequential crew (see module docstring,
    "Manager-or-solo-member decision") rather than an empty-agents
    hierarchical crew (verified unbuildable -- see module docstring).
    """
    project = commonadk.load(example_common_dir)
    crew = project.build("writer", target="crewai")

    assert crew.process == Process.sequential
    assert crew.manager_agent is None
    assert [a.role for a in crew.agents] == ["writer"]
    # writer is the sole member here (not a manager), so it keeps its own
    # tools -- the "manager can't hold tools" restriction doesn't apply.
    assert {t.name for t in crew.agents[0].tools} == {"count_words", "format_as_markdown"}
    assert crew.agents[0].allow_delegation is False


def test_multi_parent_graph_builds_with_one_shared_member(multi_parent_project):
    """KEY PROPERTY of the flat crew membership: a multi-parent graph builds
    successfully and `writer` appears exactly once in `crew.agents`, not
    duplicated per parent.
    """
    with pytest.warns(UserWarning, match="manager role cannot hold tools"):
        crew = multi_parent_project.build("coordinator", target="crewai")

    assert {a.role for a in crew.agents} == {"researcher", "writer"}
    assert len(crew.agents) == 2  # not duplicated


def test_cyclic_graph_builds_without_recursion_hazard(cyclic_project):
    """KEY PROPERTY of the flat crew membership: a cycle back to the build
    root builds successfully -- coordinator is simply never added to its
    own `crew.agents` (it's the `manager_agent`, excluded by construction).
    """
    with pytest.warns(UserWarning, match="manager role cannot hold tools"):
        crew = cyclic_project.build("coordinator", target="crewai")

    assert {a.role for a in crew.agents} == {"researcher", "writer"}
    assert crew.manager_agent.role == "coordinator"
    # writer's edge back to coordinator can't add coordinator as a further
    # member (coordinator IS the manager), but writer still gets
    # allow_delegation=True since it does have an outgoing edge.
    writer = next(a for a in crew.agents if a.role == "writer")
    assert writer.allow_delegation is True


# ---------------------------------------------------------------------------
# model routing
# ---------------------------------------------------------------------------


def test_model_string_passes_through_with_model_params(example_common_dir, tavily_env):
    """CrewAI speaks LiteLLM-format strings natively -- researcher's
    resolved model (gemini/gemini-2.5-pro, no per-target override in the
    shipped example) and its model_params (temperature, max_tokens) must
    land directly on the built agent's `llm`, with no unsupported-provider
    error anywhere in this target.
    """
    project = commonadk.load(example_common_dir)
    crew = project.build("researcher", target="crewai")

    llm = crew.manager_agent.llm
    assert isinstance(llm, BaseLLM)
    assert llm.model == "gemini-2.5-pro"  # native gemini routing strips the prefix
    assert llm.temperature == 0.2
    assert llm.max_tokens == 4096


def test_non_native_provider_falls_back_to_litellm_with_no_error(tmp_project, tavily_env):
    """A provider crewai's `LLM` factory doesn't recognize natively must
    still build -- no unsupported-provider error on this target, unlike the
    other three adapters (see module docstring, "Model routing"). The
    litellm fallback path keeps the FULL resolved string on `.model`
    (native routing strips the provider prefix; litellm's fallback doesn't).
    """
    writer_cfg = tmp_project / "writer" / "agent-config.yaml"
    data = yaml.safe_load(writer_cfg.read_text())
    data["targets"]["crewai"] = {"model": "together_ai/some-made-up-model"}
    writer_cfg.write_text(yaml.safe_dump(data))

    project = commonadk.load(tmp_project)
    crew = project.build("writer", target="crewai")  # must not raise

    llm = crew.agents[0].llm
    assert isinstance(llm, BaseLLM)
    assert llm.model == "together_ai/some-made-up-model"


def test_per_target_override_wins(tmp_project, tavily_env):
    """writer's base model is the `fast` alias (-> gemini/gemini-2.5-flash),
    but its `targets.crewai.model` override must win and is passed through
    as-is.
    """
    writer_cfg = tmp_project / "writer" / "agent-config.yaml"
    data = yaml.safe_load(writer_cfg.read_text())
    data["targets"]["crewai"] = {"model": "anthropic/claude-opus-5"}
    writer_cfg.write_text(yaml.safe_dump(data))

    project = commonadk.load(tmp_project)
    crew = project.build("writer", target="crewai")

    assert crew.agents[0].llm.model == "claude-opus-5"  # native anthropic routing


def test_unsupported_model_params_key_is_warned_and_ignored(tmp_project, tavily_env):
    writer_cfg = tmp_project / "writer" / "agent-config.yaml"
    data = yaml.safe_load(writer_cfg.read_text())
    data["model_params"]["top_p"] = 0.9
    writer_cfg.write_text(yaml.safe_dump(data))

    project = commonadk.load(tmp_project)
    with pytest.warns(UserWarning, match="model_params key 'top_p'"):
        crew = project.build("writer", target="crewai")

    assert crew is not None  # build still succeeds


# ---------------------------------------------------------------------------
# env preflight
# ---------------------------------------------------------------------------


def test_missing_required_env_var_blocks_build(example_common_dir, monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("POSTGRES_DSN", raising=False)
    project = commonadk.load(example_common_dir)

    with pytest.raises(OSError) as exc_info:
        project.build("researcher", target="crewai")

    message = str(exc_info.value)
    assert "TAVILY_API_KEY" in message
    assert "Search API key used by search_web" in message
    # POSTGRES_DSN is declared `required: false` -- its absence must not be
    # reported as a blocking problem.
    assert "POSTGRES_DSN" not in message


def test_optional_env_var_absence_does_not_block(example_common_dir, tavily_env):
    project = commonadk.load(example_common_dir)

    crew = project.build("researcher", target="crewai")  # must not raise
    assert isinstance(crew, Crew)


def test_env_preflight_checks_agents_reachable_via_edges(tmp_project, monkeypatch):
    """Building `coordinator` must also check `researcher`'s env
    requirements, since researcher is reachable from coordinator via a
    delegate edge -- even though coordinator has no `requires.env` of its
    own.
    """
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("POSTGRES_DSN", raising=False)
    project = commonadk.load(tmp_project)

    with pytest.raises(OSError, match="TAVILY_API_KEY"):
        project.build("coordinator", target="crewai")
