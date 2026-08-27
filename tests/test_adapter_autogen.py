"""Tests for the AutoGen adapter (M7).

Everything here is offline: constructing `AssistantAgent`/`Swarm`/model
client objects touches no network. Unlike every other adapter's tests in
this codebase, this file DOES need fake provider API keys set up front --
see autogen_adapter.py's module docstring, "Offline construction":
`OpenAIChatCompletionClient`/`AnthropicChatCompletionClient.__init__` eagerly
construct the underlying `openai`/`anthropic` SDK client, which raises
immediately if no api key is discoverable (kwarg or env var) -- no network
call is made either way, but a key-shaped string must exist somewhere for
`build()` to even return.

`pytest.importorskip` at module scope means this whole file is skipped, not
failed, when `autogen_agentchat` isn't installed -- the core suite must stay
green either way.
"""

from __future__ import annotations

import yaml
import pytest

pytest.importorskip("autogen_agentchat")

import commonadk  # noqa: E402  (import after importorskip, deliberately)
from autogen_agentchat.agents import AssistantAgent  # noqa: E402
from autogen_agentchat.teams import Swarm  # noqa: E402
from autogen_ext.models.anthropic import AnthropicChatCompletionClient  # noqa: E402
from autogen_ext.models.openai import OpenAIChatCompletionClient  # noqa: E402


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
def provider_keys_env(monkeypatch):
    """Fake, never-used API keys for every provider this adapter's model
    clients construct eagerly (see module docstring, "Offline construction").
    These are never sent anywhere in this test file -- construction alone is
    enough to trigger the underlying SDK's own presence check.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test")


@pytest.fixture()
def multi_parent_project(tmp_project, tavily_env, provider_keys_env):
    """The example, with a `coordinator -> writer` delegate edge added back
    on top of its shipped `coordinator -> researcher -> writer` tree, so
    `writer` becomes reachable from two parents. Must BUILD SUCCESSFULLY --
    handoffs are plain name strings resolved by `Swarm` at run time, not a
    parent-tracked tree, so a name reachable by two paths is just built once
    (memoized) and appears once in `Swarm`'s participants.
    """
    interactions_path = tmp_project / "interactions.yaml"
    data = yaml.safe_load(interactions_path.read_text())
    data["edges"].append({"from": "coordinator", "to": "writer", "type": "delegate"})
    interactions_path.write_text(yaml.safe_dump(data))
    return commonadk.load(tmp_project)


@pytest.fixture()
def cyclic_project(tmp_project, tavily_env, provider_keys_env):
    """A cycle in the reachable graph (writer -> coordinator, closing the
    loop). Must BUILD SUCCESSFULLY -- handoff targets are plain strings with
    no "already has a parent" guard anywhere in construction (see module
    docstring, "KEY PROPERTY").
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


def _handoff_targets(agent: AssistantAgent) -> set[str]:
    """AssistantAgent has no public accessor for its configured handoffs --
    `_handoffs` is a private `dict[str, HandoffBase]` keyed by the generated
    tool name (e.g. "transfer_to_writer"), so this pulls out just the
    target agent names, which is what interactions.yaml actually encodes.
    """
    return {h.target for h in agent._handoffs.values()}


def _tool_names(agent: AssistantAgent) -> set[str]:
    return {t.name for t in agent._tools}


# ---------------------------------------------------------------------------
# graph construction / return-shape decision
# ---------------------------------------------------------------------------


def test_coordinator_build_happy_path_on_example(example_common_dir, tavily_env, provider_keys_env):
    """The shipped research-crew example -- coordinator -delegate->
    researcher -handoff-> writer -- has an outgoing edge at the build root,
    so this must build a ready-to-run `Swarm` of every reachable agent, root
    first (see module docstring, "WHAT build() RETURNS").
    """
    project = commonadk.load(example_common_dir)
    team = project.build("coordinator", target="autogen")

    assert isinstance(team, Swarm)
    assert team._participant_names == ["coordinator", "researcher", "writer"]
    assert team._max_turns == 3  # len(reachable) -- see module docstring

    coordinator, researcher, writer = team._participants

    assert coordinator.name == "coordinator"
    assert coordinator.description == project.agents["coordinator"].config.description
    assert coordinator._system_messages[0].content == project.agents["coordinator"].instructions
    assert _tool_names(coordinator) == {"split_into_subtopics", "format_handoff_note"}
    assert _handoff_targets(coordinator) == {"researcher"}

    assert researcher.name == "researcher"
    assert _tool_names(researcher) == {"search_web", "fetch_page"}
    assert _handoff_targets(researcher) == {"writer"}  # the deep edge

    assert writer.name == "writer"
    assert _tool_names(writer) == {"count_words", "format_as_markdown"}
    assert _handoff_targets(writer) == set()  # no outgoing edges


def test_researcher_build_wires_writer_as_only_other_participant(
    example_common_dir, tavily_env, provider_keys_env
):
    """Building from `researcher` directly (not through `coordinator`) makes
    researcher the Swarm's first participant -- only `writer` (its one
    reachable agent) joins alongside it.
    """
    project = commonadk.load(example_common_dir)
    team = project.build("researcher", target="autogen")

    assert isinstance(team, Swarm)
    assert team._participant_names == ["researcher", "writer"]
    assert team._max_turns == 2


def test_writer_build_returns_bare_agent_not_a_team(example_common_dir, tavily_env, provider_keys_env):
    """`writer` has no outgoing edges -- nothing to hand off to, so this must
    return the bare `AssistantAgent` itself, not a one-member `Swarm` (see
    module docstring, "WHAT build() RETURNS").
    """
    project = commonadk.load(example_common_dir)
    agent = project.build("writer", target="autogen")

    assert isinstance(agent, AssistantAgent)
    assert not isinstance(agent, Swarm)
    assert agent.name == "writer"
    assert _handoff_targets(agent) == set()


def test_multi_parent_graph_builds_with_one_shared_participant(multi_parent_project):
    """KEY PROPERTY: a multi-parent graph builds successfully and `writer`
    appears exactly once among the Swarm's participants, not duplicated per
    parent that references it.
    """
    team = multi_parent_project.build("coordinator", target="autogen")

    assert team._participant_names == ["coordinator", "researcher", "writer"]
    assert len(team._participants) == 3  # not duplicated

    coordinator = team._participants[0]
    assert _handoff_targets(coordinator) == {"researcher", "writer"}


def test_cyclic_graph_builds_without_recursion_hazard(cyclic_project):
    """KEY PROPERTY: a cycle back to the build root builds successfully --
    `writer`'s handoff back to `coordinator` is just another target-name
    string, not a re-visit of already-under-construction state.
    """
    team = cyclic_project.build("coordinator", target="autogen")

    assert team._participant_names == ["coordinator", "researcher", "writer"]
    writer = next(p for p in team._participants if p.name == "writer")
    assert _handoff_targets(writer) == {"coordinator"}


# ---------------------------------------------------------------------------
# model routing
# ---------------------------------------------------------------------------


def test_openai_model_resolves_to_native_client(tmp_project, tavily_env, provider_keys_env):
    """An agent configured with an `openai/...` model must get the bare
    model id passed straight to `OpenAIChatCompletionClient`.
    """
    writer_cfg = tmp_project / "writer" / "agent-config.yaml"
    data = yaml.safe_load(writer_cfg.read_text())
    data["model"] = "openai/gpt-4o"
    writer_cfg.write_text(yaml.safe_dump(data))

    project = commonadk.load(tmp_project)
    agent = project.build("writer", target="autogen")

    client = agent._model_client
    assert isinstance(client, OpenAIChatCompletionClient)
    assert client._create_args["model"] == "gpt-4o"


def test_gemini_model_routes_through_openai_client_with_explicit_model_info(
    example_common_dir, tavily_env, provider_keys_env
):
    """researcher's model (gemini/gemini-2.5-pro in the shipped example,
    unmodified -- no override needed, unlike the Claude Agent SDK target)
    must route through `OpenAIChatCompletionClient`'s built-in Gemini
    special-casing: an auto-swapped base_url, AND an explicit `model_info`
    this adapter supplies itself (gemini-2.5-pro is not in autogen_ext's own
    bundled table -- see module docstring, "Model routing" -- so relying on
    that table would raise here).
    """
    project = commonadk.load(example_common_dir)
    team = project.build("coordinator", target="autogen")
    researcher = next(p for p in team._participants if p.name == "researcher")

    client = researcher._model_client
    assert isinstance(client, OpenAIChatCompletionClient)
    assert client._create_args["model"] == "gemini-2.5-pro"
    assert client._create_args["temperature"] == 0.2
    assert client._create_args["max_tokens"] == 4096
    assert str(client._client.base_url) == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert client.model_info["function_calling"] is True


def test_anthropic_model_routes_to_native_client_with_explicit_model_info(
    tmp_project, tavily_env, provider_keys_env
):
    """writer's base model is the `fast` alias; overriding its *base* model
    (not a per-target override) to the `smart` alias (-> anthropic/claude-
    sonnet-5) must route through `AnthropicChatCompletionClient` with THIS
    adapter's own explicit `model_info` -- verified necessary in the module
    docstring, "Model routing": trusting autogen_ext's bundled Anthropic
    table for an aliased id like "claude-sonnet-5" silently resolves to
    function_calling=False (it fuzzy-matches an unrelated legacy model),
    which would make this build blow up as soon as writer's tools are
    attached. Explicit model_info sidesteps that entirely.
    """
    writer_cfg = tmp_project / "writer" / "agent-config.yaml"
    data = yaml.safe_load(writer_cfg.read_text())
    data["model"] = "smart"
    writer_cfg.write_text(yaml.safe_dump(data))

    project = commonadk.load(tmp_project)
    agent = project.build("writer", target="autogen")  # must not raise

    client = agent._model_client
    assert isinstance(client, AnthropicChatCompletionClient)
    assert client._create_args["model"] == "claude-sonnet-5"
    assert client.model_info["function_calling"] is True
    assert _tool_names(agent) == {"count_words", "format_as_markdown"}  # tools survived


def test_unsupported_provider_raises_clear_error(tmp_project, tavily_env, provider_keys_env):
    """A provider this adapter ships no client for at all (no litellm
    fallback here, unlike CrewAI) must raise a clear, actionable error
    naming the agent and its resolved model string.
    """
    writer_cfg = tmp_project / "writer" / "agent-config.yaml"
    data = yaml.safe_load(writer_cfg.read_text())
    data["model"] = "cohere/command-r"
    writer_cfg.write_text(yaml.safe_dump(data))

    project = commonadk.load(tmp_project)
    with pytest.raises(ValueError) as exc_info:
        project.build("writer", target="autogen")

    message = str(exc_info.value)
    assert "writer" in message
    assert "cohere/command-r" in message
    assert "targets.autogen.model" in message


def test_per_target_override_wins(tmp_project, tavily_env, provider_keys_env):
    """writer's base model is the `fast` alias (-> gemini/gemini-2.5-flash),
    but its `targets.autogen.model` override must win and route through the
    default client (`OpenAIChatCompletionClient`) with no explicit
    model_info -- see module docstring, "Per-target override".
    """
    writer_cfg = tmp_project / "writer" / "agent-config.yaml"
    data = yaml.safe_load(writer_cfg.read_text())
    data["targets"]["autogen"] = {"model": "gpt-4o-mini"}
    writer_cfg.write_text(yaml.safe_dump(data))

    project = commonadk.load(tmp_project)
    agent = project.build("writer", target="autogen")

    client = agent._model_client
    assert isinstance(client, OpenAIChatCompletionClient)
    assert client._create_args["model"] == "gpt-4o-mini"


def test_unsupported_model_params_key_is_warned_and_ignored(tmp_project, tavily_env, provider_keys_env):
    writer_cfg = tmp_project / "writer" / "agent-config.yaml"
    data = yaml.safe_load(writer_cfg.read_text())
    data["model_params"]["top_p"] = 0.9
    writer_cfg.write_text(yaml.safe_dump(data))

    project = commonadk.load(tmp_project)
    with pytest.warns(UserWarning, match="model_params key 'top_p'"):
        agent = project.build("writer", target="autogen")

    assert agent is not None  # build still succeeds


# ---------------------------------------------------------------------------
# env preflight
# ---------------------------------------------------------------------------


def test_missing_required_env_var_blocks_build(example_common_dir, monkeypatch, provider_keys_env):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("POSTGRES_DSN", raising=False)
    project = commonadk.load(example_common_dir)

    with pytest.raises(OSError) as exc_info:
        project.build("researcher", target="autogen")

    message = str(exc_info.value)
    assert "TAVILY_API_KEY" in message
    assert "Search API key used by search_web" in message
    # POSTGRES_DSN is declared `required: false` -- its absence must not be
    # reported as a blocking problem.
    assert "POSTGRES_DSN" not in message


def test_optional_env_var_absence_does_not_block(example_common_dir, tavily_env, provider_keys_env):
    project = commonadk.load(example_common_dir)

    team = project.build("researcher", target="autogen")  # must not raise
    assert isinstance(team, Swarm)


def test_env_preflight_checks_agents_reachable_via_edges(tmp_project, monkeypatch, provider_keys_env):
    """Building `coordinator` must also check `researcher`'s env
    requirements, since researcher is reachable from coordinator via a
    delegate edge -- even though coordinator has no `requires.env` of its
    own.
    """
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("POSTGRES_DSN", raising=False)
    project = commonadk.load(tmp_project)

    with pytest.raises(OSError, match="TAVILY_API_KEY"):
        project.build("coordinator", target="autogen")


# ---------------------------------------------------------------------------
# the cross-target hypothesis test (plan.md "v1 success criterion") lives in
# tests/test_hypothesis.py, parametrized over every SDK target.
# ---------------------------------------------------------------------------
