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
from autogen_agentchat.tools import AgentTool, TeamTool  # noqa: E402
from autogen_ext.models.anthropic import AnthropicChatCompletionClient  # noqa: E402
from autogen_ext.models.openai import OpenAIChatCompletionClient  # noqa: E402
from commonadk.adapters import autogen_adapter  # noqa: E402


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
    """All three edges rewritten to `handoff` (`coordinator -> researcher ->
    writer` plus a direct `coordinator -> writer`), so `writer` is
    `handoff`-reachable from two parents within the same Swarm's
    participant set. Must BUILD SUCCESSFULLY -- handoffs are plain name
    strings resolved by `Swarm` at run time, not a parent-tracked tree, so a
    name reachable by two paths is just built once (memoized) and appears
    once in `Swarm`'s participants.

    Since issue #10 (delegate/handoff distinction): a `delegate` edge here
    instead would put `writer` behind an independent `AgentTool` on
    coordinator, entirely separate from the Swarm's participant set -- not
    what this fixture is testing (see `test_delegate_edge_bypasses_swarm_
    participant_dedup` below, which exercises exactly that).
    """
    interactions_path = tmp_project / "interactions.yaml"
    data = yaml.safe_load(interactions_path.read_text())
    data["edges"] = [
        {"from": "coordinator", "to": "researcher", "type": "handoff"},
        {"from": "researcher", "to": "writer", "type": "handoff"},
        {"from": "coordinator", "to": "writer", "type": "handoff"},
    ]
    interactions_path.write_text(yaml.safe_dump(data))
    return commonadk.load(tmp_project)


@pytest.fixture()
def cyclic_project(tmp_project, tavily_env, provider_keys_env):
    """A cycle in the reachable graph (writer -> coordinator, closing the
    loop), all edges `handoff`. Must BUILD SUCCESSFULLY -- handoff targets
    are plain strings with no "already has a parent" guard anywhere in
    construction (see module docstring, "KEY PROPERTY"). (A cycle closed by
    `delegate` edges instead is a genuine construction-time hazard now --
    see `test_delegate_cycle_is_rejected` below.)
    """
    interactions_path = tmp_project / "interactions.yaml"
    data = yaml.safe_load(interactions_path.read_text())
    data["edges"] = [
        {"from": "coordinator", "to": "researcher", "type": "handoff"},
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
    researcher -handoff-> writer.

    Since issue #10 (delegate/handoff distinction): coordinator's only edge
    is `delegate`, and it has no *handoff* edges of its own, so this must
    return the BARE `coordinator` AssistantAgent (not a Swarm) with a
    `delegate_to_researcher` tool wired in -- see module docstring, "WHAT
    build() RETURNS". researcher itself hands off to writer, so
    researcher's own recursive build is a 2-participant Swarm, and that
    delegate tool must be a `TeamTool` wrapping it, not a plain `AgentTool`
    (see module docstring, "Recursive construction, and the AgentTool/
    TeamTool split").
    """
    project = commonadk.load(example_common_dir)
    coordinator = project.build("coordinator", target="autogen")

    assert isinstance(coordinator, AssistantAgent)
    assert not isinstance(coordinator, Swarm)
    assert coordinator.name == "coordinator"
    assert coordinator.description == project.agents["coordinator"].config.description
    assert coordinator._system_messages[0].content == project.agents["coordinator"].instructions
    assert _handoff_targets(coordinator) == set()  # coordinator has no handoff edges

    tool_names = {t.name for t in coordinator._tools}
    assert {"split_into_subtopics", "format_handoff_note", "delegate_to_researcher"} <= tool_names

    delegate_tool = next(t for t in coordinator._tools if t.name == "delegate_to_researcher")
    assert isinstance(delegate_tool, TeamTool)
    team = delegate_tool._team
    assert isinstance(team, Swarm)
    assert team._participant_names == ["researcher", "writer"]
    assert team._max_turns == 2

    researcher, writer = team._participants

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
# delegate/handoff distinction (issue #10, first checkbox)
# ---------------------------------------------------------------------------


def test_delegate_edge_to_a_leaf_becomes_agent_tool(tmp_project, tavily_env, provider_keys_env):
    """A `delegate` edge to a destination with no outgoing `handoff` edges
    of its own must be wrapped in a plain `AgentTool` (its own build is a
    bare AssistantAgent, not a Swarm) -- see module docstring, "Recursive
    construction, and the AgentTool/TeamTool split". `AgentTool` derives its
    tool name from the wrapped agent's own name (verified in the module
    docstring), so the tool is named "writer", not "delegate_to_writer".
    """
    interactions_path = tmp_project / "interactions.yaml"
    data = yaml.safe_load(interactions_path.read_text())
    data["edges"] = [{"from": "coordinator", "to": "writer", "type": "delegate"}]
    interactions_path.write_text(yaml.safe_dump(data))
    project = commonadk.load(tmp_project)

    coordinator = project.build("coordinator", target="autogen")

    assert isinstance(coordinator, AssistantAgent)
    assert not isinstance(coordinator, Swarm)
    assert _handoff_targets(coordinator) == set()

    delegate_tool = next(t for t in coordinator._tools if t.name == "writer")
    assert isinstance(delegate_tool, AgentTool)
    assert delegate_tool._agent.name == "writer"


def test_handoff_edge_stays_in_handoffs_not_delegate_tools(
    tmp_project, tavily_env, provider_keys_env
):
    """A `handoff` edge from researcher -> writer must produce an entry in
    `researcher._handoffs` and must NOT also appear as a delegate tool."""
    project = commonadk.load(tmp_project)  # unmodified: researcher -handoff-> writer
    researcher = project.build("researcher", target="autogen")
    assert isinstance(researcher, Swarm)

    researcher_agent = researcher._participants[0]
    assert _handoff_targets(researcher_agent) == {"writer"}
    assert "writer" not in _tool_names(researcher_agent)
    assert "delegate_to_writer" not in _tool_names(researcher_agent)


def test_mixed_edges_from_same_source_split_correctly(tmp_project, tavily_env, provider_keys_env):
    """A source with one delegate edge and one handoff edge (to different
    destinations) must split them correctly: one becomes a delegate tool,
    the other a handoff -- neither mechanism swallows the other.
    """
    interactions_path = tmp_project / "interactions.yaml"
    data = yaml.safe_load(interactions_path.read_text())
    data["edges"] = [
        {"from": "coordinator", "to": "researcher", "type": "delegate"},
        {"from": "coordinator", "to": "writer", "type": "handoff"},
    ]
    interactions_path.write_text(yaml.safe_dump(data))
    project = commonadk.load(tmp_project)

    team = project.build("coordinator", target="autogen")

    assert isinstance(team, Swarm)  # coordinator DOES have a handoff edge now
    assert team._participant_names == ["coordinator", "writer"]  # researcher is NOT a participant

    coordinator = team._participants[0]
    assert _handoff_targets(coordinator) == {"writer"}
    assert "researcher" in {t.name for t in coordinator._tools}


def test_delegate_edge_bypasses_swarm_participant_dedup(tmp_project, tavily_env, provider_keys_env):
    """A `delegate` edge to a destination that is ALSO reachable via
    `handoff` from elsewhere does not join the Swarm's participant set at
    all -- it gets its own, entirely independent, second build wrapped in a
    delegate tool. Unlike `test_multi_parent_graph_builds_with_one_shared_
    participant` (an all-`handoff` graph, deduped to one shared Swarm
    participant), a `delegate` edge to the same destination is NOT deduped
    against the Swarm -- it is a structurally separate object graph (see
    module docstring, "Recursive construction").
    """
    interactions_path = tmp_project / "interactions.yaml"
    data = yaml.safe_load(interactions_path.read_text())
    data["edges"] = [
        {"from": "coordinator", "to": "researcher", "type": "handoff"},
        {"from": "researcher", "to": "writer", "type": "handoff"},
        {"from": "coordinator", "to": "writer", "type": "delegate"},
    ]
    interactions_path.write_text(yaml.safe_dump(data))
    project = commonadk.load(tmp_project)

    team = project.build("coordinator", target="autogen")

    assert isinstance(team, Swarm)
    assert team._participant_names == ["coordinator", "researcher", "writer"]

    coordinator = team._participants[0]
    assert _handoff_targets(coordinator) == {"researcher"}
    delegate_tool = next(t for t in coordinator._tools if t.name == "writer")
    assert isinstance(delegate_tool, AgentTool)
    # A second, independent `writer` AssistantAgent instance -- not the same
    # object as the Swarm's own `writer` participant.
    assert delegate_tool._agent is not team._participants[2]
    assert delegate_tool._agent.name == "writer"


def test_delegate_cycle_is_rejected(tmp_project, tavily_env, provider_keys_env):
    """A cycle closed entirely by `delegate` edges must raise a clear error
    at build time rather than recursing forever -- see module docstring,
    "Recursion and cycles".
    """
    interactions_path = tmp_project / "interactions.yaml"
    data = yaml.safe_load(interactions_path.read_text())
    data["edges"] = [
        {"from": "coordinator", "to": "researcher", "type": "delegate"},
        {"from": "researcher", "to": "coordinator", "type": "delegate"},
    ]
    interactions_path.write_text(yaml.safe_dump(data))
    project = commonadk.load(tmp_project)

    with pytest.raises(ValueError, match="cycle"):
        project.build("coordinator", target="autogen")


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

    Since issue #10 (delegate/handoff distinction): coordinator -delegate->
    researcher, and researcher itself hands off to writer, so researcher is
    reached through coordinator's `delegate_to_researcher` TeamTool's own
    Swarm, not `team._participants` directly (see
    `test_coordinator_build_happy_path_on_example` above).
    """
    project = commonadk.load(example_common_dir)
    coordinator = project.build("coordinator", target="autogen")
    delegate_tool = next(t for t in coordinator._tools if t.name == "delegate_to_researcher")
    researcher = delegate_tool._team._participants[0]

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


def test_anthropic_version_guard_allows_compatible_anthropic(
    tmp_project, tavily_env, provider_keys_env, monkeypatch
):
    """Verified-compatible `anthropic` versions (<1) must not block the
    build -- exercised explicitly (independent of whatever `anthropic`
    happens to be installed in the test environment) by faking the version
    the guard reads (see module docstring, "VERIFIED UPSTREAM
    INCOMPATIBILITY" and `_check_anthropic_temperature_compat`).
    """

    real_installed_version = autogen_adapter._installed_version

    def fake_installed_version(package: str) -> str | None:
        if package == "anthropic":
            return "0.122.0"
        return real_installed_version(package)

    monkeypatch.setattr(autogen_adapter, "_installed_version", fake_installed_version)

    writer_cfg = tmp_project / "writer" / "agent-config.yaml"
    data = yaml.safe_load(writer_cfg.read_text())
    data["model"] = "smart"  # -> anthropic/claude-sonnet-5
    writer_cfg.write_text(yaml.safe_dump(data))

    project = commonadk.load(tmp_project)
    agent = project.build("writer", target="autogen")  # must not raise

    assert isinstance(agent._model_client, AnthropicChatCompletionClient)


def test_anthropic_version_guard_raises_clear_error_for_incompatible_anthropic(
    tmp_project, tavily_env, provider_keys_env, monkeypatch
):
    """`anthropic>=1` removed `temperature` from `messages.create()`, which
    breaks autogen_ext's Anthropic client unconditionally (see module
    docstring, "VERIFIED UPSTREAM INCOMPATIBILITY" -- reproduced live in a
    real venv, not asserted from memory). This must surface as a clear
    commonadk `RuntimeError` naming both installed versions and the fix,
    never as the raw upstream `TypeError` a caller would otherwise only see
    at `run()` time. Faking the installed version here (rather than
    installing a real `anthropic>=1`) keeps this test offline and
    independent of what's actually installed -- the guard's own logic is
    what's under test, and it was verified for real against a live
    `anthropic==1.5.0` install (see the drafted repro in the scratchpad /
    PR description).
    """

    def fake_installed_version(package: str) -> str | None:
        if package == "anthropic":
            return "1.5.0"
        if package == "autogen-ext":
            return "0.7.5"
        return None

    monkeypatch.setattr(autogen_adapter, "_installed_version", fake_installed_version)

    writer_cfg = tmp_project / "writer" / "agent-config.yaml"
    data = yaml.safe_load(writer_cfg.read_text())
    data["model"] = "smart"  # -> anthropic/claude-sonnet-5
    writer_cfg.write_text(yaml.safe_dump(data))

    project = commonadk.load(tmp_project)
    with pytest.raises(RuntimeError) as exc_info:
        project.build("writer", target="autogen")

    message = str(exc_info.value)
    assert "anthropic==1.5.0" in message
    assert "autogen-ext==0.7.5" in message
    assert "temperature" in message
    assert "pip install 'anthropic<1'" in message
    assert "TypeError" not in type(exc_info.value).__name__


def test_anthropic_version_guard_skips_when_anthropic_not_installed(
    tmp_project, tavily_env, provider_keys_env, monkeypatch
):
    """No installed `anthropic` package at all -- the guard must not raise
    its own error (a different, clearer ImportError from `autogen_ext`
    itself would surface first in that real scenario); this only exercises
    the guard function's own early-return, not a real missing-package
    environment (this test file's module-level `pytest.importorskip`
    already guarantees `autogen_agentchat` -- and transitively `anthropic`,
    since `AnthropicChatCompletionClient` is imported at module scope here
    -- really is installed).
    """
    monkeypatch.setattr(autogen_adapter, "_installed_version", lambda package: None)

    autogen_adapter._check_anthropic_temperature_compat()  # must not raise


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


def test_openai_family_model_params_land_on_client(tmp_project, tavily_env, provider_keys_env):
    """Every key in `_OPENAI_MODEL_PARAM_MAP` must land on the built
    `OpenAIChatCompletionClient`'s `_create_args` (writer's base model, the
    `fast` alias, resolves to gemini/... which also routes through this
    client -- see module docstring, 'Model routing').
    """
    writer_cfg = tmp_project / "writer" / "agent-config.yaml"
    data = yaml.safe_load(writer_cfg.read_text())
    data["model_params"] = {
        "temperature": 0.3,
        "max_tokens": 2048,
        "top_p": 0.9,
        "stop": ["END"],
        "presence_penalty": 0.1,
        "frequency_penalty": 0.2,
        "seed": 42,
    }
    writer_cfg.write_text(yaml.safe_dump(data))

    project = commonadk.load(tmp_project)
    agent = project.build("writer", target="autogen")

    client = agent._model_client
    assert isinstance(client, OpenAIChatCompletionClient)
    args = client._create_args
    assert args["temperature"] == 0.3
    assert args["max_tokens"] == 2048
    assert args["top_p"] == 0.9
    assert args["stop"] == ["END"]
    assert args["presence_penalty"] == 0.1
    assert args["frequency_penalty"] == 0.2
    assert args["seed"] == 42


def test_openai_family_top_k_is_deliberately_unmapped(
    tmp_project, tavily_env, provider_keys_env
):
    """`top_k` isn't in `autogen_ext`'s own OpenAI-client whitelist
    (`create_kwargs`, see module docstring, 'model_params') -- must warn,
    not silently reach the client.
    """
    writer_cfg = tmp_project / "writer" / "agent-config.yaml"
    data = yaml.safe_load(writer_cfg.read_text())
    data["model_params"]["top_k"] = 40
    writer_cfg.write_text(yaml.safe_dump(data))

    project = commonadk.load(tmp_project)
    with pytest.warns(UserWarning, match="model_params key 'top_k'"):
        agent = project.build("writer", target="autogen")

    assert "top_k" not in agent._model_client._create_args


def test_anthropic_model_params_land_on_client(tmp_project, tavily_env, provider_keys_env):
    """Every key in `_ANTHROPIC_MODEL_PARAM_MAP` must land on the built
    `AnthropicChatCompletionClient`'s `_create_args`, including the
    `stop` -> `stop_sequences` rename (see module docstring, 'model_params').
    """
    writer_cfg = tmp_project / "writer" / "agent-config.yaml"
    data = yaml.safe_load(writer_cfg.read_text())
    data["model"] = "smart"  # -> anthropic/claude-sonnet-5
    data["model_params"] = {
        "temperature": 0.3,
        "max_tokens": 2048,
        "top_p": 0.9,
        "top_k": 40,
        "stop": ["END"],
    }
    writer_cfg.write_text(yaml.safe_dump(data))

    project = commonadk.load(tmp_project)
    agent = project.build("writer", target="autogen")

    client = agent._model_client
    assert isinstance(client, AnthropicChatCompletionClient)
    args = client._create_args
    assert args["temperature"] == 0.3
    assert args["max_tokens"] == 2048
    assert args["top_p"] == 0.9
    assert args["top_k"] == 40
    assert args["stop_sequences"] == ["END"]


@pytest.mark.parametrize(
    "unsupported_key", ["presence_penalty", "frequency_penalty", "seed"]
)
def test_anthropic_openai_only_keys_are_deliberately_unmapped(
    tmp_project, tavily_env, provider_keys_env, unsupported_key
):
    """`presence_penalty`/`frequency_penalty`/`seed` aren't in
    `autogen_ext`'s Anthropic-client whitelist (`anthropic_message_params`,
    see module docstring, 'model_params') -- passing them to
    `AnthropicChatCompletionClient` doesn't raise, it silently drops them
    (verified directly), which is exactly why this adapter must warn at
    `build()` time instead of letting them vanish unnoticed.
    """
    writer_cfg = tmp_project / "writer" / "agent-config.yaml"
    data = yaml.safe_load(writer_cfg.read_text())
    data["model"] = "smart"  # -> anthropic/claude-sonnet-5
    data["model_params"][unsupported_key] = 0.5
    writer_cfg.write_text(yaml.safe_dump(data))

    project = commonadk.load(tmp_project)
    with pytest.warns(UserWarning, match=f"model_params key '{unsupported_key}'"):
        agent = project.build("writer", target="autogen")

    assert unsupported_key not in agent._model_client._create_args


def test_unsupported_model_params_key_is_warned_and_ignored(tmp_project, tavily_env, provider_keys_env):
    writer_cfg = tmp_project / "writer" / "agent-config.yaml"
    data = yaml.safe_load(writer_cfg.read_text())
    # "stop_sequences" is a plausible-looking but wrong key -- this
    # adapter's canonical key is "stop" (see autogen_adapter.py's module
    # docstring, "model_params"); genuinely unsupported on both the
    # OpenAI-family and Anthropic param maps.
    data["model_params"]["stop_sequences"] = ["END"]
    writer_cfg.write_text(yaml.safe_dump(data))

    project = commonadk.load(tmp_project)
    with pytest.warns(UserWarning, match="model_params key 'stop_sequences'"):
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
