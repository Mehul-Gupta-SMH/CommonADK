"""Tests for the LangGraph adapter (M8).

Everything here is offline: constructing `CompiledStateGraph`/chat-model
objects touches no network. Unlike the Google ADK, OpenAI Agents, Claude
Agent SDK, and CrewAI adapters' tests -- and LIKE test_adapter_autogen.py --
this file needs fake provider API keys set up front (see
`provider_keys_env` below): `ChatGoogleGenerativeAI` and `ChatOpenAI`
construct their underlying provider client EAGERLY and raise immediately if
no key-shaped env var is discoverable; `ChatAnthropic` does not (see
langgraph_adapter.py's module docstring, "Model routing" / "Offline
construction") -- all three keys are set anyway for symmetry and so no test
here depends on that asymmetry by accident.

Introspecting model routing through `build()`'s returned `CompiledStateGraph`
is impractical: the chat model instance ends up buried inside a closure
several layers deep in `create_agent`'s own generated node function, with no
public accessor (verified directly -- unlike the private-but-reachable
attributes the other adapters' tests read, e.g. autogen's
`agent._model_client`). So the model-routing tests below call
`LangGraphAdapter()._model_for(project, spec)` directly instead -- still
exercising the adapter's real resolution logic, just one layer above the
opaque compiled graph, and documented here as a deliberate deviation from
this codebase's usual "inspect the built object" test style.

`pytest.importorskip` at module scope means this whole file is skipped, not
failed, when `langgraph`/`langchain` aren't installed -- the core suite must
stay green either way.
"""

from __future__ import annotations

import yaml
import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langchain")

import commonadk  # noqa: E402  (import after importorskip, deliberately)
from langchain_anthropic import ChatAnthropic  # noqa: E402
from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from langgraph.graph.state import CompiledStateGraph  # noqa: E402

from commonadk.adapters.langgraph_adapter import LangGraphAdapter  # noqa: E402


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
    """Fake, never-used API keys for every provider this adapter's chat
    models construct (eagerly, for two of the three -- see module
    docstring, "Offline construction"). Never sent anywhere in this test
    file -- construction alone is enough to trigger the underlying SDK's
    own presence check.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "AIza-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")


@pytest.fixture()
def multi_parent_project(tmp_project, tavily_env, provider_keys_env):
    """The example, with a `coordinator -> writer` delegate edge added back
    on top of its shipped `coordinator -> researcher -> writer` tree, so
    `writer` becomes reachable from two parents. Must BUILD SUCCESSFULLY --
    every reachable agent is built once into a flat `dict[str,
    CompiledStateGraph]` keyed by name, so a name reachable by two paths is
    simply the same node referenced by two different handoff tools.
    """
    interactions_path = tmp_project / "interactions.yaml"
    data = yaml.safe_load(interactions_path.read_text())
    data["edges"].append({"from": "coordinator", "to": "writer", "type": "delegate"})
    interactions_path.write_text(yaml.safe_dump(data))
    return commonadk.load(tmp_project)


@pytest.fixture()
def cyclic_project(tmp_project, tavily_env, provider_keys_env):
    """A cycle in the reachable graph (writer -> coordinator, closing the
    loop). Must BUILD SUCCESSFULLY -- a handoff tool targeting an already-
    registered node name is resolved by `Command(goto=..., graph=Command.
    PARENT)` at run time, not a construction-time hazard.
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
# helpers
# ---------------------------------------------------------------------------


def _tool_names(agent_graph: CompiledStateGraph) -> set[str]:
    """The tool names wired onto a single react-agent `CompiledStateGraph`
    (either the bare object `build()` returns for a leaf, or one entry of
    `graph.get_subgraphs()` for a multi-agent build) -- reaching into
    `ToolNode.tools_by_name` (a real public attribute of `ToolNode`, just
    nested under `create_agent`'s generated `RunnableSeq`), mirroring how
    other adapters' tests reach into SDK internals with no cleaner public
    accessor (e.g. autogen's `agent._handoffs`).
    """
    if "tools" not in agent_graph.nodes:
        return set()
    tool_node = agent_graph.nodes["tools"].node.steps[0]
    return set(tool_node.tools_by_name.keys())


def _subgraph(graph: CompiledStateGraph, name: str) -> CompiledStateGraph:
    return dict(graph.get_subgraphs())[name]


# ---------------------------------------------------------------------------
# graph construction / return-shape decision
# ---------------------------------------------------------------------------


def test_coordinator_build_happy_path_on_example(example_common_dir, tavily_env, provider_keys_env):
    """The shipped research-crew example -- coordinator -delegate->
    researcher -handoff-> writer -- has an outgoing edge at the build root,
    so this must build a compiled multi-agent StateGraph with one node per
    reachable agent, each carrying exactly the handoff tool(s) its own
    outgoing edges call for (see module docstring, "Edge mapping" -- LangGraph
    is the one target that honors edge *targets* precisely).
    """
    project = commonadk.load(example_common_dir)
    graph = project.build("coordinator", target="langgraph")

    assert isinstance(graph, CompiledStateGraph)
    assert set(graph.nodes.keys()) == {"__start__", "coordinator", "researcher", "writer"}

    coordinator = _subgraph(graph, "coordinator")
    researcher = _subgraph(graph, "researcher")
    writer = _subgraph(graph, "writer")

    assert _tool_names(coordinator) == {
        "split_into_subtopics",
        "format_handoff_note",
        "transfer_to_researcher",
    }
    assert _tool_names(researcher) == {
        "search_web",
        "fetch_page",
        "transfer_to_writer",  # the deep edge
    }
    assert _tool_names(writer) == {"count_words", "format_as_markdown"}  # no outgoing edges


def test_researcher_build_wires_writer_as_only_other_node(
    example_common_dir, tavily_env, provider_keys_env
):
    """Building from `researcher` directly (not through `coordinator`)
    yields a graph with only `researcher` and `writer` as nodes."""
    project = commonadk.load(example_common_dir)
    graph = project.build("researcher", target="langgraph")

    assert isinstance(graph, CompiledStateGraph)
    assert set(graph.nodes.keys()) == {"__start__", "researcher", "writer"}
    assert _tool_names(_subgraph(graph, "researcher")) == {
        "search_web",
        "fetch_page",
        "transfer_to_writer",
    }


def test_writer_build_returns_bare_react_agent_not_a_multi_agent_graph(
    example_common_dir, tavily_env, provider_keys_env
):
    """`writer` has no outgoing edges -- nothing to hand off to, so this must
    return its own compiled react agent directly, not wrapped in a parent
    StateGraph (see module docstring, "WHAT build() RETURNS").
    """
    project = commonadk.load(example_common_dir)
    graph = project.build("writer", target="langgraph")

    assert isinstance(graph, CompiledStateGraph)
    assert set(graph.nodes.keys()) == {"__start__", "model", "tools"}  # no coordinator/researcher
    assert _tool_names(graph) == {"count_words", "format_as_markdown"}


def test_multi_parent_graph_builds_with_one_shared_node(multi_parent_project):
    """KEY PROPERTY: a multi-parent graph builds successfully, `writer`
    appears exactly once as a node, and `coordinator` gets a handoff tool
    for EACH of its two distinct outgoing edges.
    """
    graph = multi_parent_project.build("coordinator", target="langgraph")

    assert set(graph.nodes.keys()) == {"__start__", "coordinator", "researcher", "writer"}
    coordinator = _subgraph(graph, "coordinator")
    assert _tool_names(coordinator) >= {"transfer_to_researcher", "transfer_to_writer"}


def test_cyclic_graph_builds_without_recursion_hazard(cyclic_project):
    """KEY PROPERTY: a cycle back to the build root builds successfully --
    `writer` gets a `transfer_to_coordinator` handoff tool targeting a node
    that already exists in the same StateGraph.
    """
    graph = cyclic_project.build("coordinator", target="langgraph")

    assert set(graph.nodes.keys()) == {"__start__", "coordinator", "researcher", "writer"}
    writer = _subgraph(graph, "writer")
    assert "transfer_to_coordinator" in _tool_names(writer)


def test_duplicate_edges_to_same_destination_yield_one_handoff_tool(tmp_project, tavily_env, provider_keys_env):
    """Two edges from the same source to the same destination (one delegate,
    one handoff -- both map to the same mechanism in v1) must still produce
    exactly one `transfer_to_<dest>` tool, not a duplicate-named one.
    """
    interactions_path = tmp_project / "interactions.yaml"
    data = yaml.safe_load(interactions_path.read_text())
    data["edges"].append({"from": "coordinator", "to": "researcher", "type": "handoff"})
    interactions_path.write_text(yaml.safe_dump(data))
    project = commonadk.load(tmp_project)

    graph = project.build("coordinator", target="langgraph")
    coordinator = _subgraph(graph, "coordinator")
    tool_names = [
        t
        for t in coordinator.nodes["tools"].node.steps[0].tools_by_name
        if t.startswith("transfer_to_")
    ]
    assert tool_names == ["transfer_to_researcher"]  # exactly one, not two


# ---------------------------------------------------------------------------
# model routing
#
# Model routing is exercised through `LangGraphAdapter()._model_for(...)`
# directly rather than through `build()`'s returned graph -- see module
# docstring for why the compiled graph doesn't expose this cleanly.
# ---------------------------------------------------------------------------


def test_gemini_model_routes_to_google_genai(example_common_dir, provider_keys_env):
    """researcher's model (gemini/gemini-2.5-pro in the shipped example,
    unmodified -- no override needed) must route through
    `ChatGoogleGenerativeAI` with the bare model id and model_params passed
    straight through.
    """
    project = commonadk.load(example_common_dir)
    model = LangGraphAdapter()._model_for(project, project.agents["researcher"])

    assert isinstance(model, ChatGoogleGenerativeAI)
    assert model.model == "gemini-2.5-pro"
    assert model.temperature == 0.2
    assert model.max_output_tokens == 4096  # model_params max_tokens -> max_output_tokens


def test_openai_model_routes_to_openai(tmp_project, provider_keys_env):
    """An agent configured with an `openai/...` model must route through
    `ChatOpenAI` with the bare model id."""
    writer_cfg = tmp_project / "writer" / "agent-config.yaml"
    data = yaml.safe_load(writer_cfg.read_text())
    data["model"] = "openai/gpt-4o"
    writer_cfg.write_text(yaml.safe_dump(data))
    project = commonadk.load(tmp_project)

    model = LangGraphAdapter()._model_for(project, project.agents["writer"])

    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "gpt-4o"


def test_anthropic_model_routes_to_anthropic(tmp_project, provider_keys_env):
    """writer's base model is the `fast` alias; overriding its *base* model
    (not a per-target override) to the `smart` alias (-> anthropic/claude-
    sonnet-5) must route through `ChatAnthropic`.
    """
    writer_cfg = tmp_project / "writer" / "agent-config.yaml"
    data = yaml.safe_load(writer_cfg.read_text())
    data["model"] = "smart"
    writer_cfg.write_text(yaml.safe_dump(data))
    project = commonadk.load(tmp_project)

    model = LangGraphAdapter()._model_for(project, project.agents["writer"])

    assert isinstance(model, ChatAnthropic)
    assert model.model == "claude-sonnet-5"


def test_unsupported_provider_raises_clear_error(tmp_project, provider_keys_env):
    """A provider this adapter ships no langchain integration package for
    (no litellm fallback here, unlike CrewAI) must raise a clear, actionable
    error naming the agent and its resolved model string.
    """
    writer_cfg = tmp_project / "writer" / "agent-config.yaml"
    data = yaml.safe_load(writer_cfg.read_text())
    data["model"] = "cohere/command-r"
    writer_cfg.write_text(yaml.safe_dump(data))
    project = commonadk.load(tmp_project)

    with pytest.raises(ValueError) as exc_info:
        LangGraphAdapter()._model_for(project, project.agents["writer"])

    message = str(exc_info.value)
    assert "writer" in message
    assert "cohere/command-r" in message
    assert "targets.langgraph.model" in message


def test_per_target_override_wins(tmp_project, provider_keys_env):
    """writer's base model is the `fast` alias (-> gemini/gemini-2.5-flash),
    but its `targets.langgraph.model` override must win and is passed
    through as-is (already langchain-native "provider:model" form) -- see
    module docstring, "Per-target override".
    """
    writer_cfg = tmp_project / "writer" / "agent-config.yaml"
    data = yaml.safe_load(writer_cfg.read_text())
    data["targets"]["langgraph"] = {"model": "openai:gpt-4o-mini"}
    writer_cfg.write_text(yaml.safe_dump(data))
    project = commonadk.load(tmp_project)

    model = LangGraphAdapter()._model_for(project, project.agents["writer"])

    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "gpt-4o-mini"


def test_gemini_full_candidate_set_lands_on_chat_model(tmp_project, provider_keys_env):
    """`google_genai` is the one provider here that genuinely accepts every
    candidate key (see module docstring, "model_params") -- all of them must
    land on the built `ChatGoogleGenerativeAI`.
    """
    writer_cfg = tmp_project / "writer" / "agent-config.yaml"
    data = yaml.safe_load(writer_cfg.read_text())
    data["model_params"] = {
        "temperature": 0.3,
        "max_tokens": 2048,
        "top_p": 0.9,
        "top_k": 40,
        "stop": ["END"],
        "presence_penalty": 0.1,
        "frequency_penalty": 0.2,
        "seed": 42,
    }
    writer_cfg.write_text(yaml.safe_dump(data))
    project = commonadk.load(tmp_project)

    model = LangGraphAdapter()._model_for(project, project.agents["writer"])

    assert isinstance(model, ChatGoogleGenerativeAI)
    assert model.temperature == 0.3
    assert model.max_output_tokens == 2048
    assert model.top_p == 0.9
    assert model.top_k == 40
    assert model.stop == ["END"]
    assert model.presence_penalty == 0.1
    assert model.frequency_penalty == 0.2
    assert model.seed == 42


def test_openai_provider_top_k_is_deliberately_unmapped(tmp_project, provider_keys_env):
    """`ChatOpenAI` has no `top_k` field (see module docstring,
    "model_params") -- must warn, not silently land in `model_kwargs`.
    """
    writer_cfg = tmp_project / "writer" / "agent-config.yaml"
    data = yaml.safe_load(writer_cfg.read_text())
    data["model"] = "openai/gpt-4o"
    data["model_params"] = {
        "temperature": 0.3,
        "top_p": 0.9,
        "stop": ["END"],
        "presence_penalty": 0.1,
        "frequency_penalty": 0.2,
        "seed": 42,
        "top_k": 40,
    }
    writer_cfg.write_text(yaml.safe_dump(data))
    project = commonadk.load(tmp_project)

    with pytest.warns(UserWarning, match="model_params key 'top_k'"):
        model = LangGraphAdapter()._model_for(project, project.agents["writer"])

    assert isinstance(model, ChatOpenAI)
    assert model.temperature == 0.3
    assert model.top_p == 0.9
    assert model.stop == ["END"]
    assert model.presence_penalty == 0.1
    assert model.frequency_penalty == 0.2
    assert model.seed == 42
    # top_k must never have reached the client at all -- not even stashed
    # in model_kwargs (see module docstring: unmapped kwargs silently land
    # there and get forwarded to the real API, which is exactly what the
    # warn-and-ignore policy exists to prevent).
    assert "top_k" not in (model.model_kwargs or {})


def test_anthropic_provider_openai_only_keys_are_deliberately_unmapped(
    tmp_project, provider_keys_env
):
    """`ChatAnthropic` has no `presence_penalty`/`frequency_penalty`/`seed`
    fields (see module docstring, "model_params") -- each must warn, and
    `top_p`/`top_k`/`stop` (which it DOES support) must still land.
    """
    writer_cfg = tmp_project / "writer" / "agent-config.yaml"
    data = yaml.safe_load(writer_cfg.read_text())
    data["model"] = "smart"  # -> anthropic/claude-sonnet-5
    data["model_params"] = {
        "temperature": 0.3,
        "top_p": 0.9,
        "top_k": 40,
        "stop": ["END"],
        "presence_penalty": 0.1,
        "frequency_penalty": 0.2,
        "seed": 42,
    }
    writer_cfg.write_text(yaml.safe_dump(data))
    project = commonadk.load(tmp_project)

    with pytest.warns(UserWarning) as records:
        model = LangGraphAdapter()._model_for(project, project.agents["writer"])

    warned_keys = {
        str(r.message).split("model_params key '")[1].split("'")[0] for r in records
    }
    assert warned_keys == {"presence_penalty", "frequency_penalty", "seed"}

    assert isinstance(model, ChatAnthropic)
    assert model.temperature == 0.3
    assert model.top_p == 0.9
    assert model.top_k == 40
    assert model.stop_sequences == ["END"]
    assert "presence_penalty" not in (model.model_kwargs or {})
    assert "frequency_penalty" not in (model.model_kwargs or {})
    assert "seed" not in (model.model_kwargs or {})


def test_override_to_known_provider_uses_that_providers_full_map(
    tmp_project, provider_keys_env
):
    """A `targets.langgraph.model` override whose provider prefix IS one of
    this adapter's three known providers (here, `anthropic:...`) must use
    THAT provider's own param map, not the conservative default (see module
    docstring, "model_params" / "Per-target override") -- `top_k` must land.
    """
    writer_cfg = tmp_project / "writer" / "agent-config.yaml"
    data = yaml.safe_load(writer_cfg.read_text())
    data["targets"]["langgraph"] = {"model": "anthropic:claude-opus-5"}
    data["model_params"] = {"top_k": 40}
    writer_cfg.write_text(yaml.safe_dump(data))
    project = commonadk.load(tmp_project)

    import warnings as _warnings

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        model = LangGraphAdapter()._model_for(project, project.agents["writer"])

    assert not caught  # top_k must NOT be warned-and-ignored for this override
    assert isinstance(model, ChatAnthropic)
    assert model.top_k == 40


def test_override_to_unknown_provider_only_maps_the_conservative_default(
    tmp_project, provider_keys_env
):
    """A `targets.langgraph.model` override to a provider prefix outside
    this adapter's known three must fall back to `_DEFAULT_MODEL_PARAM_MAP`
    (temperature/max_tokens only) -- `top_p` must warn even though the
    underlying `init_chat_model` call might otherwise accept it, since this
    adapter has never verified that for an arbitrary provider (see module
    docstring, "model_params").
    """
    writer_cfg = tmp_project / "writer" / "agent-config.yaml"
    data = yaml.safe_load(writer_cfg.read_text())
    # "fake" isn't openai/anthropic/google_genai -- init_chat_model would
    # reject it outright once it tries to actually resolve a client, but
    # `_model_param_kwargs` is computed before that happens, so this proves
    # the param-map fallback runs first.
    data["targets"]["langgraph"] = {"model": "fake:some-model"}
    data["model_params"] = {"temperature": 0.3, "top_p": 0.9}
    writer_cfg.write_text(yaml.safe_dump(data))
    project = commonadk.load(tmp_project)

    with pytest.warns(UserWarning, match="model_params key 'top_p'"):
        with pytest.raises(Exception):
            # The bogus provider itself is rejected by init_chat_model --
            # only the *kwargs it was called with* are under test here, via
            # the warning fired before that call.
            LangGraphAdapter()._model_for(project, project.agents["writer"])


def test_unsupported_model_params_key_is_warned_and_ignored(tmp_project, tavily_env, provider_keys_env):
    writer_cfg = tmp_project / "writer" / "agent-config.yaml"
    data = yaml.safe_load(writer_cfg.read_text())
    # "stop_sequences" is a plausible-looking but wrong key -- every
    # provider map here uses the canonical key "stop" instead (see
    # langgraph_adapter.py's module docstring, "model_params"); writer's
    # provider (gemini/google_genai) supports every other candidate key, so
    # this is the one still-genuinely-unsupported key left to exercise the
    # warn-and-ignore path for it.
    data["model_params"]["stop_sequences"] = ["END"]
    writer_cfg.write_text(yaml.safe_dump(data))
    project = commonadk.load(tmp_project)

    with pytest.warns(UserWarning, match="model_params key 'stop_sequences'"):
        graph = project.build("writer", target="langgraph")

    assert graph is not None  # build still succeeds


# ---------------------------------------------------------------------------
# leaf build shape (no tools, no handoffs)
# ---------------------------------------------------------------------------


def test_leaf_build_with_no_tools_or_handoffs(tmp_project, provider_keys_env):
    """An agent with neither tools nor outgoing edges still builds a valid,
    directly runnable react agent (an empty `tools=[]` is a fine input to
    `create_agent`).
    """
    writer_cfg = tmp_project / "writer" / "agent-config.yaml"
    data = yaml.safe_load(writer_cfg.read_text())
    data["tools"] = []
    writer_cfg.write_text(yaml.safe_dump(data))
    # writer.py must still define no *referenced* tools for validation to pass;
    # dropping the tools list to empty is enough since validation only checks
    # that referenced tools exist, not that all defined tools are referenced.
    project = commonadk.load(tmp_project)

    graph = project.build("writer", target="langgraph")

    assert isinstance(graph, CompiledStateGraph)
    assert "tools" not in graph.nodes  # no ToolNode at all with zero tools
    assert set(graph.nodes.keys()) == {"__start__", "model"}


# ---------------------------------------------------------------------------
# env preflight
# ---------------------------------------------------------------------------


def test_missing_required_env_var_blocks_build(example_common_dir, monkeypatch, provider_keys_env):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("POSTGRES_DSN", raising=False)
    project = commonadk.load(example_common_dir)

    with pytest.raises(OSError) as exc_info:
        project.build("researcher", target="langgraph")

    message = str(exc_info.value)
    assert "TAVILY_API_KEY" in message
    assert "Search API key used by search_web" in message
    # POSTGRES_DSN is declared `required: false` -- its absence must not be
    # reported as a blocking problem.
    assert "POSTGRES_DSN" not in message


def test_optional_env_var_absence_does_not_block(example_common_dir, tavily_env, provider_keys_env):
    project = commonadk.load(example_common_dir)

    graph = project.build("researcher", target="langgraph")  # must not raise
    assert isinstance(graph, CompiledStateGraph)


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
        project.build("coordinator", target="langgraph")


# ---------------------------------------------------------------------------
# the cross-target hypothesis test (plan.md "v1 success criterion") lives in
# tests/test_hypothesis.py, parametrized over every SDK target.
# ---------------------------------------------------------------------------
