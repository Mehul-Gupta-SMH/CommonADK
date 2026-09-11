"""Tests for the AutoGen and LangGraph runners (issue #22, follow-up to the
Google ADK / OpenAI Agents runners in `tests/test_runners.py`).

Kept in a separate file, per the task this was implemented against, so it
never collides with another agent's edits to `tests/test_runners.py` (which
covers the `claude`/`crewai` runners in parallel). Same idiom as that file:
everything here is offline, `pytest.importorskip` per-file (not just
per-test, since this whole file needs both `autogen_agentchat` and
`langgraph`/`langchain` to mean anything) skips it rather than failing it
when either SDK is missing, and every normalization test feeds each runner
*real, constructed SDK objects* (`autogen_core.FunctionCall`,
`langchain_core.messages.AIMessage`, etc.) through a monkeypatched native
run/stream call -- never a fabricated stand-in class, never a network call,
never an API key.

Building each runner's `project.build(...)` object for real (not mocked) is
deliberate, mirroring `tests/test_adapter_autogen.py` /
`tests/test_adapter_langgraph.py`'s own fixtures: both adapters construct
their model clients EAGERLY (see each adapter module's own docstring,
"Offline construction"), so `provider_keys_env` below sets fake,
never-used API keys purely to satisfy that construction-time check -- no
network call is made either way, and this file's runner-level tests only
ever monkeypatch the *run* surface (`Swarm.run_stream` /
`CompiledStateGraph.astream`), never the model client itself.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("autogen_agentchat")
pytest.importorskip("langgraph")
pytest.importorskip("langchain")

from commonadk import load  # noqa: E402
from commonadk.runners import RunSession, get_runner, known_targets, known_unported_targets  # noqa: E402
from commonadk.runners.events import (  # noqa: E402
    AgentFinished,
    AgentStarted,
    LLMCall,
    RunError,
    RunFinished,
    ToolCall,
    Transfer,
)


@pytest.fixture()
def tavily_env(monkeypatch):
    """Satisfy researcher's one *required* env var -- mirrors every other
    test file in this repo that builds against the shipped example."""
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.delenv("POSTGRES_DSN", raising=False)


@pytest.fixture()
def provider_keys_env(monkeypatch):
    """Fake, never-used API keys for every provider either adapter's model
    clients construct eagerly (see autogen_adapter.py / langgraph_adapter.py,
    "Offline construction"). Never sent anywhere -- construction alone is
    enough to trigger each SDK's own presence check."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test")
    monkeypatch.setenv("GOOGLE_API_KEY", "AIza-test")


# ---------------------------------------------------------------------------
# registry: both targets now ported
# ---------------------------------------------------------------------------


def test_autogen_and_langgraph_are_registered_and_no_longer_unported():
    assert "autogen" in known_targets()
    assert "langgraph" in known_targets()
    assert "autogen" not in known_unported_targets()
    assert "langgraph" not in known_unported_targets()


def test_get_runner_returns_the_right_class_for_each_target():
    from commonadk.runners.autogen import AutoGenRunner
    from commonadk.runners.langgraph import LangGraphRunner

    assert isinstance(get_runner("autogen"), AutoGenRunner)
    assert isinstance(get_runner("langgraph"), LangGraphRunner)


# ---------------------------------------------------------------------------
# AutoGen normalization -- real autogen_core.FunctionCall / RequestUsage /
# autogen_agentchat.messages.* objects
# ---------------------------------------------------------------------------


def test_autogen_runner_normalizes_tool_call_and_handoff(
    tmp_project, tavily_env, provider_keys_env, monkeypatch
):
    from autogen_agentchat.base import TaskResult
    from autogen_agentchat.messages import (
        HandoffMessage,
        TextMessage,
        ToolCallExecutionEvent,
        ToolCallRequestEvent,
    )
    from autogen_agentchat.teams import Swarm
    from autogen_core import FunctionCall
    from autogen_core.models import FunctionExecutionResult, RequestUsage

    from commonadk.runners.autogen import AutoGenRunner

    tool_request = ToolCallRequestEvent(
        source="coordinator",
        content=[FunctionCall(id="call1", name="search_web", arguments=json.dumps({"query": "ev"}))],
        models_usage=RequestUsage(prompt_tokens=10, completion_tokens=5),
    )
    tool_result = ToolCallExecutionEvent(
        source="coordinator",
        content=[
            FunctionExecutionResult(content="search results here", name="search_web", call_id="call1", is_error=False)
        ],
    )
    handoff = HandoffMessage(source="coordinator", target="researcher", content="handing off")
    final_answer = TextMessage(
        source="researcher",
        content="Final answer text",
        models_usage=RequestUsage(prompt_tokens=20, completion_tokens=8),
    )
    stream_items = [tool_request, tool_result, handoff, final_answer]

    async def fake_run_stream(self, *, task=None, cancellation_token=None, output_task_messages=True):
        for item in stream_items:
            yield item
        yield TaskResult(messages=[final_answer])

    monkeypatch.setattr(Swarm, "run_stream", fake_run_stream)

    # This test drives a `Swarm`, so the project must actually build one.
    # Since issue #10 the adapter only wraps the build root in a `Swarm`
    # when it has an outgoing *handoff* edge; the shipped example's
    # coordinator edge is `delegate`, which now builds a bare
    # `AssistantAgent` instead. Patching `Swarm.run_stream` would then
    # silently miss, and the runner would call the real AssistantAgent --
    # i.e. a live API request from an offline test. Flip the edge so the
    # build shape matches what this test is actually about.
    interactions = tmp_project / "interactions.yaml"
    interactions.write_text(
        interactions.read_text().replace("type: delegate", "type: handoff")
    )

    project = load(str(tmp_project))
    runner = AutoGenRunner()
    trace = runner.run_sync(project, "coordinator", "research EV adoption")

    tool_call = next(e for e in trace.events if isinstance(e, ToolCall))
    assert tool_call.agent_name == "coordinator"
    assert tool_call.tool_name == "search_web"
    assert tool_call.arguments == {"query": "ev"}
    assert tool_call.result_summary == "search results here"
    assert tool_call.error is None
    # The handoff is reported ONLY as a Transfer, never also as a ToolCall.
    assert sum(1 for e in trace.events if isinstance(e, ToolCall)) == 1

    transfer = next(e for e in trace.events if isinstance(e, Transfer))
    assert transfer.from_agent == "coordinator"
    assert transfer.to_agent == "researcher"
    assert transfer.transfer_kind == "autogen:handoff"

    llm_calls = [e for e in trace.events if isinstance(e, LLMCall)]
    assert [c.agent_name for c in llm_calls] == ["coordinator", "researcher"]
    assert [c.total_tokens for c in llm_calls] == [15, 28]
    assert all(c.model is not None for c in llm_calls)
    assert all(c.cost_usd is not None for c in llm_calls)  # both gemini models are priced

    agent_started = [e.agent_name for e in trace.events if isinstance(e, AgentStarted)]
    agent_finished = [e.agent_name for e in trace.events if isinstance(e, AgentFinished)]
    assert agent_started == ["coordinator", "researcher"]
    assert agent_finished == ["coordinator", "researcher"]

    finished = trace.events[-1]
    assert isinstance(finished, RunFinished)
    assert finished.final_text == "Final answer text"
    assert finished.total_tokens == 43
    assert finished.usage_complete is True


def test_autogen_runner_all_zero_usage_becomes_none_not_zero(
    tmp_project, provider_keys_env, monkeypatch
):
    """A `RequestUsage(prompt_tokens=0, completion_tokens=0)` -- the exact
    shape `autogen_ext`'s OpenAI-family client defaults to when the
    provider's own response carried no usage at all (verified in
    runners/autogen.py's module docstring) -- must become `None` on every
    `LLMCall` token/cost field, never a confident zero."""
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.base import TaskResult
    from autogen_agentchat.messages import TextMessage
    from autogen_core.models import RequestUsage

    from commonadk.runners.autogen import AutoGenRunner

    message = TextMessage(
        source="writer", content="hi", models_usage=RequestUsage(prompt_tokens=0, completion_tokens=0)
    )

    async def fake_run_stream(self, *, task=None, cancellation_token=None, output_task_messages=True):
        yield message
        yield TaskResult(messages=[message])

    monkeypatch.setattr(AssistantAgent, "run_stream", fake_run_stream)

    project = load(str(tmp_project))
    runner = AutoGenRunner()
    trace = runner.run_sync(project, "writer", "hi")

    llm_call = next(e for e in trace.events if isinstance(e, LLMCall))
    assert llm_call.prompt_tokens is None
    assert llm_call.completion_tokens is None
    assert llm_call.total_tokens is None
    assert llm_call.cost_usd is None
    # `model` is still attached -- it's descriptive metadata, independent of
    # whether that call's usage happened to be reported.
    assert llm_call.model is not None

    finished = trace.events[-1]
    assert isinstance(finished, RunFinished)
    assert finished.usage_complete is False
    assert finished.total_tokens is None


def test_autogen_runner_emits_run_error_and_reraises_on_exception(
    tmp_project, provider_keys_env, monkeypatch
):
    from autogen_agentchat.agents import AssistantAgent

    from commonadk.runners.autogen import AutoGenRunner

    async def fake_run_stream(self, *, task=None, cancellation_token=None, output_task_messages=True):
        raise RuntimeError("simulated SDK failure")
        yield  # pragma: no cover -- makes this an async generator

    monkeypatch.setattr(AssistantAgent, "run_stream", fake_run_stream)

    project = load(str(tmp_project))
    runner = AutoGenRunner()

    with pytest.raises(RuntimeError, match="simulated SDK failure"):
        runner.run_sync(project, "writer", "hi")


def test_autogen_runner_session_reuses_same_built_object_across_turns(
    tmp_project, provider_keys_env, monkeypatch
):
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.base import TaskResult

    from commonadk.runners.autogen import AutoGenRunner

    seen_instances: list[int] = []

    async def fake_run_stream(self, *, task=None, cancellation_token=None, output_task_messages=True):
        seen_instances.append(id(self))
        yield TaskResult(messages=[])

    monkeypatch.setattr(AssistantAgent, "run_stream", fake_run_stream)

    project = load(str(tmp_project))
    runner = AutoGenRunner()
    session = RunSession()

    runner.run_sync(project, "writer", "turn 1", session=session)
    runner.run_sync(project, "writer", "turn 2", session=session)

    assert len(seen_instances) == 2
    assert seen_instances[0] == seen_instances[1]  # same AssistantAgent both turns
    assert session.turns == 2
    assert session.native["autogen"]["built"] is not None


# ---------------------------------------------------------------------------
# LangGraph normalization -- real langchain_core.messages.* objects, driven
# through a monkeypatched CompiledStateGraph.astream
# ---------------------------------------------------------------------------


def test_langgraph_runner_normalizes_tool_call_llm_usage_and_handoff(
    tmp_project, tavily_env, provider_keys_env, monkeypatch
):
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langchain_core.messages.ai import UsageMetadata
    from langgraph.graph.state import CompiledStateGraph

    from commonadk.runners.langgraph import LangGraphRunner

    human = HumanMessage(id="h1", content="go")
    ai_tool_call = AIMessage(
        id="ai1",
        name="coordinator",
        content="",
        tool_calls=[{"name": "search_web", "args": {"query": "ev"}, "id": "call1"}],
        usage_metadata=UsageMetadata(input_tokens=10, output_tokens=5, total_tokens=15),
    )
    tool_result = ToolMessage(id="tm1", content="search results here", name="search_web", tool_call_id="call1")
    ai_handoff_call = AIMessage(
        id="ai2",
        name="coordinator",
        content="",
        tool_calls=[{"name": "transfer_to_researcher", "args": {}, "id": "call2"}],
        usage_metadata=UsageMetadata(input_tokens=7, output_tokens=3, total_tokens=10),
    )
    handoff_result = ToolMessage(
        id="tm2", content="Successfully transferred to researcher", name="transfer_to_researcher", tool_call_id="call2"
    )
    final_answer = AIMessage(
        id="ai3",
        name="researcher",
        content="Final answer text",
        usage_metadata=UsageMetadata(input_tokens=20, output_tokens=8, total_tokens=28),
    )

    coordinator_ns = ("coordinator:uuid1",)
    researcher_ns = ("researcher:uuid2",)

    chunks = [
        ((), {"messages": [human]}),
        (coordinator_ns, {"messages": [human]}),
        (coordinator_ns, {"messages": [human, ai_tool_call]}),
        (coordinator_ns, {"messages": [human, ai_tool_call, tool_result]}),
        (coordinator_ns, {"messages": [human, ai_tool_call, tool_result, ai_handoff_call]}),
        ((), {"messages": [human, ai_tool_call, tool_result, ai_handoff_call, handoff_result]}),
        (researcher_ns, {"messages": [human, ai_tool_call, tool_result, ai_handoff_call, handoff_result]}),
        (
            researcher_ns,
            {"messages": [human, ai_tool_call, tool_result, ai_handoff_call, handoff_result, final_answer]},
        ),
        (
            (),
            {"messages": [human, ai_tool_call, tool_result, ai_handoff_call, handoff_result, final_answer]},
        ),
    ]

    async def fake_astream(self, input, *, config=None, stream_mode=None, subgraphs=False, **kwargs):
        for namespace, values in chunks:
            yield namespace, values

    monkeypatch.setattr(CompiledStateGraph, "astream", fake_astream)

    project = load(str(tmp_project))
    runner = LangGraphRunner()
    trace = runner.run_sync(project, "coordinator", "research EV adoption")

    tool_call = next(e for e in trace.events if isinstance(e, ToolCall))
    assert tool_call.agent_name == "coordinator"
    assert tool_call.tool_name == "search_web"
    assert tool_call.arguments == {"query": "ev"}
    assert tool_call.result_summary == "search results here"
    assert tool_call.error is None
    # The handoff tool call must be reported ONLY as a Transfer.
    assert sum(1 for e in trace.events if isinstance(e, ToolCall)) == 1

    transfer = next(e for e in trace.events if isinstance(e, Transfer))
    assert transfer.from_agent == "coordinator"
    assert transfer.to_agent == "researcher"
    assert transfer.transfer_kind == "langgraph:command_handoff"

    llm_calls = [e for e in trace.events if isinstance(e, LLMCall)]
    assert [c.agent_name for c in llm_calls] == ["coordinator", "coordinator", "researcher"]
    assert [c.total_tokens for c in llm_calls] == [15, 10, 28]
    assert all(c.model is not None for c in llm_calls)
    assert all(c.cost_usd is not None for c in llm_calls)  # both gemini models are priced

    agent_started = [e.agent_name for e in trace.events if isinstance(e, AgentStarted)]
    agent_finished = [e.agent_name for e in trace.events if isinstance(e, AgentFinished)]
    assert agent_started == ["coordinator", "researcher"]
    assert agent_finished == ["coordinator", "researcher"]

    finished = trace.events[-1]
    assert isinstance(finished, RunFinished)
    assert finished.final_text == "Final answer text"
    assert finished.total_tokens == 53
    assert finished.usage_complete is True


def test_langgraph_runner_all_zero_usage_becomes_none_not_zero(
    tmp_project, provider_keys_env, monkeypatch
):
    """A `UsageMetadata` with every token field at `0` -- the shape
    `langchain_openai` can produce when a present-but-incomplete usage dict
    is missing a field (see runners/langgraph.py's module docstring) --
    must become `None` on every `LLMCall` token/cost field."""
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.messages.ai import UsageMetadata
    from langgraph.graph.state import CompiledStateGraph

    from commonadk.runners.langgraph import LangGraphRunner

    human = HumanMessage(id="h1", content="go")
    answer = AIMessage(
        id="ai1",
        name="writer",
        content="hi",
        usage_metadata=UsageMetadata(input_tokens=0, output_tokens=0, total_tokens=0),
    )

    async def fake_astream(self, input, *, config=None, stream_mode=None, subgraphs=False, **kwargs):
        yield (), {"messages": [human]}
        yield (), {"messages": [human, answer]}

    monkeypatch.setattr(CompiledStateGraph, "astream", fake_astream)

    project = load(str(tmp_project))
    runner = LangGraphRunner()
    trace = runner.run_sync(project, "writer", "hi")

    llm_call = next(e for e in trace.events if isinstance(e, LLMCall))
    assert llm_call.prompt_tokens is None
    assert llm_call.completion_tokens is None
    assert llm_call.total_tokens is None
    assert llm_call.cost_usd is None
    assert llm_call.model is not None

    finished = trace.events[-1]
    assert isinstance(finished, RunFinished)
    assert finished.usage_complete is False
    assert finished.total_tokens is None
    assert finished.final_text == "hi"


def test_langgraph_runner_emits_run_error_and_reraises_on_exception(
    tmp_project, provider_keys_env, monkeypatch
):
    from langgraph.graph.state import CompiledStateGraph

    from commonadk.runners.langgraph import LangGraphRunner

    async def fake_astream(self, input, *, config=None, stream_mode=None, subgraphs=False, **kwargs):
        raise RuntimeError("simulated SDK failure")
        yield  # pragma: no cover -- makes this an async generator

    monkeypatch.setattr(CompiledStateGraph, "astream", fake_astream)

    project = load(str(tmp_project))
    runner = LangGraphRunner()

    with pytest.raises(RuntimeError, match="simulated SDK failure"):
        runner.run_sync(project, "writer", "hi")


def test_langgraph_runner_session_accumulates_history_across_turns(
    tmp_project, provider_keys_env, monkeypatch
):
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.graph.state import CompiledStateGraph

    from commonadk.runners.langgraph import LangGraphRunner

    seen_input_lengths: list[int] = []

    async def fake_astream(self, input, *, config=None, stream_mode=None, subgraphs=False, **kwargs):
        seen_input_lengths.append(len(input["messages"]))
        human = HumanMessage(id=f"h{len(seen_input_lengths)}", content="hi")
        answer = AIMessage(id=f"ai{len(seen_input_lengths)}", name="writer", content="ok")
        yield (), {"messages": [*input["messages"], answer]}

    monkeypatch.setattr(CompiledStateGraph, "astream", fake_astream)

    project = load(str(tmp_project))
    runner = LangGraphRunner()
    session = RunSession()

    runner.run_sync(project, "writer", "turn 1", session=session)
    runner.run_sync(project, "writer", "turn 2", session=session)

    # Turn 2's input carries turn 1's own input (1 message) PLUS its final
    # state (2 messages: the plain-dict human message plus the AIMessage
    # answer) PLUS turn 2's own new user message -- i.e. strictly more than
    # turn 1 saw, proving history threads across turns via RunSession
    # rather than each turn starting blank.
    assert seen_input_lengths[0] == 1
    assert seen_input_lengths[1] > seen_input_lengths[0]
    assert session.turns == 2
    # After turn 2: turn 1's own 2-message final state, plus turn 2's new
    # user message, plus turn 2's own answer -- 4 messages total.
    assert len(session.native["langgraph"]["messages"]) == 4
