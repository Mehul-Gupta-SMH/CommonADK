"""Tests for the Claude Agent SDK and CrewAI runners (issue #22, this
agent's two targets: `claude`, `crewai`).

A new file, deliberately kept separate from `tests/test_runners.py` (which
the parallel autogen/langgraph agent may also be touching) -- see that
file's own docstring for the shared idiom this one follows: everything here
is offline (no network, no API key, no real LLM call); `pytest.importorskip`
per-test (not at module scope) so this file degrades gracefully in an
environment missing either SDK; and each normalization test feeds a runner
*real, constructed SDK objects* (`claude_agent_sdk.types.AssistantMessage`,
`crewai.events.types.tool_usage_events.ToolUsageFinishedEvent`, etc.)
through a monkeypatched native entry point, never a fabricated stand-in.

Claude Agent SDK tests monkeypatch `ClaudeSDKClient.connect`/`.query`/
`.receive_messages`/`.disconnect` at the class level -- `.receive_response`
itself is left completely unmodified, so its real termination-on-
`ResultMessage` logic (`claude_agent_sdk/client.py`) is exercised exactly as
written, not reimplemented here.

CrewAI tests build a REAL `crewai.Crew` (via `project.build(..., target=
"crewai")`, the example project, offline) and monkeypatch only `Crew.
kickoff_async` to emit real `crewai.events` objects onto the real, global
`crewai_event_bus` and return a stub result -- this exercises the runner's
actual event-bus subscription, filtering, and `flush()` handshake against
the SDK's real (process-wide) bus, not a fake one.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from commonadk import load
from commonadk.runners import RunSession
from commonadk.runners.events import (
    AgentFinished,
    AgentStarted,
    LLMCall,
    RunError,
    RunFinished,
    RunStarted,
    ToolCall,
    Transfer,
)


@pytest.fixture()
def tavily_env(monkeypatch):
    """Satisfy researcher's one required env var -- mirrors
    tests/test_runners.py's fixture of the same name."""
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.delenv("POSTGRES_DSN", raising=False)


@pytest.fixture()
def anthropic_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


# ---------------------------------------------------------------------------
# Claude Agent SDK runner
# ---------------------------------------------------------------------------


def test_claude_runner_normalizes_tool_call_and_llm_usage(
    tmp_project, tavily_env, anthropic_env, monkeypatch
):
    pytest.importorskip("claude_agent_sdk")

    from claude_agent_sdk import ClaudeSDKClient
    from claude_agent_sdk.types import AssistantMessage, ResultMessage, TextBlock, ToolResultBlock, ToolUseBlock, UserMessage

    from commonadk.runners.claude_agent import ClaudeAgentSDKRunner

    call_message = AssistantMessage(
        content=[
            ToolUseBlock(id="call1", name="split_into_subtopics", input={"topic": "EV adoption"}),
        ],
        model="claude-sonnet-5",
    )
    result_message_for_tool = UserMessage(
        content=[ToolResultBlock(tool_use_id="call1", content="ok", is_error=False)],
    )
    text_message = AssistantMessage(content=[TextBlock(text="Answer text")], model="claude-sonnet-5")
    result_message = ResultMessage(
        subtype="success",
        duration_ms=120,
        duration_api_ms=80,
        is_error=False,
        num_turns=1,
        session_id="s1",
        total_cost_usd=0.001234,
        result="Answer text",
        model_usage={
            "claude-sonnet-5": {
                "inputTokens": 100,
                "outputTokens": 40,
                "cacheReadInputTokens": 0,
                "cacheCreationInputTokens": 0,
                "webSearchRequests": 0,
                "costUSD": 0.001234,
                "contextWindow": 200000,
                "maxOutputTokens": 8192,
            }
        },
    )
    fake_messages = [call_message, result_message_for_tool, text_message, result_message]

    async def fake_connect(self, prompt=None):
        return None

    async def fake_query(self, prompt, session_id="default"):
        return None

    async def fake_receive_messages(self):
        for message in fake_messages:
            yield message

    async def fake_disconnect(self):
        fake_disconnect.called = True

    fake_disconnect.called = False

    monkeypatch.setattr(ClaudeSDKClient, "connect", fake_connect)
    monkeypatch.setattr(ClaudeSDKClient, "query", fake_query)
    monkeypatch.setattr(ClaudeSDKClient, "receive_messages", fake_receive_messages)
    monkeypatch.setattr(ClaudeSDKClient, "disconnect", fake_disconnect)

    project = load(str(tmp_project))
    runner = ClaudeAgentSDKRunner()
    trace = runner.run_sync(project, "coordinator", "research EV adoption")

    kinds = [e.kind for e in trace.events]
    assert kinds == [
        "run_started",
        "agent_started",
        "tool_call",
        "llm_call",
        "agent_finished",
        "run_finished",
    ]

    tool_call = next(e for e in trace.events if isinstance(e, ToolCall))
    assert tool_call.agent_name == "coordinator"
    assert tool_call.tool_name == "split_into_subtopics"
    assert tool_call.arguments == {"topic": "EV adoption"}
    assert tool_call.error is None
    assert tool_call.duration_ms is not None and tool_call.duration_ms >= 0

    llm_call = next(e for e in trace.events if isinstance(e, LLMCall))
    assert llm_call.agent_name == "coordinator"
    assert llm_call.model == "claude-sonnet-5"
    assert llm_call.prompt_tokens == 100
    assert llm_call.completion_tokens == 40
    assert llm_call.total_tokens == 140
    # SDK-computed cost, passed through verbatim -- pricing.py is never
    # consulted for this target (see claude_agent.py's docstring).
    assert llm_call.cost_usd == 0.001234
    assert llm_call.duration_ms == 80.0  # ResultMessage.duration_api_ms

    finished = trace.events[-1]
    assert isinstance(finished, RunFinished)
    assert finished.final_text == "Answer text"
    assert finished.total_tokens == 140
    assert finished.total_cost_usd == 0.001234
    assert finished.usage_complete is True

    # session=None -- the ad hoc client this runner created must be closed.
    assert fake_disconnect.called is True


def test_claude_runner_usage_unreported_becomes_none_not_zero(
    tmp_project, tavily_env, anthropic_env, monkeypatch
):
    """The load-bearing None-vs-0 rule, this SDK's version: an empty
    `model_usage` must never be summed into a confident `0` -- see
    claude_agent.py's `_llm_call_for` docstring."""
    pytest.importorskip("claude_agent_sdk")

    from claude_agent_sdk import ClaudeSDKClient
    from claude_agent_sdk.types import ResultMessage

    from commonadk.runners.claude_agent import ClaudeAgentSDKRunner

    result_message = ResultMessage(
        subtype="success",
        duration_ms=50,
        duration_api_ms=None,
        is_error=False,
        num_turns=1,
        session_id="s1",
        total_cost_usd=None,
        result="hi",
        model_usage=None,
        usage=None,
    )

    async def fake_connect(self, prompt=None):
        return None

    async def fake_query(self, prompt, session_id="default"):
        return None

    async def fake_receive_messages(self):
        yield result_message

    async def fake_disconnect(self):
        return None

    monkeypatch.setattr(ClaudeSDKClient, "connect", fake_connect)
    monkeypatch.setattr(ClaudeSDKClient, "query", fake_query)
    monkeypatch.setattr(ClaudeSDKClient, "receive_messages", fake_receive_messages)
    monkeypatch.setattr(ClaudeSDKClient, "disconnect", fake_disconnect)

    project = load(str(tmp_project))
    runner = ClaudeAgentSDKRunner()
    trace = runner.run_sync(project, "coordinator", "hi")

    llm_call = next(e for e in trace.events if isinstance(e, LLMCall))
    assert llm_call.prompt_tokens is None
    assert llm_call.completion_tokens is None
    assert llm_call.total_tokens is None
    assert llm_call.cost_usd is None
    assert llm_call.duration_ms is None

    finished = trace.events[-1]
    assert isinstance(finished, RunFinished)
    assert finished.usage_complete is False  # one LLMCall, unreported -- honest, not "complete"
    assert finished.total_tokens is None
    assert finished.total_cost_usd is None


def test_claude_runner_subagent_boundary_and_transfer_via_agent_tool(
    tmp_project, tavily_env, anthropic_env, monkeypatch
):
    """The `"Agent"` tool is this SDK's subagent-invocation mechanism (see
    adapters/claude_agent.py's docstring); `parent_tool_use_id` on later
    messages is the real SDK signal this runner uses to attribute them to
    the subagent instead of the root -- not a guess."""
    pytest.importorskip("claude_agent_sdk")

    from claude_agent_sdk import ClaudeSDKClient
    from claude_agent_sdk.types import AssistantMessage, ResultMessage, TextBlock, ToolResultBlock, ToolUseBlock, UserMessage

    from commonadk.runners.claude_agent import ClaudeAgentSDKRunner

    agent_call = AssistantMessage(
        content=[ToolUseBlock(id="agent1", name="Agent", input={"subagent_type": "researcher", "prompt": "find sources"})],
        model="claude-sonnet-5",
    )
    subagent_reply = AssistantMessage(
        content=[TextBlock(text="Found 3 sources")],
        model="claude-sonnet-5",
        parent_tool_use_id="agent1",
    )
    agent_result = UserMessage(
        content=[ToolResultBlock(tool_use_id="agent1", content="Found 3 sources", is_error=False)],
    )
    result_message = ResultMessage(
        subtype="success",
        duration_ms=200,
        duration_api_ms=150,
        is_error=False,
        num_turns=2,
        session_id="s1",
        total_cost_usd=0.01,
        result="Done",
        model_usage=None,
    )
    fake_messages = [agent_call, subagent_reply, agent_result, result_message]

    async def fake_connect(self, prompt=None):
        return None

    async def fake_query(self, prompt, session_id="default"):
        return None

    async def fake_receive_messages(self):
        for message in fake_messages:
            yield message

    async def fake_disconnect(self):
        return None

    monkeypatch.setattr(ClaudeSDKClient, "connect", fake_connect)
    monkeypatch.setattr(ClaudeSDKClient, "query", fake_query)
    monkeypatch.setattr(ClaudeSDKClient, "receive_messages", fake_receive_messages)
    monkeypatch.setattr(ClaudeSDKClient, "disconnect", fake_disconnect)

    project = load(str(tmp_project))
    runner = ClaudeAgentSDKRunner()
    trace = runner.run_sync(project, "coordinator", "research EV adoption")

    transfer = next(e for e in trace.events if isinstance(e, Transfer))
    assert transfer.from_agent == "coordinator"
    assert transfer.to_agent == "researcher"
    assert transfer.transfer_kind == "claude-agent-sdk:agent_tool"

    agent_started = [e for e in trace.events if isinstance(e, AgentStarted)]
    agent_finished = [e for e in trace.events if isinstance(e, AgentFinished)]
    # coordinator -> researcher (the subagent's own reply) -> coordinator
    # again (the ToolResultBlock closing the "Agent" call arrives back in
    # the root's own context, parent_tool_use_id=None) -> closed at the end.
    assert [e.agent_name for e in agent_started] == ["coordinator", "researcher", "coordinator"]
    assert [e.agent_name for e in agent_finished] == ["coordinator", "researcher", "coordinator"]

    tool_call = next(e for e in trace.events if isinstance(e, ToolCall))
    assert tool_call.tool_name == "Agent"
    assert tool_call.agent_name == "coordinator"  # the call itself belongs to the delegator


def test_claude_runner_result_message_is_error_is_fatal_and_reraises(
    tmp_project, tavily_env, anthropic_env, monkeypatch
):
    pytest.importorskip("claude_agent_sdk")

    from claude_agent_sdk import ClaudeSDKClient
    from claude_agent_sdk.types import ResultMessage

    from commonadk.runners.claude_agent import ClaudeAgentSDKRunner

    result_message = ResultMessage(
        subtype="error_max_turns",
        duration_ms=10,
        duration_api_ms=5,
        is_error=True,
        num_turns=1,
        session_id="s1",
        result=None,
    )

    async def fake_connect(self, prompt=None):
        return None

    async def fake_query(self, prompt, session_id="default"):
        return None

    async def fake_receive_messages(self):
        yield result_message

    async def fake_disconnect(self):
        return None

    monkeypatch.setattr(ClaudeSDKClient, "connect", fake_connect)
    monkeypatch.setattr(ClaudeSDKClient, "query", fake_query)
    monkeypatch.setattr(ClaudeSDKClient, "receive_messages", fake_receive_messages)
    monkeypatch.setattr(ClaudeSDKClient, "disconnect", fake_disconnect)

    project = load(str(tmp_project))
    runner = ClaudeAgentSDKRunner()

    with pytest.raises(RuntimeError, match="turn ended in error"):
        runner.run_sync(project, "coordinator", "hi")


def test_claude_runner_missing_api_key_raises_oserror(tmp_project, tavily_env, monkeypatch):
    pytest.importorskip("claude_agent_sdk")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    from commonadk.runners.claude_agent import ClaudeAgentSDKRunner

    project = load(str(tmp_project))
    runner = ClaudeAgentSDKRunner()

    with pytest.raises(OSError, match="ANTHROPIC_API_KEY"):
        runner.run_sync(project, "coordinator", "hi")


def test_claude_runner_session_reuses_same_client_across_turns(
    tmp_project, tavily_env, anthropic_env, monkeypatch
):
    pytest.importorskip("claude_agent_sdk")

    from claude_agent_sdk import ClaudeSDKClient
    from claude_agent_sdk.types import ResultMessage

    from commonadk.runners.claude_agent import ClaudeAgentSDKRunner

    connect_calls = []
    disconnect_calls = []

    async def fake_connect(self, prompt=None):
        connect_calls.append(self)

    async def fake_query(self, prompt, session_id="default"):
        return None

    async def fake_receive_messages(self):
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="s1",
            result="ok",
        )

    async def fake_disconnect(self):
        disconnect_calls.append(self)

    monkeypatch.setattr(ClaudeSDKClient, "connect", fake_connect)
    monkeypatch.setattr(ClaudeSDKClient, "query", fake_query)
    monkeypatch.setattr(ClaudeSDKClient, "receive_messages", fake_receive_messages)
    monkeypatch.setattr(ClaudeSDKClient, "disconnect", fake_disconnect)

    project = load(str(tmp_project))
    runner = ClaudeAgentSDKRunner()
    session = RunSession()

    runner.run_sync(project, "coordinator", "turn 1", session=session)
    runner.run_sync(project, "coordinator", "turn 2", session=session)

    assert len(connect_calls) == 1  # same client reused, not reconnected
    assert disconnect_calls == []  # a session-bound client is never closed by this runner
    assert session.turns == 2


# ---------------------------------------------------------------------------
# CrewAI runner
# ---------------------------------------------------------------------------


def test_crewai_runner_session_raises_not_implemented(tmp_project, tavily_env):
    pytest.importorskip("crewai")

    from commonadk.runners.crewai_runner import CrewAIRunner

    project = load(str(tmp_project))
    runner = CrewAIRunner()

    with pytest.raises(NotImplementedError, match="no multi-turn session"):
        runner.run_sync(project, "coordinator", "hi", session=RunSession())


def test_crewai_runner_normalizes_agent_tool_and_llm_events_and_delegation_transfer(
    tmp_project, tavily_env, monkeypatch
):
    pytest.importorskip("crewai")
    os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
    os.environ.setdefault("OTEL_SDK_DISABLED", "true")

    from crewai import Crew
    from crewai.events import crewai_event_bus
    from crewai.events.types.agent_events import (
        AgentExecutionCompletedEvent,
        AgentExecutionStartedEvent,
    )
    from crewai.events.types.llm_events import LLMCallCompletedEvent, LLMCallType
    from crewai.events.types.tool_usage_events import ToolUsageFinishedEvent

    from commonadk.runners.crewai_runner import CrewAIRunner

    project = load(str(tmp_project))

    # NOTE: `project.build(...)` constructs a FRESH `Crew` (with fresh
    # `Agent` instances, and thus fresh `.id`s) every call -- the runner
    # calls it itself inside `run()`, so this fake `kickoff_async` reads
    # `self`/`self.manager_agent`/`self.agents` (the exact `Crew` instance
    # the runner just built and is now kicking off) rather than a
    # separately pre-built crew, whose agent ids would not match the ones
    # the runner's own `relevant_agent_ids` filter actually uses.
    async def fake_kickoff_async(self, inputs=None, input_files=None, from_checkpoint=None):
        manager = self.manager_agent
        researcher = next(a for a in self.agents if a.role == "researcher")
        started_at = datetime.now(timezone.utc)
        crewai_event_bus.emit(
            manager,
            event=AgentExecutionStartedEvent(
                agent=manager, task=None, tools=None, task_prompt="route the request"
            ),
        )
        crewai_event_bus.emit(
            manager,
            event=ToolUsageFinishedEvent(
                from_agent=manager,
                tool_name="Delegate work to coworker",
                tool_args={"task": "find sources", "context": "ev adoption", "coworker": "researcher"},
                started_at=started_at,
                finished_at=started_at + timedelta(milliseconds=50),
                output="delegated ok",
            ),
        )
        crewai_event_bus.emit(
            researcher,
            event=LLMCallCompletedEvent(
                from_agent=researcher,
                call_id="call-1",
                model="gemini/gemini-2.5-pro",
                response="the research findings",
                call_type=LLMCallType.LLM_CALL,
                usage={"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
            ),
        )
        crewai_event_bus.emit(
            manager,
            event=AgentExecutionCompletedEvent(agent=manager, task=None, output="final answer"),
        )
        # Give the bus's ThreadPoolExecutor-dispatched handlers a chance to
        # run before this coroutine returns -- the runner's own flush()
        # call afterwards is the real synchronization point production
        # code relies on, but emit() dispatch is async relative to this
        # function returning, so a real crew would already have this gap
        # too (kickoff() itself runs in a to_thread worker).
        await asyncio.sleep(0)
        return SimpleNamespace(raw="final answer")

    monkeypatch.setattr(Crew, "kickoff_async", fake_kickoff_async)

    runner = CrewAIRunner()
    trace = runner.run_sync(project, "coordinator", "research EV adoption")

    # See crewai_runner.py's module docstring, "THE THREAD-CROSSING
    # CAVEAT": events of *different* kinds are dispatched on a 10-worker
    # thread pool and race, so only run_started-first/run_finished-last and
    # each kind's multiplicity are guaranteed across the whole trace --
    # NOT a strict total order between e.g. tool_call and llm_call.
    kinds = [e.kind for e in trace.events]
    assert kinds[0] == "run_started"
    assert kinds[-1] == "run_finished"
    assert sorted(kinds) == sorted(
        ["run_started", "agent_started", "tool_call", "transfer", "llm_call", "agent_finished", "run_finished"]
    )

    tool_call = next(e for e in trace.events if isinstance(e, ToolCall))
    assert tool_call.agent_name == "coordinator"
    assert tool_call.tool_name == "Delegate work to coworker"
    assert tool_call.duration_ms == pytest.approx(50.0, abs=1.0)
    assert tool_call.error is None

    transfer = next(e for e in trace.events if isinstance(e, Transfer))
    assert transfer.from_agent == "coordinator"
    assert transfer.to_agent == "researcher"
    assert transfer.transfer_kind == "crewai:Delegate work to coworker"
    # Both emitted from the same on_tool_finished handler invocation, so
    # THIS pair's relative order is guaranteed unlike the cross-handler case.
    assert tool_call.seq < transfer.seq

    llm_call = next(e for e in trace.events if isinstance(e, LLMCall))
    assert llm_call.agent_name == "researcher"
    assert llm_call.model == "gemini/gemini-2.5-pro"
    assert llm_call.prompt_tokens == 50
    assert llm_call.completion_tokens == 20
    assert llm_call.total_tokens == 70
    # gemini-2.5-pro IS in pricing.py's table -- (50/1e6)*1.25 + (20/1e6)*10.00
    assert llm_call.cost_usd == pytest.approx(0.000263, abs=1e-6)

    finished = trace.events[-1]
    assert isinstance(finished, RunFinished)
    assert finished.final_text == "final answer"
    assert finished.total_tokens == 70
    assert finished.usage_complete is True


def test_crewai_runner_llm_usage_unreported_becomes_none_not_zero(
    tmp_project, tavily_env, monkeypatch
):
    """`UsageMetrics`' fields are plain `int`s defaulting to `0`, exactly
    like `agents.usage.Usage` in openai_agents.py -- `usage=None` on the
    event must become `None` fields on `LLMCall`, never a confident `0`."""
    pytest.importorskip("crewai")
    os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
    os.environ.setdefault("OTEL_SDK_DISABLED", "true")

    from crewai import Crew
    from crewai.events import crewai_event_bus
    from crewai.events.types.llm_events import LLMCallCompletedEvent, LLMCallType

    from commonadk.runners.crewai_runner import CrewAIRunner

    project = load(str(tmp_project))

    async def fake_kickoff_async(self, inputs=None, input_files=None, from_checkpoint=None):
        manager = self.manager_agent
        crewai_event_bus.emit(
            manager,
            event=LLMCallCompletedEvent(
                from_agent=manager,
                call_id="call-2",
                model="gemini/gemini-2.5-flash",
                response="ok",
                call_type=LLMCallType.LLM_CALL,
                usage=None,
            ),
        )
        await asyncio.sleep(0)
        return SimpleNamespace(raw="ok")

    monkeypatch.setattr(Crew, "kickoff_async", fake_kickoff_async)

    runner = CrewAIRunner()
    trace = runner.run_sync(project, "coordinator", "hi")

    llm_call = next(e for e in trace.events if isinstance(e, LLMCall))
    assert llm_call.prompt_tokens is None
    assert llm_call.completion_tokens is None
    assert llm_call.total_tokens is None
    assert llm_call.cost_usd is None

    finished = trace.events[-1]
    assert isinstance(finished, RunFinished)
    assert finished.usage_complete is False


def test_crewai_runner_tool_error_is_reported_on_the_tool_call(
    tmp_project, tavily_env, monkeypatch
):
    pytest.importorskip("crewai")
    os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
    os.environ.setdefault("OTEL_SDK_DISABLED", "true")

    from crewai import Crew
    from crewai.events import crewai_event_bus
    from crewai.events.types.tool_usage_events import ToolUsageErrorEvent

    from commonadk.runners.crewai_runner import CrewAIRunner

    project = load(str(tmp_project))

    async def fake_kickoff_async(self, inputs=None, input_files=None, from_checkpoint=None):
        researcher = next(a for a in self.agents if a.role == "researcher")
        crewai_event_bus.emit(
            researcher,
            event=ToolUsageErrorEvent(
                from_agent=researcher,
                tool_name="search_web",
                tool_args={"query": "ev adoption"},
                error="connection timed out",
            ),
        )
        await asyncio.sleep(0)
        return SimpleNamespace(raw="failed")

    monkeypatch.setattr(Crew, "kickoff_async", fake_kickoff_async)

    runner = CrewAIRunner()
    trace = runner.run_sync(project, "coordinator", "hi")

    tool_call = next(e for e in trace.events if isinstance(e, ToolCall))
    assert tool_call.tool_name == "search_web"
    assert tool_call.agent_name == "researcher"
    assert tool_call.error == "connection timed out"
    assert tool_call.duration_ms is None  # ToolUsageErrorEvent carries no started_at/finished_at


def test_crewai_runner_ignores_events_from_an_unrelated_crew(
    tmp_project, tavily_env, monkeypatch
):
    """The global-bus caveat this runner exists to guard against (see
    crewai_runner.py's module docstring, "THE GLOBAL-BUS CAVEAT"): an event
    tagged with an agent id that isn't part of *this* run's `Crew` must
    never leak into this run's trace."""
    pytest.importorskip("crewai")
    os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
    os.environ.setdefault("OTEL_SDK_DISABLED", "true")

    from crewai import Agent, Crew
    from crewai.events import crewai_event_bus
    from crewai.events.types.tool_usage_events import ToolUsageFinishedEvent

    from commonadk.runners.crewai_runner import CrewAIRunner

    project = load(str(tmp_project))
    # A real Agent, but never added to any Crew this runner builds -- its
    # id can never appear in relevant_agent_ids.
    intruder = Agent(role="intruder", goal="cause trouble", backstory="not part of this crew", llm="gemini/gemini-2.5-flash")

    async def fake_kickoff_async(self, inputs=None, input_files=None, from_checkpoint=None):
        now = datetime.now(timezone.utc)
        crewai_event_bus.emit(
            intruder,
            event=ToolUsageFinishedEvent(
                from_agent=intruder,
                tool_name="unrelated_tool",
                tool_args={},
                started_at=now,
                finished_at=now,
                output="should never appear",
            ),
        )
        await asyncio.sleep(0)
        return SimpleNamespace(raw="ok")

    monkeypatch.setattr(Crew, "kickoff_async", fake_kickoff_async)

    runner = CrewAIRunner()
    trace = runner.run_sync(project, "coordinator", "hi")

    assert [e for e in trace.events if isinstance(e, ToolCall)] == []


def test_crewai_runner_unregisters_handlers_so_a_second_run_does_not_double_count(
    tmp_project, tavily_env, monkeypatch
):
    """If this runner ever failed to `crewai_event_bus.off(...)` its
    handlers, a second, unrelated `run()` call would double-emit (or worse,
    cross-contaminate) events on the shared, process-wide bus -- see
    crewai_runner.py's module docstring."""
    pytest.importorskip("crewai")
    os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
    os.environ.setdefault("OTEL_SDK_DISABLED", "true")

    from crewai import Crew
    from crewai.events import crewai_event_bus
    from crewai.events.types.tool_usage_events import ToolUsageFinishedEvent

    from commonadk.runners.crewai_runner import CrewAIRunner

    project = load(str(tmp_project))

    async def fake_kickoff_async(self, inputs=None, input_files=None, from_checkpoint=None):
        manager = self.manager_agent
        now = datetime.now(timezone.utc)
        crewai_event_bus.emit(
            manager,
            event=ToolUsageFinishedEvent(
                from_agent=manager,
                tool_name="a_tool",
                tool_args={},
                started_at=now,
                finished_at=now,
                output="ok",
            ),
        )
        await asyncio.sleep(0)
        return SimpleNamespace(raw="ok")

    monkeypatch.setattr(Crew, "kickoff_async", fake_kickoff_async)

    runner = CrewAIRunner()
    trace1 = runner.run_sync(project, "coordinator", "first run")
    trace2 = runner.run_sync(project, "coordinator", "second run")

    assert len([e for e in trace1.events if isinstance(e, ToolCall)]) == 1
    assert len([e for e in trace2.events if isinstance(e, ToolCall)]) == 1
