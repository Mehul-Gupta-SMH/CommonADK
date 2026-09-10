"""Tests for `commonadk.runners` (issue #22 -- execution & telemetry layer).

Everything here is offline: no test makes a real LLM call or touches the
network. The event model, hook registry, trace roll-up, and runner registry
tests need no agent SDK at all. The Google ADK / OpenAI Agents
*normalization* tests need their respective SDK installed --
`pytest.importorskip` inside each such test (not at module scope, since the
registry/event/hook/trace tests must keep running even in an environment
that has neither SDK) skips them, not fails them, when the SDK is missing --
mirroring `tests/test_adapter_*.py`'s module-scope pattern, applied
per-test here because this one file spans both SDK-free and per-SDK
coverage.

The Google ADK and OpenAI Agents normalization tests feed each runner
*real, constructed SDK objects* (`google.adk.events.event.Event`,
`agents.items.ToolCallItem`, etc. -- actual installed classes, never
fabricated stand-ins) through a monkeypatched native run/stream call, so
the assertions exercise the exact mapping code each runner ships, without
a network call or an API key. This is the same style the task's
instructions require: never trust memory of an SDK's API, never fabricate
a real LLM turn.
"""

from __future__ import annotations

import json

import pytest

from commonadk import cli, load
from commonadk.runners import HookRegistry, RunSession, Trace, get_runner
from commonadk.runners import known_targets as runner_known_targets
from commonadk.runners import known_unported_targets
from commonadk.runners.events import (
    AgentFinished,
    AgentStarted,
    LLMCall,
    RunError,
    RunFinished,
    RunStarted,
    ToolCall,
    Transfer,
    event_from_dict,
)
from commonadk.runners.pricing import estimate_cost_usd


@pytest.fixture()
def tavily_env(monkeypatch):
    """Satisfy researcher's one required env var -- mirrors
    tests/test_adapter_google.py's fixture of the same name."""
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.delenv("POSTGRES_DSN", raising=False)


# ---------------------------------------------------------------------------
# event model: ordering, sequence numbers, JSON round-trip, None-vs-0
# ---------------------------------------------------------------------------


def test_sequence_numbers_are_monotonic_across_event_types():
    a = RunStarted(run_id="r1", target="openai", agent_name="x", prompt="hi")
    b = AgentStarted(run_id="r1", agent_name="x")
    c = LLMCall(run_id="r1")
    assert a.seq < b.seq < c.seq


def test_event_to_dict_carries_a_type_discriminator():
    event = ToolCall(run_id="r1", agent_name="x", tool_name="search_web")
    d = event.to_dict()
    assert d["type"] == "tool_call"
    assert d["tool_name"] == "search_web"


def test_event_json_round_trip_is_lossless():
    original = LLMCall(
        run_id="r1",
        agent_name="coordinator",
        model="gemini/gemini-2.5-flash",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        cost_usd=0.000016,
    )
    restored = event_from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored == original


def test_event_from_dict_rejects_unknown_type():
    with pytest.raises(ValueError, match="unknown event type"):
        event_from_dict({"type": "not_a_real_event", "seq": 1, "ts": 0.0, "run_id": "r1"})


def test_llm_call_defaults_usage_fields_to_none_not_zero():
    """The load-bearing None-vs-0 rule: an LLMCall constructed with no
    usage info must default every token/cost field to None, never 0 --
    0 would silently claim "this call cost nothing" instead of "this SDK
    reported nothing"."""
    event = LLMCall(run_id="r1")
    assert event.prompt_tokens is None
    assert event.completion_tokens is None
    assert event.total_tokens is None
    assert event.cost_usd is None
    assert event.duration_ms is None


def test_run_finished_and_run_error_are_distinct_terminal_events():
    finished = RunFinished(run_id="r1", final_text="done")
    error = RunError(run_id="r1", message="boom", error_type="ValueError")
    assert finished.kind == "run_finished"
    assert error.kind == "run_error"


# ---------------------------------------------------------------------------
# hooks: ordering, catch-all, isolate-and-report exception policy
# ---------------------------------------------------------------------------


def test_hooks_fire_in_registration_order_type_specific_then_catch_all():
    calls: list[str] = []
    registry = HookRegistry()
    registry.register(lambda e: calls.append("specific"), event_type=RunStarted)
    registry.register(lambda e: calls.append("catch-all-1"))
    registry.register(lambda e: calls.append("catch-all-2"))

    registry.fire(RunStarted(run_id="r1", target="openai", agent_name="x", prompt="hi"))

    assert calls == ["specific", "catch-all-1", "catch-all-2"]


def test_hook_registered_for_other_type_does_not_fire():
    calls: list[str] = []
    registry = HookRegistry()
    registry.register(lambda e: calls.append("tool"), event_type=ToolCall)

    registry.fire(RunStarted(run_id="r1", target="openai", agent_name="x", prompt="hi"))

    assert calls == []


def test_raising_hook_is_isolated_and_reported_run_continues():
    calls: list[str] = []

    def broken_hook(event):
        raise RuntimeError("boom")

    registry = HookRegistry()
    registry.register(broken_hook)
    registry.register(lambda e: calls.append("second"))

    with pytest.warns(UserWarning, match="boom"):
        registry.fire(RunStarted(run_id="r1", target="openai", agent_name="x", prompt="hi"))

    # The second hook still ran -- one broken observer never blocks another.
    assert calls == ["second"]
    # The failure is recorded, not silently dropped.
    assert len(registry.errors) == 1
    failed_event, failed_callback, exc = registry.errors[0]
    assert failed_callback is broken_hook
    assert isinstance(exc, RuntimeError)


def test_multiple_raising_hooks_each_isolated():
    registry = HookRegistry()
    registry.register(lambda e: (_ for _ in ()).throw(ValueError("first")))
    registry.register(lambda e: (_ for _ in ()).throw(TypeError("second")))

    with pytest.warns(UserWarning):
        registry.fire(RunStarted(run_id="r1", target="openai", agent_name="x", prompt="hi"))

    assert len(registry.errors) == 2


# ---------------------------------------------------------------------------
# trace: roll-ups, None-vs-0/incomplete-usage handling, JSON (de)serialization
# ---------------------------------------------------------------------------


def test_trace_rollup_sums_reported_usage_correctly():
    trace = Trace()
    trace.append(LLMCall(run_id="r1", agent_name="a", model="gpt-4o", prompt_tokens=100, completion_tokens=50, total_tokens=150, cost_usd=0.0025))
    trace.append(LLMCall(run_id="r1", agent_name="a", model="gpt-4o", prompt_tokens=200, completion_tokens=25, total_tokens=225, cost_usd=0.00075))
    trace.append(ToolCall(run_id="r1", agent_name="a", tool_name="search_web"))

    totals = trace.rollup()
    assert totals["llm_calls"]["usage_complete"] is True
    assert totals["llm_calls"]["prompt_tokens"] == 300
    assert totals["llm_calls"]["completion_tokens"] == 75
    assert totals["llm_calls"]["total_tokens"] == 375
    assert totals["llm_calls"]["cost_usd"] == pytest.approx(0.00325)
    assert totals["tool_calls"]["count"] == 1
    assert totals["per_agent"]["a"]["llm_calls"]["total_tokens"] == 375


def test_trace_marks_incomplete_usage_instead_of_summing_a_misleading_total():
    trace = Trace()
    trace.append(LLMCall(run_id="r1", agent_name="a", model="gpt-4o", prompt_tokens=100, completion_tokens=50, total_tokens=150, cost_usd=0.0025))
    # Second call: the SDK reported nothing for it.
    trace.append(LLMCall(run_id="r1", agent_name="a"))

    totals = trace.rollup()["llm_calls"]
    assert totals["count"] == 2
    assert totals["reported_count"] == 1
    assert totals["usage_complete"] is False
    # The partial sum is still surfaced (not nulled out)...
    assert totals["total_tokens"] == 150
    # ...but explicitly labeled as partial, not the true total.
    assert totals["note"] is not None
    assert "1 of 2" in totals["note"]


def test_trace_marks_incomplete_cost_when_model_unpriced_but_usage_complete():
    trace = Trace()
    trace.append(
        LLMCall(
            run_id="r1",
            agent_name="a",
            model="some-unpriced-model",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cost_usd=None,
        )
    )
    totals = trace.rollup()["llm_calls"]
    assert totals["usage_complete"] is True
    assert totals["cost_complete"] is False
    assert totals["cost_usd"] is None
    assert "pricing table" in totals["note"]


def test_trace_llm_calls_with_no_agent_excluded_from_per_agent_but_counted_overall():
    trace = Trace()
    trace.append(LLMCall(run_id="r1", agent_name=None, total_tokens=10))
    totals = trace.rollup()
    assert totals["llm_calls"]["count"] == 1
    assert totals["per_agent"] == {}


def test_trace_json_round_trip_via_file(tmp_path):
    trace = Trace()
    trace.append(RunStarted(run_id="r1", target="openai", agent_name="coordinator", prompt="hi"))
    trace.append(LLMCall(run_id="r1", agent_name="coordinator", model="gpt-4o", prompt_tokens=1, completion_tokens=1, total_tokens=2, cost_usd=0.00001))
    trace.append(RunFinished(run_id="r1", final_text="done", usage_complete=True))

    path = tmp_path / "trace.json"
    trace.write(path)

    restored = Trace.read(path)
    assert [e.to_dict() for e in restored.events] == [e.to_dict() for e in trace.events]
    assert restored.rollup() == trace.rollup()


def test_trace_final_event_finds_last_run_finished_or_run_error():
    trace = Trace()
    trace.append(RunStarted(run_id="r1", target="openai", agent_name="x", prompt="hi"))
    finished = RunFinished(run_id="r1", final_text="ok")
    trace.append(finished)
    assert trace.final_event() is finished


# ---------------------------------------------------------------------------
# pricing: never applied to a token count the SDK didn't report
# ---------------------------------------------------------------------------


def test_estimate_cost_usd_none_for_unpriced_model():
    assert estimate_cost_usd("not-a-real-model", 100, 100) is None


def test_estimate_cost_usd_none_when_tokens_missing():
    assert estimate_cost_usd("gpt-4o", None, 10) is None
    assert estimate_cost_usd("gpt-4o", 10, None) is None


def test_estimate_cost_usd_computes_for_known_model_bare_or_prefixed():
    bare = estimate_cost_usd("gpt-4o", 1_000_000, 0)
    prefixed = estimate_cost_usd("openai/gpt-4o", 1_000_000, 0)
    assert bare == prefixed == pytest.approx(2.50)


# ---------------------------------------------------------------------------
# registry: unknown target, unported target, missing-SDK install hint
# ---------------------------------------------------------------------------


def test_get_runner_unknown_target_names_known_targets():
    with pytest.raises(ValueError) as exc_info:
        get_runner("not-a-real-sdk")
    message = str(exc_info.value)
    assert "not-a-real-sdk" in message
    assert "google-adk" in message


def test_get_runner_unported_target_gives_clear_not_yet_available_message():
    for target in known_unported_targets():
        with pytest.raises(NotImplementedError) as exc_info:
            get_runner(target)
        message = str(exc_info.value)
        assert target in message
        assert "not available yet" in message
        assert "google-adk" in message and "openai" in message


def test_get_runner_missing_sdk_gives_install_hint(monkeypatch):
    import importlib

    import commonadk.runners as runners_pkg

    real_import_module = importlib.import_module

    def fake_import_module(name, *args, **kwargs):
        if name == "commonadk.runners.google_adk":
            raise ImportError("No module named 'google'")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(runners_pkg, "import_module", fake_import_module)

    with pytest.raises(ImportError) as exc_info:
        runners_pkg.get_runner("google-adk")

    assert 'pip install "commonadk[google]"' in str(exc_info.value)


def test_known_targets_and_unported_targets_are_disjoint():
    assert set(runner_known_targets()) & set(known_unported_targets()) == set()


# ---------------------------------------------------------------------------
# Google ADK normalization -- real google.adk.events.event.Event objects
# ---------------------------------------------------------------------------


def test_google_adk_runner_normalizes_llm_tool_and_transfer_events(
    tmp_project, tavily_env, monkeypatch
):
    pytest.importorskip("google.adk")

    import google.adk.runners as adk_runners_module
    from google.adk.events.event import Event as AdkEvent
    from google.adk.events.event_actions import EventActions
    from google.genai import types as genai_types

    from commonadk.runners.google_adk import GoogleADKRunner

    function_call = genai_types.FunctionCall(
        id="call1", name="search_web", args={"query": "ev adoption"}
    )
    call_event = AdkEvent(
        author="coordinator",
        content=genai_types.Content(role="model", parts=[genai_types.Part(function_call=function_call)]),
        usage_metadata=genai_types.GenerateContentResponseUsageMetadata(
            prompt_token_count=10, candidates_token_count=5, total_token_count=15
        ),
    )
    function_response = genai_types.FunctionResponse(
        id="call1", name="search_web", response={"result": "ok"}
    )
    response_event = AdkEvent(
        author="coordinator",
        content=genai_types.Content(role="user", parts=[genai_types.Part(function_response=function_response)]),
    )
    transfer_event = AdkEvent(
        author="coordinator", actions=EventActions(transfer_to_agent="researcher")
    )
    final_event = AdkEvent(
        author="researcher",
        content=genai_types.Content(role="model", parts=[genai_types.Part(text="Answer text")]),
        usage_metadata=genai_types.GenerateContentResponseUsageMetadata(
            prompt_token_count=20, candidates_token_count=8, total_token_count=28
        ),
    )
    fake_events = [call_event, response_event, transfer_event, final_event]

    async def fake_run_async(self, *, user_id, session_id, new_message=None, **kwargs):
        for event in fake_events:
            yield event

    monkeypatch.setattr(adk_runners_module.Runner, "run_async", fake_run_async)

    project = load(str(tmp_project))
    runner = GoogleADKRunner()
    trace = runner.run_sync(project, "coordinator", "research EV adoption")

    kinds = [e.kind for e in trace.events]
    assert kinds == [
        "run_started",
        "agent_started",
        "llm_call",
        "tool_call",
        "transfer",
        "agent_finished",
        "agent_started",
        "llm_call",
        "agent_finished",
        "run_finished",
    ]

    tool_call = next(e for e in trace.events if isinstance(e, ToolCall))
    assert tool_call.agent_name == "coordinator"
    assert tool_call.tool_name == "search_web"
    assert tool_call.arguments == {"query": "ev adoption"}
    assert tool_call.error is None

    transfer = next(e for e in trace.events if isinstance(e, Transfer))
    assert transfer.from_agent == "coordinator"
    assert transfer.to_agent == "researcher"
    assert transfer.transfer_kind == "google-adk:transfer_to_agent"

    llm_calls = [e for e in trace.events if isinstance(e, LLMCall)]
    assert [c.agent_name for c in llm_calls] == ["coordinator", "researcher"]
    assert [c.total_tokens for c in llm_calls] == [15, 28]
    assert all(c.cost_usd is not None for c in llm_calls)  # gemini-2.5-flash is priced

    finished = trace.events[-1]
    assert isinstance(finished, RunFinished)
    assert finished.final_text == "Answer text"
    assert finished.total_tokens == 43
    assert finished.usage_complete is True


def test_google_adk_runner_marks_usage_incomplete_when_a_call_has_no_usage_metadata(
    tmp_project, tavily_env, monkeypatch
):
    pytest.importorskip("google.adk")

    import google.adk.runners as adk_runners_module
    from google.adk.events.event import Event as AdkEvent
    from google.genai import types as genai_types

    from commonadk.runners.google_adk import GoogleADKRunner

    # A final-response event with NO usage_metadata at all -- e.g. an SDK
    # version/backend that doesn't report it for this call.
    no_usage_event = AdkEvent(
        author="coordinator",
        content=genai_types.Content(role="model", parts=[genai_types.Part(text="Answer")]),
    )

    async def fake_run_async(self, *, user_id, session_id, new_message=None, **kwargs):
        yield no_usage_event

    monkeypatch.setattr(adk_runners_module.Runner, "run_async", fake_run_async)

    project = load(str(tmp_project))
    runner = GoogleADKRunner()
    trace = runner.run_sync(project, "coordinator", "hi")

    # No usage_metadata on the only event -> no LLMCall was even emitted for
    # it (nothing to report), so the run-wide rollup is trivially complete
    # (zero calls, not a false "usage_complete").
    llm_calls = [e for e in trace.events if isinstance(e, LLMCall)]
    assert llm_calls == []
    finished = trace.events[-1]
    assert isinstance(finished, RunFinished)
    assert finished.usage_complete is True
    assert finished.total_tokens is None


def test_google_adk_runner_emits_run_error_and_reraises_on_exception(
    tmp_project, tavily_env, monkeypatch
):
    pytest.importorskip("google.adk")

    import google.adk.runners as adk_runners_module

    from commonadk.runners.google_adk import GoogleADKRunner

    async def fake_run_async(self, *, user_id, session_id, new_message=None, **kwargs):
        raise RuntimeError("simulated SDK failure")
        yield  # pragma: no cover -- makes this an async generator

    monkeypatch.setattr(adk_runners_module.Runner, "run_async", fake_run_async)

    project = load(str(tmp_project))
    runner = GoogleADKRunner()

    trace_holder: dict[str, Trace] = {}
    hooks = HookRegistry()
    captured_trace = Trace()
    hooks.register(captured_trace.append)

    with pytest.raises(RuntimeError, match="simulated SDK failure"):
        runner.run_sync(project, "coordinator", "hi", hooks=hooks)

    kinds = [e.kind for e in captured_trace.events]
    assert kinds[0] == "run_started"
    assert kinds[-1] == "run_error"
    assert "run_finished" not in kinds
    error_event = captured_trace.events[-1]
    assert isinstance(error_event, RunError)
    assert "simulated SDK failure" in error_event.message


def test_google_adk_runner_session_reuses_same_adk_session_across_turns(
    tmp_project, tavily_env, monkeypatch
):
    pytest.importorskip("google.adk")

    import google.adk.runners as adk_runners_module
    from google.adk.events.event import Event as AdkEvent
    from google.genai import types as genai_types

    from commonadk.runners.google_adk import GoogleADKRunner

    seen_session_ids: list[str] = []

    async def fake_run_async(self, *, user_id, session_id, new_message=None, **kwargs):
        seen_session_ids.append(session_id)
        yield AdkEvent(
            author="coordinator",
            content=genai_types.Content(role="model", parts=[genai_types.Part(text="ok")]),
        )

    monkeypatch.setattr(adk_runners_module.Runner, "run_async", fake_run_async)

    project = load(str(tmp_project))
    runner = GoogleADKRunner()
    session = RunSession()

    runner.run_sync(project, "coordinator", "turn 1", session=session)
    runner.run_sync(project, "coordinator", "turn 2", session=session)

    assert len(seen_session_ids) == 2
    assert seen_session_ids[0] == seen_session_ids[1]
    assert session.turns == 2


# ---------------------------------------------------------------------------
# OpenAI Agents normalization -- real agents.items.* / agents.usage.Usage
# ---------------------------------------------------------------------------


def _fake_openai_stream_result(items, raw_responses, final_output):
    class FakeStreamResult:
        def __init__(self):
            self.raw_responses = raw_responses
            self.final_output = final_output

        async def stream_events(self):
            for item in items:
                yield item

    return FakeStreamResult()


def test_openai_agents_runner_normalizes_tool_call_and_handoff(
    tmp_project, tavily_env, monkeypatch
):
    pytest.importorskip("agents")

    from agents import Agent as OaAgent
    from agents import Runner as OaRunner
    from agents.items import HandoffOutputItem, ModelResponse, ToolCallItem, ToolCallOutputItem
    from agents.stream_events import AgentUpdatedStreamEvent, RunItemStreamEvent
    from agents.usage import Usage as OaUsage

    from commonadk.runners.openai_agents import OpenAIAgentsRunner

    coordinator = OaAgent(name="coordinator", instructions="x")
    researcher = OaAgent(name="researcher", instructions="y")

    tool_call_item = ToolCallItem(
        agent=coordinator,
        raw_item={"call_id": "call1", "name": "search_web", "arguments": '{"query": "ev"}'},
    )
    tool_output_item = ToolCallOutputItem(
        agent=coordinator, raw_item={"call_id": "call1"}, output="search results here"
    )
    handoff_item = HandoffOutputItem(
        agent=coordinator,
        raw_item={"call_id": "h1"},
        source_agent=coordinator,
        target_agent=researcher,
    )
    items = [
        RunItemStreamEvent(name="tool_called", item=tool_call_item),
        RunItemStreamEvent(name="tool_output", item=tool_output_item),
        AgentUpdatedStreamEvent(new_agent=researcher),
        RunItemStreamEvent(name="handoff_occured", item=handoff_item),
    ]
    raw_responses = [
        ModelResponse(output=[], usage=OaUsage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15), response_id="r1"),
        ModelResponse(output=[], usage=OaUsage(requests=1, input_tokens=20, output_tokens=8, total_tokens=28), response_id="r2"),
    ]
    fake_result = _fake_openai_stream_result(items, raw_responses, "Answer text")

    monkeypatch.setattr(
        OaRunner, "run_streamed", classmethod(lambda cls, starting_agent, input, **kw: fake_result)
    )

    project = load(str(tmp_project))
    runner = OpenAIAgentsRunner()
    trace = runner.run_sync(project, "coordinator", "research EV adoption")

    tool_call = next(e for e in trace.events if isinstance(e, ToolCall))
    assert tool_call.agent_name == "coordinator"
    assert tool_call.tool_name == "search_web"
    assert tool_call.arguments == {"query": "ev"}
    assert tool_call.result_summary == "search results here"

    transfer = next(e for e in trace.events if isinstance(e, Transfer))
    assert transfer.from_agent == "coordinator"
    assert transfer.to_agent == "researcher"
    assert transfer.transfer_kind == "openai-agents:handoff"

    # Documented gap: a run spanning a handoff can't attribute individual
    # ModelResponse usage to a specific agent -- see runner-design.md.
    llm_calls = [e for e in trace.events if isinstance(e, LLMCall)]
    assert len(llm_calls) == 2
    assert all(c.agent_name is None for c in llm_calls)
    assert [c.total_tokens for c in llm_calls] == [15, 28]

    finished = trace.events[-1]
    assert isinstance(finished, RunFinished)
    assert finished.final_text == "Answer text"
    assert finished.total_tokens == 43
    assert finished.usage_complete is True


def test_openai_agents_runner_attributes_llm_call_to_single_agent_when_no_handoff(
    tmp_project, tavily_env, monkeypatch
):
    pytest.importorskip("agents")

    from agents import Runner as OaRunner
    from agents.items import ModelResponse
    from agents.usage import Usage as OaUsage

    from commonadk.runners.openai_agents import OpenAIAgentsRunner

    raw_responses = [
        ModelResponse(output=[], usage=OaUsage(requests=1, input_tokens=5, output_tokens=3, total_tokens=8), response_id="r1"),
    ]
    fake_result = _fake_openai_stream_result([], raw_responses, "Answer")

    monkeypatch.setattr(
        OaRunner, "run_streamed", classmethod(lambda cls, starting_agent, input, **kw: fake_result)
    )

    project = load(str(tmp_project))
    runner = OpenAIAgentsRunner()
    trace = runner.run_sync(project, "coordinator", "hi")

    llm_call = next(e for e in trace.events if isinstance(e, LLMCall))
    assert llm_call.agent_name == "coordinator"
    assert llm_call.model is not None
    assert llm_call.total_tokens == 8


def test_openai_agents_runner_usage_not_reported_becomes_none_not_zero(
    tmp_project, tavily_env, monkeypatch
):
    """`Usage`'s int fields default to 0, not None (see runner-design.md,
    "the 0-not-None wrinkle") -- `Usage.requests == 0` is the signal this
    runner uses to still report None on LLMCall rather than a fabricated 0."""
    pytest.importorskip("agents")

    from agents import Runner as OaRunner
    from agents.items import ModelResponse
    from agents.usage import Usage as OaUsage

    from commonadk.runners.openai_agents import OpenAIAgentsRunner

    unreported_usage = OaUsage()  # requests=0, tokens all default to 0
    raw_responses = [ModelResponse(output=[], usage=unreported_usage, response_id="r1")]
    fake_result = _fake_openai_stream_result([], raw_responses, "Answer")

    monkeypatch.setattr(
        OaRunner, "run_streamed", classmethod(lambda cls, starting_agent, input, **kw: fake_result)
    )

    project = load(str(tmp_project))
    runner = OpenAIAgentsRunner()
    trace = runner.run_sync(project, "coordinator", "hi")

    llm_call = next(e for e in trace.events if isinstance(e, LLMCall))
    assert llm_call.prompt_tokens is None
    assert llm_call.completion_tokens is None
    assert llm_call.total_tokens is None
    assert llm_call.cost_usd is None

    finished = trace.events[-1]
    assert isinstance(finished, RunFinished)
    assert finished.usage_complete is False


def test_openai_agents_runner_session_reuses_same_sqlite_session_across_turns(
    tmp_project, tavily_env, monkeypatch
):
    pytest.importorskip("agents")

    from agents import Runner as OaRunner
    from agents.memory import SQLiteSession

    from commonadk.runners.openai_agents import OpenAIAgentsRunner

    seen_sessions: list[SQLiteSession] = []

    def fake_run_streamed(cls, starting_agent, input, session=None, **kw):
        seen_sessions.append(session)
        return _fake_openai_stream_result([], [], "ok")

    monkeypatch.setattr(OaRunner, "run_streamed", classmethod(fake_run_streamed))

    project = load(str(tmp_project))
    runner = OpenAIAgentsRunner()
    session = RunSession()

    runner.run_sync(project, "coordinator", "turn 1", session=session)
    runner.run_sync(project, "coordinator", "turn 2", session=session)

    assert len(seen_sessions) == 2
    assert seen_sessions[0] is seen_sessions[1]
    assert isinstance(seen_sessions[0], SQLiteSession)
    assert session.turns == 2


# ---------------------------------------------------------------------------
# CLI: --trace writes a valid file, --stream prints events, unported target
# gives a clear message. These exercise a REAL failed run (no API key is
# configured anywhere in this offline environment) rather than a mock --
# google-adk's own client raises a clean, fast, offline ValueError before
# any network attempt (verified directly), so the run reliably reaches
# RunError without ever touching the network.
# ---------------------------------------------------------------------------


def test_cli_trace_writes_a_valid_file_even_on_a_failed_run(
    example_common_dir, tavily_env, monkeypatch, tmp_path, capsys
):
    pytest.importorskip("google.adk")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    trace_path = tmp_path / "trace.json"
    rc = cli.main(
        [
            "run",
            str(example_common_dir),
            "--target",
            "google-adk",
            "--trace",
            str(trace_path),
            "research this",
        ]
    )

    assert rc != 0
    assert trace_path.exists()
    data = json.loads(trace_path.read_text())
    kinds = [e["type"] for e in data["events"]]
    assert kinds[0] == "run_started"
    assert "run_error" in kinds
    assert "totals" in data


def test_cli_stream_prints_events_live(
    example_common_dir, tavily_env, monkeypatch, capsys
):
    pytest.importorskip("google.adk")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    rc = cli.main(
        ["run", str(example_common_dir), "--target", "google-adk", "--stream", "research this"]
    )

    assert rc != 0
    out = capsys.readouterr().out
    assert "run_started" in out
    assert "run_error" in out


def test_cli_stream_on_unported_target_gives_clear_message(
    example_common_dir, tavily_env, monkeypatch, capsys
):
    pytest.importorskip("claude_agent_sdk")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    rc = cli.main(
        ["run", str(example_common_dir), "--target", "claude", "--stream", "hi"]
    )

    assert rc != 0
    err = capsys.readouterr().err
    assert "claude" in err
    assert "does not have one yet" in err


def test_cli_trace_on_unported_target_gives_clear_message(
    example_common_dir, tavily_env, monkeypatch, capsys, tmp_path
):
    pytest.importorskip("claude_agent_sdk")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    trace_path = tmp_path / "trace.json"

    rc = cli.main(
        [
            "run",
            str(example_common_dir),
            "--target",
            "claude",
            "--trace",
            str(trace_path),
            "hi",
        ]
    )

    assert rc != 0
    assert not trace_path.exists()
    err = capsys.readouterr().err
    assert "claude" in err


def test_cli_run_without_stream_or_trace_still_just_prints_final_text_on_unported_target(
    example_common_dir, monkeypatch, capsys
):
    """Unchanged default behavior for the four unported targets: no
    --stream/--trace means the original build-and-print path runs exactly
    as before."""
    pytest.importorskip("claude_agent_sdk")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    rc = cli.main(["run", str(example_common_dir), "--target", "claude", "hi"])

    assert rc != 0
    err = capsys.readouterr().err
    assert "ANTHROPIC_API_KEY" in err
