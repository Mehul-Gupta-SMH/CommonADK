"""CrewAI runner: drives `crewai.Crew.kickoff_async` and normalizes CrewAI's
own `crewai.events` pub/sub bus into commonadk's runner events.

Verified against the installed package: crewai 1.15.16. The pre-implementation
plan in docs/runner-design.md ("CrewAI" row) predates this file and gets two
things wrong, both corrected here after reading the installed package's
actual classes (not memory) -- see "Corrections versus the original plan"
below. This module is named `crewai_runner.py`, not `crewai.py`, purely for
readability at a glance in `_REGISTRY`/directory listings next to
`google_adk.py`/`openai_agents.py`/`claude_agent.py` -- Python 3's absolute
import semantics mean a module named `crewai.py` importing the top-level
`crewai` package would resolve correctly regardless (this module's own
fully-qualified name is `commonadk.runners.crewai`, not `crewai`), so this
is a naming-clarity choice, not a workaround for a real import hazard.

WHY THIS RUNNER READS FROM AN EVENT BUS, UNLIKE THE OTHER THREE: CrewAI's
`kickoff()`/`kickoff_async()` return only a single terminal `CrewOutput` --
there is no async generator of steps to drive, unlike Google ADK's
`run_async`, OpenAI Agents' `stream_events()`, or the Claude Agent SDK's
`receive_response()`. But CrewAI ships its own internal pub/sub bus,
`crewai.events.crewai_event_bus` (`crewai/events/event_bus.py`), which every
one of CrewAI's own subsystems (agents, tools, LLM calls) already publishes
typed events onto -- `AgentExecutionStartedEvent`/`Completed`/`Error`
(`crewai/events/types/agent_events.py`), `ToolUsageStartedEvent`/
`FinishedEvent`/`ErrorEvent` (`crewai/events/types/tool_usage_events.py`),
and `LLMCallStartedEvent`/`CompletedEvent`/`FailedEvent`
(`crewai/events/types/llm_events.py`). Subscribing to this bus for the
duration of one `kickoff_async()` call gives this runner a genuine event
stream to normalize, the same shape of problem the other three runners
solve, just via a different native mechanism.

Corrections versus the original plan:

1. The plan said CrewAI has "no per-agent or per-call breakdown" and would
   need one crew-wide `LLMCall` per `run()` call, and that delegation "has
   no observable event ... likely needs the same callback mechanism as tool
   calls" (flagged as unconfirmed). Both are wrong: `LLMCallCompletedEvent`
   carries per-call `usage`/`model`/`agent_role`, and CrewAI's own
   delegation mechanism (`crewai/tools/agent_tools/delegate_work_tool.py`,
   `ask_question_tool.py` -- literally named `"Delegate work to coworker"`/
   `"Ask question to coworker"`, verified via `inspect.getsource`) is
   implemented as an ordinary tool call, fully visible on the same
   `ToolUsageFinishedEvent` every other tool call produces, with a
   `coworker` key in `tool_args` naming the delegation target directly --
   no callback plumbing needed at all. This runner never touches
   `Crew.step_callback`/`task_callback`, the mechanism the plan flagged as
   "the next thing to investigate."
2. `ToolUsageFinishedEvent` carries real `started_at`/`finished_at`
   `datetime`s from the SDK itself -- `ToolCall.duration_ms` here is
   genuinely SDK-reported, not self-timed the way every other runner in
   this codebase has to (see each of their docstrings' "we time it
   ourselves" caveats).

THE GLOBAL-BUS CAVEAT (the one real limitation this design has that the
other three don't): `crewai_event_bus` is a process-wide singleton, not
scoped to one `Crew`/one `kickoff()` call -- two concurrent `run()` calls in
the same process (two turns of two different `RunSession`s running
concurrently, say) would both receive every event the other's crew emits if
this runner didn't filter. It does: every handler checks the emitting
agent's id against `{str(a.id) for a in crew.agents} | {str(crew.
manager_agent.id)}` for *this* `run()` call's own `Crew` (built fresh by
`project.build(...)` every call, so its `Agent` objects' ids are unique per
call), so a foreign run's events are silently ignored rather than
cross-contaminating this trace. Handlers are registered right before
`kickoff_async()` and unregistered in a `finally` immediately after, so no
handler outlives its own `run()` call.

THE THREAD-CROSSING CAVEAT: `crewai_event_bus.emit()` dispatches sync
handlers on its own internal `ThreadPoolExecutor` (`max_workers=10`, see
`crewai/events/event_bus.py`), not necessarily the thread `kickoff_async()`
itself runs on (which is already a `asyncio.to_thread` worker, since
`Crew.kickoff_async` is `await asyncio.to_thread(self.kickoff, ...)`). So
this runner's handlers run on a bus-owned worker thread, appending to
`trace.events`/firing `hooks` from there -- safe under the GIL for the
simple `list.append`/callback-invocation this project's `HookRegistry` and
`Trace` do, but worth stating plainly: a caller's own hook callback is not
guaranteed to run on the event loop's thread for this one target.

A direct, tested consequence of `max_workers=10`: this runner's `Event.seq`
ordering reflects the order handlers actually *ran* on that pool, not the
order CrewAI's own executor produced the underlying events in -- two
events of *different* types (say a `ToolUsageFinishedEvent` and an
`LLMCallCompletedEvent` emitted moments apart) can legitimately race and
land in either order in the trace. Every event's own fields (which agent,
which tool, how many tokens) are still exactly right regardless -- only the
*relative order* of unrelated event kinds in one run's trace is not a
promise this runner can make, unlike the other three runners (whose native
async generators/streams are strictly ordered by construction). Because
handler dispatch is itself async relative to `kickoff()` returning, this
runner calls `crewai_event_bus.flush()` (a real SDK method built exactly
for this: "block until all pending event handlers complete") before
finalizing the trace, so `RunFinished`'s rollup totals are never computed
while a `LLMCallCompletedEvent` handler is still in flight.

NO SESSION STORY (v1): CrewAI's own multi-turn primitive doesn't exist for
this runner's chosen shape (a fresh `Crew` + a single `Task` per `run()`
call, matching `crewai_adapter.py`'s own "no persistent task list" design) --
`crewai_adapter.py`'s docstring confirms `Crew(..., tasks=[])` is deliberate
because the actual task only exists at run time. Per `base.py`'s documented
session contract, this runner raises `NotImplementedError` naming the gap
the moment a non-`None` `session` is passed, rather than silently starting
an unrelated crew and pretending it continued a conversation.
"""

from __future__ import annotations

import time
import uuid as _uuid
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ..models import Project

from .base import BaseRunner, RunSession
from .events import (
    AgentFinished,
    AgentStarted,
    LLMCall,
    RunError,
    RunFinished,
    RunStarted,
    ToolCall,
    Transfer,
)
from .hooks import HookRegistry
from .pricing import estimate_cost_usd
from .trace import Trace

# The exact tool names CrewAI's own delegation tools register under --
# verified via `inspect.getsource(crewai.tools.agent_tools.delegate_work_tool)`
# / `.ask_question_tool` against the installed package, not guessed.
_DELEGATION_TOOL_NAMES = frozenset({"Delegate work to coworker", "Ask question to coworker"})


class CrewAIRunner(BaseRunner):
    target = "crewai"

    async def run(
        self,
        project: "Project",
        agent_name: str,
        prompt: str,
        *,
        session: Optional[RunSession] = None,
        hooks: Optional[HookRegistry] = None,
    ) -> Trace:
        if session is not None:
            # See module docstring, "NO SESSION STORY (v1)" -- raised before
            # anything is built, mirroring every other precondition failure
            # in this codebase: a run that can't even start emits no trace.
            raise NotImplementedError(
                "commonadk: the CrewAI runner has no multi-turn session "
                "story yet -- crewai_adapter.py's `Crew` is built fresh per "
                "run() call with no persistent task list, and CrewAI itself "
                "has no first-class 'continue this exact conversation' "
                "primitive analogous to Google ADK's session service or "
                "OpenAI Agents' SQLiteSession. Pass session=None. See "
                "docs/runner-design.md, 'CrewAI', 'NO SESSION STORY (v1)'."
            )

        import asyncio

        from crewai import Process, Task
        from crewai.events import crewai_event_bus
        from crewai.events.types.agent_events import (
            AgentExecutionCompletedEvent,
            AgentExecutionErrorEvent,
            AgentExecutionStartedEvent,
        )
        from crewai.events.types.llm_events import LLMCallCompletedEvent, LLMCallFailedEvent
        from crewai.events.types.tool_usage_events import (
            ToolUsageErrorEvent,
            ToolUsageFinishedEvent,
        )
        from crewai.types.usage_metrics import UsageMetrics

        trace = Trace()
        run_id = _uuid.uuid4().hex
        t0 = time.monotonic()
        model_name = self._resolved_model(project, agent_name)

        # Left outside try/except for the same reason as every other
        # runner: a build failure means the run never actually started.
        crew = project.build(agent_name, target="crewai")

        relevant_agent_ids = {str(a.id) for a in crew.agents}
        if crew.manager_agent is not None:
            relevant_agent_ids.add(str(crew.manager_agent.id))

        def is_relevant(agent_id: Optional[str]) -> bool:
            return agent_id is not None and agent_id in relevant_agent_ids

        self._emit(
            trace,
            hooks,
            RunStarted(
                run_id=run_id,
                target=self.target,
                agent_name=agent_name,
                prompt=prompt,
                session_id=None,
            ),
        )

        # -- event bus handlers (see module docstring, "GLOBAL-BUS CAVEAT"
        # and "THREAD-CROSSING CAVEAT" for why every handler filters by
        # agent id and why this all ends in a flush()) ---------------------

        def on_agent_started(source: Any, event: Any) -> None:
            if str(event.agent.id) not in relevant_agent_ids:
                return
            self._emit(trace, hooks, AgentStarted(run_id=run_id, agent_name=event.agent.role))

        def on_agent_completed(source: Any, event: Any) -> None:
            if str(event.agent.id) not in relevant_agent_ids:
                return
            self._emit(
                trace,
                hooks,
                AgentFinished(
                    run_id=run_id,
                    agent_name=event.agent.role,
                    output_summary=self._summarize(event.output),
                ),
            )

        def on_agent_error(source: Any, event: Any) -> None:
            if str(event.agent.id) not in relevant_agent_ids:
                return
            # Inline, non-fatal -- see google_adk.py's identical pattern:
            # CrewAI's own executor may retry/recover after this, so this
            # is not necessarily the run's terminal outcome.
            self._emit(
                trace,
                hooks,
                RunError(
                    run_id=run_id,
                    message=str(event.error),
                    error_type="crewai:AgentExecutionErrorEvent",
                ),
            )

        def on_tool_finished(source: Any, event: Any) -> None:
            if not is_relevant(event.agent_id):
                return
            duration_ms = (event.finished_at - event.started_at).total_seconds() * 1000.0
            args = event.tool_args if isinstance(event.tool_args, dict) else None
            attributed_agent = event.agent_role or agent_name
            self._emit(
                trace,
                hooks,
                ToolCall(
                    run_id=run_id,
                    agent_name=attributed_agent,
                    tool_name=event.tool_name,
                    arguments=args,
                    result_summary=self._summarize(event.output),
                    duration_ms=duration_ms,
                    error=str(event.failure) if event.failure else None,
                ),
            )
            coworker = args.get("coworker") if args else None
            if event.tool_name in _DELEGATION_TOOL_NAMES and coworker:
                self._emit(
                    trace,
                    hooks,
                    Transfer(
                        run_id=run_id,
                        from_agent=attributed_agent,
                        to_agent=str(coworker),
                        transfer_kind=f"crewai:{event.tool_name}",
                    ),
                )

        def on_tool_error(source: Any, event: Any) -> None:
            if not is_relevant(event.agent_id):
                return
            args = event.tool_args if isinstance(event.tool_args, dict) else None
            self._emit(
                trace,
                hooks,
                ToolCall(
                    run_id=run_id,
                    agent_name=event.agent_role or agent_name,
                    tool_name=event.tool_name,
                    arguments=args,
                    result_summary=None,
                    duration_ms=None,  # no started_at on the error event -- see design doc
                    error=str(event.error),
                ),
            )

        def on_llm_completed(source: Any, event: Any) -> None:
            if not is_relevant(event.agent_id):
                return
            # `event.usage` is `dict[str, Any] | None`, the SAME raw
            # provider payload `BaseLLM._track_token_usage_internal` feeds
            # into `UsageMetrics.from_provider_dict` for the crew-wide
            # accumulator -- reusing that exact normalizer here (rather
            # than hand-rolling key aliasing across Anthropic/Gemini/OpenAI
            # shapes) means this runner's per-call numbers agree with
            # CrewAI's own crew-wide `CrewOutput.token_usage` by
            # construction. `usage=None` or an all-zero result is treated
            # as unreported -- the same conservative "0-not-None" reading
            # `openai_agents.py` settled on (see its docstring), applied
            # here because `UsageMetrics`' fields are plain `int`s
            # defaulting to `0`, not `Optional[int]`, so `0` cannot be
            # told apart from "not reported" at this layer either.
            metrics = UsageMetrics.from_provider_dict(event.usage) if event.usage else None
            reported = metrics is not None and bool(
                metrics.total_tokens or metrics.prompt_tokens or metrics.completion_tokens
            )
            prompt_tokens = metrics.prompt_tokens if reported and metrics else None
            completion_tokens = metrics.completion_tokens if reported and metrics else None
            total_tokens = metrics.total_tokens if reported and metrics else None
            model = event.model or model_name
            self._emit(
                trace,
                hooks,
                LLMCall(
                    run_id=run_id,
                    agent_name=event.agent_role or agent_name,
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cost_usd=estimate_cost_usd(model, prompt_tokens, completion_tokens),
                    duration_ms=None,  # LLMCallCompletedEvent carries no timing field
                ),
            )

        def on_llm_failed(source: Any, event: Any) -> None:
            if not is_relevant(event.agent_id):
                return
            self._emit(
                trace,
                hooks,
                RunError(
                    run_id=run_id,
                    message=str(event.error),
                    error_type="crewai:LLMCallFailedEvent",
                ),
            )

        handlers: list[tuple[type, Any]] = [
            (AgentExecutionStartedEvent, on_agent_started),
            (AgentExecutionCompletedEvent, on_agent_completed),
            (AgentExecutionErrorEvent, on_agent_error),
            (ToolUsageFinishedEvent, on_tool_finished),
            (ToolUsageErrorEvent, on_tool_error),
            (LLMCallCompletedEvent, on_llm_completed),
            (LLMCallFailedEvent, on_llm_failed),
        ]
        for event_type, handler in handlers:
            crewai_event_bus.on(event_type)(handler)

        final_text: Optional[str] = None
        try:
            try:
                # Hierarchical crews (build root has outgoing edges): leave
                # the task unassigned so the manager picks the crew member;
                # sequential crews here are always the solo-member fallback
                # (see crewai_adapter.py's docstring, "Manager-or-solo-member
                # decision") -- same pattern as cli.py's `_run_crewai`.
                task_agent = None if crew.process == Process.hierarchical else crew.agents[0]
                crew.tasks = [
                    Task(
                        description=prompt,
                        expected_output="A complete response to the request above.",
                        agent=task_agent,
                    )
                ]
                result = await crew.kickoff_async()
                # See module docstring, "THREAD-CROSSING CAVEAT" -- blocks
                # (off the event loop thread) until every handler fired
                # above has actually finished running.
                await asyncio.to_thread(crewai_event_bus.flush, 30.0)
            finally:
                for event_type, handler in handlers:
                    crewai_event_bus.off(event_type, handler)

            final_text = str(result.raw)

        except Exception as exc:
            self._emit(
                trace,
                hooks,
                RunError(run_id=run_id, message=str(exc), error_type=type(exc).__name__),
            )
            raise

        llm_totals = trace.rollup()["llm_calls"]
        self._emit(
            trace,
            hooks,
            RunFinished(
                run_id=run_id,
                final_text=final_text,
                total_prompt_tokens=llm_totals["prompt_tokens"],
                total_completion_tokens=llm_totals["completion_tokens"],
                total_tokens=llm_totals["total_tokens"],
                total_cost_usd=llm_totals["cost_usd"],
                duration_ms=(time.monotonic() - t0) * 1000.0,
                usage_complete=llm_totals["usage_complete"],
            ),
        )
        return trace

    @staticmethod
    def _resolved_model(project: "Project", agent_name: str) -> Optional[str]:
        spec = project.agents.get(agent_name)
        if spec is None:
            return None
        override = spec.config.targets.get("crewai", {})
        if "model" in override:
            return str(override["model"])
        try:
            return project.resolve_model(agent_name)
        except ValueError:
            return None

    @staticmethod
    def _summarize(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value)
        return text if len(text) <= 500 else text[:500] + "..."


__all__ = ["CrewAIRunner"]
