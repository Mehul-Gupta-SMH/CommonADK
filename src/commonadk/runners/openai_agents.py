"""OpenAI Agents SDK runner: drives `agents.Runner.run_streamed` and
normalizes its stream-event / result surface into commonadk's runner events.

Verified against the installed package: openai-agents 0.21.1 --
`agents/run.py` (`Runner.run_streamed`), `agents/result.py`
(`RunResultStreaming.stream_events()`, `.raw_responses`, `.final_output`),
`agents/stream_events.py` (`AgentUpdatedStreamEvent`, `RunItemStreamEvent`,
`RawResponsesStreamEvent`), `agents/items.py` (`RunItemBase.agent`,
`ToolCallItem`, `ToolCallOutputItem`, `HandoffOutputItem`, `ModelResponse`),
`agents/usage.py` (`Usage.requests`/`.input_tokens`/`.output_tokens`/
`.total_tokens`), `agents/run_internal/run_loop.py` (`_requests_for_response_without_usage`,
the "request completed without usage" path a LiteLLM-bridged call actually
takes), `agents/memory/sqlite_session.py` (`SQLiteSession`). See
docs/runner-design.md, "OpenAI Agents SDK" for the full mapping table,
including the `0`-not-`None` usage wrinkle (corrected after a live run
exposed the `requests > 0` heuristic reporting a confident `0`/$0.000000`
for usage the SDK never actually reported) and the documented multi-agent
LLMCall-attribution gap this module implements exactly as described there.
"""

from __future__ import annotations

import json
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


class OpenAIAgentsRunner(BaseRunner):
    target = "openai"

    async def run(
        self,
        project: "Project",
        agent_name: str,
        prompt: str,
        *,
        session: Optional[RunSession] = None,
        hooks: Optional[HookRegistry] = None,
    ) -> Trace:
        from agents import Runner, SQLiteSession
        from agents.stream_events import AgentUpdatedStreamEvent, RunItemStreamEvent

        trace = Trace()
        run_id = _uuid.uuid4().hex
        t0 = time.monotonic()
        model_name = self._resolved_model(project, agent_name)

        # Left outside try/except for the same reason as the Google ADK
        # runner: a build failure (missing env var, unbuildable graph)
        # means the run never started at all -- see that module's comment.
        starting_agent = project.build(agent_name, target="openai")

        native_session = None
        if session is not None:
            state = session.native.setdefault(self.target, {})
            native_session = state.get("sqlite_session")
            if native_session is None:
                native_session = SQLiteSession(session_id=session.session_id)
                state["sqlite_session"] = native_session

        self._emit(
            trace,
            hooks,
            RunStarted(
                run_id=run_id,
                target=self.target,
                agent_name=agent_name,
                prompt=prompt,
                session_id=session.session_id if session is not None else None,
            ),
        )

        current_agent_name = starting_agent.name
        multi_agent = False
        pending_tools: dict[Any, dict[str, Any]] = {}
        final_text: Optional[str] = None

        try:
            self._emit(trace, hooks, AgentStarted(run_id=run_id, agent_name=current_agent_name))

            result = Runner.run_streamed(starting_agent, prompt, session=native_session)

            async for stream_event in result.stream_events():
                if isinstance(stream_event, AgentUpdatedStreamEvent):
                    new_name = stream_event.new_agent.name
                    if new_name != current_agent_name:
                        multi_agent = True
                        self._emit(
                            trace, hooks, AgentFinished(run_id=run_id, agent_name=current_agent_name)
                        )
                        current_agent_name = new_name
                        self._emit(
                            trace, hooks, AgentStarted(run_id=run_id, agent_name=current_agent_name)
                        )
                    continue

                if not isinstance(stream_event, RunItemStreamEvent):
                    # raw_response_event: token-level deltas, not mapped in
                    # v1 -- see design doc.
                    continue

                item = stream_event.item
                item_agent_name = getattr(getattr(item, "agent", None), "name", current_agent_name)

                if stream_event.name == "tool_called":
                    call_id = getattr(item, "call_id", None)
                    key = call_id if call_id is not None else id(item)
                    pending_tools[key] = {
                        "name": getattr(item, "tool_name", None) or "<unknown>",
                        "arguments": self._parse_arguments(item),
                        "start": time.monotonic(),
                        "agent": item_agent_name,
                    }
                elif stream_event.name == "tool_output":
                    call_id = self._output_call_id(item)
                    started = pending_tools.pop(call_id, None) if call_id is not None else None
                    duration_ms = (
                        (time.monotonic() - started["start"]) * 1000.0
                        if started is not None
                        else None
                    )
                    self._emit(
                        trace,
                        hooks,
                        ToolCall(
                            run_id=run_id,
                            agent_name=(started or {}).get("agent", item_agent_name),
                            tool_name=(started or {}).get("name") or "<unknown>",
                            arguments=(started or {}).get("arguments"),
                            result_summary=self._summarize(getattr(item, "output", None)),
                            duration_ms=duration_ms,
                            error=None,
                        ),
                    )
                elif stream_event.name == "handoff_occured":
                    source = getattr(item, "source_agent", None)
                    target = getattr(item, "target_agent", None)
                    if source is not None and target is not None:
                        self._emit(
                            trace,
                            hooks,
                            Transfer(
                                run_id=run_id,
                                from_agent=source.name,
                                to_agent=target.name,
                                transfer_kind="openai-agents:handoff",
                            ),
                        )
                # message_output_created / reasoning_item_created / mcp_* /
                # compaction / tool_approval / tool_search_*: no normalized
                # equivalent in v1 -- see design doc.

            final_text = str(result.final_output)

            for raw_response in result.raw_responses:
                usage = raw_response.usage
                reported = bool(
                    usage.input_tokens or usage.output_tokens or usage.total_tokens
                )
                attributed_agent = current_agent_name if not multi_agent else None
                attributed_model = model_name if not multi_agent else None
                prompt_tokens = usage.input_tokens if reported else None
                completion_tokens = usage.output_tokens if reported else None
                self._emit(
                    trace,
                    hooks,
                    LLMCall(
                        run_id=run_id,
                        agent_name=attributed_agent,
                        model=attributed_model,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=usage.total_tokens if reported else None,
                        cost_usd=estimate_cost_usd(attributed_model, prompt_tokens, completion_tokens),
                        duration_ms=None,  # ModelResponse carries no per-call timing
                    ),
                )

            self._emit(trace, hooks, AgentFinished(run_id=run_id, agent_name=current_agent_name))

            if session is not None:
                session.turns += 1

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
        override = spec.config.targets.get("openai", {})
        if "model" in override:
            return str(override["model"])
        try:
            return project.resolve_model(agent_name)
        except ValueError:
            return None

    @staticmethod
    def _parse_arguments(item: Any) -> Optional[dict[str, Any]]:
        raw = getattr(item, "raw_item", None)
        arguments = raw.get("arguments") if isinstance(raw, dict) else getattr(raw, "arguments", None)
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except (json.JSONDecodeError, TypeError):
                return {"_raw": arguments}
            return parsed if isinstance(parsed, dict) else {"_raw": arguments}
        return arguments if isinstance(arguments, dict) else None

    @staticmethod
    def _output_call_id(item: Any) -> Any:
        raw = getattr(item, "raw_item", None)
        if isinstance(raw, dict):
            return raw.get("call_id") or raw.get("id")
        return getattr(raw, "call_id", None) or getattr(raw, "id", None)

    @staticmethod
    def _summarize(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value)
        return text if len(text) <= 500 else text[:500] + "..."
