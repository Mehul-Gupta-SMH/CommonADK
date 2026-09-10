"""Google ADK runner: drives `google.adk.runners.Runner`/`InMemoryRunner`
and normalizes its async `Event` stream into commonadk's runner events.

Verified against the installed package: google-adk 2.7.1 --
`google/adk/runners.py` (`Runner.run_async`, `InMemoryRunner`),
`google/adk/events/event.py` (`Event.author`, `.get_function_calls()`,
`.get_function_responses()`, `.is_final_response()`),
`google/adk/models/llm_response.py` (`LlmResponse.usage_metadata`,
`.error_code`, `.error_message`), `google/adk/events/event_actions.py`
(`EventActions.transfer_to_agent`), `google/genai/types.py`
(`GenerateContentResponseUsageMetadata`, `FunctionCall`, `FunctionResponse`).
See docs/runner-design.md, "Google ADK" for the full event-by-event table
this module implements, including what's genuinely not available
(per-call duration).
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


class GoogleADKRunner(BaseRunner):
    target = "google-adk"

    async def run(
        self,
        project: "Project",
        agent_name: str,
        prompt: str,
        *,
        session: Optional[RunSession] = None,
        hooks: Optional[HookRegistry] = None,
    ) -> Trace:
        from google.adk.runners import InMemoryRunner
        from google.genai import types as genai_types

        trace = Trace()
        run_id = _uuid.uuid4().hex
        t0 = time.monotonic()
        model_name = self._resolved_model(project, agent_name)

        # project.build(...) runs BaseAdapter._check_env up front and can
        # raise OSError/ValueError before anything is constructed -- left
        # outside the try/except below so that failure surfaces exactly
        # like it always has (a clean exception, no trace, no RunStarted
        # for a run that never actually started) rather than being
        # reframed as a run that "started and then errored".
        agent = project.build(agent_name, target="google-adk")

        state = session.native.setdefault(self.target, {}) if session is not None else {}
        adk_runner = state.get("runner")
        if adk_runner is None:
            adk_runner = InMemoryRunner(agent=agent, app_name=project.config.name)
            if session is not None:
                state["runner"] = adk_runner
        user_id = state.setdefault("user_id", "commonadk-runner")

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

        current_author: Optional[str] = None
        final_text: Optional[str] = None

        try:
            adk_session_id = state.get("adk_session_id")
            if adk_session_id is None:
                adk_session = await adk_runner.session_service.create_session(
                    app_name=adk_runner.app_name, user_id=user_id
                )
                adk_session_id = adk_session.id
                if session is not None:
                    state["adk_session_id"] = adk_session_id

            message = genai_types.Content(
                role="user", parts=[genai_types.Part(text=prompt)]
            )
            pending_calls: dict[str, dict[str, Any]] = {}
            text_chunks: list[str] = []

            async for event in adk_runner.run_async(
                user_id=user_id, session_id=adk_session_id, new_message=message
            ):
                author = event.author or current_author or agent_name
                if author != current_author:
                    if current_author is not None:
                        self._emit(
                            trace, hooks, AgentFinished(run_id=run_id, agent_name=current_author)
                        )
                    current_author = author
                    self._emit(
                        trace, hooks, AgentStarted(run_id=run_id, agent_name=current_author)
                    )

                for call in event.get_function_calls():
                    key = call.id or call.name or f"<anon-{len(pending_calls)}>"
                    pending_calls[key] = {
                        "name": call.name,
                        "arguments": dict(call.args or {}),
                        "start": time.monotonic(),
                    }

                for response in event.get_function_responses():
                    key = response.id or response.name
                    started = pending_calls.pop(key, None) if key else None
                    duration_ms = (
                        (time.monotonic() - started["start"]) * 1000.0
                        if started is not None
                        else None
                    )
                    payload = response.response or {}
                    error = payload.get("error") if isinstance(payload, dict) else None
                    self._emit(
                        trace,
                        hooks,
                        ToolCall(
                            run_id=run_id,
                            agent_name=current_author or agent_name,
                            tool_name=response.name or (started or {}).get("name") or "<unknown>",
                            arguments=(started or {}).get("arguments"),
                            result_summary=self._summarize(payload),
                            duration_ms=duration_ms,
                            error=str(error) if error else None,
                        ),
                    )

                if event.actions is not None and event.actions.transfer_to_agent:
                    self._emit(
                        trace,
                        hooks,
                        Transfer(
                            run_id=run_id,
                            from_agent=current_author or agent_name,
                            to_agent=event.actions.transfer_to_agent,
                            transfer_kind="google-adk:transfer_to_agent",
                        ),
                    )

                if event.usage_metadata is not None:
                    usage = event.usage_metadata
                    self._emit(
                        trace,
                        hooks,
                        LLMCall(
                            run_id=run_id,
                            agent_name=current_author,
                            model=model_name,
                            prompt_tokens=usage.prompt_token_count,
                            completion_tokens=usage.candidates_token_count,
                            total_tokens=usage.total_token_count,
                            cost_usd=estimate_cost_usd(
                                model_name,
                                usage.prompt_token_count,
                                usage.candidates_token_count,
                            ),
                            duration_ms=None,  # not reported by ADK's Event -- see design doc
                        ),
                    )

                if event.error_code or event.error_message:
                    # Inline, non-fatal -- the ADK run loop kept going after
                    # this; see design doc "Fatal vs. inline RunError".
                    self._emit(
                        trace,
                        hooks,
                        RunError(
                            run_id=run_id,
                            message=str(event.error_message or event.error_code),
                            error_type="google-adk:LlmResponse.error",
                        ),
                    )

                if event.is_final_response() and event.content and event.content.parts:
                    text_chunks.extend(
                        part.text for part in event.content.parts if getattr(part, "text", None)
                    )

            if current_author is not None:
                self._emit(trace, hooks, AgentFinished(run_id=run_id, agent_name=current_author))

            final_text = "\n".join(text_chunks)
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
        override = spec.config.targets.get("google-adk", {})
        if "model" in override:
            return str(override["model"])
        try:
            return project.resolve_model(agent_name)
        except ValueError:
            return None

    @staticmethod
    def _summarize(value: Any) -> Optional[str]:
        text = str(value)
        return text if len(text) <= 500 else text[:500] + "..."
