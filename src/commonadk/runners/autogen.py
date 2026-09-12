"""AutoGen runner: drives `AssistantAgent`/`Swarm.run_stream` and normalizes
its async message/event stream into commonadk's runner events.

Verified against the installed packages: autogen-agentchat 0.7.5,
autogen-core 0.7.5, autogen-ext 0.7.5 -- `autogen_agentchat/base/_task.py`
(`TaskRunner.run_stream`, `TaskResult`), `autogen_agentchat/messages.py`
(`BaseChatMessage`/`BaseAgentEvent.source`/`.models_usage`,
`ToolCallRequestEvent`, `ToolCallExecutionEvent`, `HandoffMessage`),
`autogen_core/models/_types.py` (`RequestUsage.prompt_tokens`/
`.completion_tokens`), `autogen_core/_types.py` (`FunctionCall.id`/
`.arguments`/`.name`), `autogen_core/models/_types.py`
(`FunctionExecutionResult.call_id`/`.name`/`.is_error`). See
docs/runner-design.md, "AutoGen" for the mapping table this module
implements, including the verified `RequestUsage` `0`-not-`None` wrinkle
(see "The `0`-not-`None` wrinkle, AutoGen edition" below) this module
inherited from investigating the OpenAI Agents SDK runner's own version of
the same problem.

WHAT `built` IS: `project.build(agent_name, target="autogen")` returns
either a bare `AssistantAgent` (build root has no outgoing edges) or a
`Swarm` (build root has at least one) -- see `adapters/autogen_adapter.py`'s
own module docstring, "WHAT build() RETURNS". Both implement the same
`TaskRunner` protocol (`autogen_agentchat/base/_task.py:19`), so this runner
never branches on which one it got -- `built.run_stream(task=prompt,
output_task_messages=False, ...)` works identically either way.

`output_task_messages=False`: verified via `inspect.signature` that
`run`/`run_stream` accept this kwarg (default `True`, "for backward
compatibility") -- passing `False` keeps the echoed user-task message (a
plain `TextMessage(source="user", ...)`) out of the output stream entirely,
so this runner never has to special-case a message whose `.source` is
`"user"` rather than a real agent name.

Per-message agent attribution -- genuinely finer-grained than either shipped
runner, verified not assumed: EVERY `BaseChatMessage`/`BaseAgentEvent`
carries its own `.source: str` (`messages.py:86`, `:161`) naming the exact
agent that produced it. Unlike Google ADK's author-change heuristic (no
explicit lifecycle boundary in that SDK's `Event` stream) or OpenAI Agents'
explicit-but-coarser `AgentUpdatedStreamEvent`, this runner opens/closes
`AgentStarted`/`AgentFinished` on every `.source` change, which is both a
real per-message signal AND happens to coincide exactly with the boundary a
caller would want.

Tool calls -- clean, direct pairing, verified against
`autogen_agentchat/agents/_assistant_agent.py`: a tool-calling round yields
exactly one `ToolCallRequestEvent` (`content: List[FunctionCall]`, one entry
per parallel tool call the model requested in that round) followed by
exactly one `ToolCallExecutionEvent` (`content: List[FunctionExecutionResult]`)
once every call in that round has executed -- this runner pairs them by
`FunctionCall.id` / `FunctionExecutionResult.call_id` (a `pending_calls`
dict, same pattern as the two shipped runners' own tool-call correlation),
timing the pairing itself the same way both shipped runners do (AutoGen
does not timestamp tool execution either). `FunctionExecutionResult.call_id`
"may be empty for some models" per its own docstring -- when that happens
(or the id was never queued, e.g. a stream that starts mid-call) the lookup
misses and this runner falls back to `result.name` for the tool name and
`current_agent_name` for attribution, exactly like the ADK/OpenAI runners'
own fallback for an unmatched pairing.

Handoffs -- a dedicated message type, no inference needed: `HandoffMessage`
(`messages.py:421`) carries `.source`/`.target` directly -- verified this is
the ONLY built-in AutoGen mechanism this codebase's adapter uses (both
`delegate` and `handoff` edges compile to `AssistantAgent(handoffs=[...])`,
see `autogen_adapter.py`'s own docstring, "Edge mapping") -- mapped straight
to `Transfer(transfer_kind="autogen:handoff")`.

LLM usage -- genuinely per-message, the finest-grained of the four SDKs this
project didn't ship a runner for at issue #8 time, verified by reading
`_assistant_agent.py` directly (not assumed from the message schema alone):
every message-producing model round-trip (a direct text `Response`, a
`ToolCallRequestEvent` opening a tool round, or the post-reflection
`Response`) is constructed with `models_usage=<that round-trip's own
current_model_result.usage>` (`_assistant_agent.py:1159,1170,1186,1481`) --
one `RequestUsage` per actual model call, not per `run_stream()` turn. The
deterministic `ToolCallSummaryMessage` that follows a tool round (built from
`tool_call_summary_format`, no model call involved) is never constructed
with `models_usage` set, so it correctly never produces a spurious `LLMCall`
here -- this runner emits exactly one `LLMCall` per message whose
`.models_usage is not None`, no more, no fewer.

The `0`-not-`None` wrinkle, AutoGen edition -- verified directly against the
installed `autogen_ext` model clients, not assumed from the OpenAI Agents
SDK precedent alone: `autogen_ext.models.openai._openai_client`
(`_openai_client.py:710-712`, "Handle the case where OpenAI API might
return None for token counts even when result.usage is not None") builds
`RequestUsage(prompt_tokens=getattr(result.usage, "prompt_tokens", 0) if
result.usage is not None else 0, completion_tokens=...)` -- i.e. it
defaults BOTH fields to plain `0`, not `None`, whenever the provider's own
response carries no usage at all. This is the exact same shape of bug this
project already fixed once for the OpenAI Agents SDK runner (see
docs/runner-design.md, "OpenAI Agents' `0`-not-`None` wrinkle") -- and it
applies here to every AutoGen agent routed through
`OpenAIChatCompletionClient` (the `openai/...` AND `gemini/...` provider
branches of `autogen_adapter.py`, both use this same client class). The
`anthropic/...` branch (`autogen_ext.models.anthropic._anthropic_client.py:685-688`)
is NOT affected -- it reads `result.usage.input_tokens`/`.output_tokens`
directly off the Anthropic API's own `Message.usage`, a field the real
Anthropic API always populates, verified via the same file. Since this
runner has no reliable way to know, from a bare `RequestUsage` object alone,
which model client produced it (a `Swarm` can mix providers across its
participants), it applies the SAME conservative rule this codebase already
established for the OpenAI Agents SDK runner, uniformly, regardless of
provider: `reported = bool(usage.prompt_tokens or usage.completion_tokens)`
-- an all-zero `RequestUsage` is treated as **unreported** (`None` on every
`LLMCall` token/cost field), never as a confident zero. A genuinely
zero-token completion is not a case any provider this project targets
actually produces, so this reading is the conservative, honest one per this
project's own founding rule (see docs/runner-design.md, "The `None`-vs-`0`
rule") -- it is also, separately, confirmed correct for the `anthropic/...`
branch too: a real Anthropic response cannot report zero prompt tokens for
a non-empty conversation, so the same all-zero check never produces a false
negative there either.

Per-agent model resolution -- a deliberate difference from both shipped
runners' `_resolved_model`, which resolve once for the whole run using only
the build root's name: since AutoGen genuinely attributes every message to
its real producing agent (see above), and a `Swarm`'s participants can each
use a different model/provider (the shipped example does exactly this:
`coordinator`/`writer` on the `fast` alias, `researcher` on
`gemini/gemini-2.5-pro` directly), resolving `model_name`/`cost_usd` once
for the whole run and stamping it onto every `LLMCall` regardless of which
participant actually made the call would misprice every call from any
agent other than the build root. This runner instead resolves the model
PER SOURCE AGENT (`_resolved_model(project, source)`, called with the
message's own `.source`, memoized in a small per-run dict since
`Project.resolve_model` does real work) -- more precise than either shipped
runner needs to be, made possible by AutoGen's own finer-grained attribution.

Not available: per-call duration (`RequestUsage` carries no timing field,
same limitation as both shipped runners); an SDK-native *inline*,
recoverable run error the way Google ADK's `LlmResponse.error_code` is --
nothing in the message stream signals "this happened but the run kept
going" the way ADK's per-`Event` error fields do, so this runner (like the
OpenAI Agents SDK runner) only ever emits the one, fatal `RunError` from
its own top-level `try/except`.

Session/multi-turn: `RunSession.native["autogen"]` holds `{"built": <the
`AssistantAgent`/`Swarm` `project.build()` returned on the first turn>}`.
`TaskRunner.run_stream`'s own docstring (`_task.py:32`) states it "is
stateful and a subsequent call ... will continue from where the previous
call left off" -- so this runner's whole multi-turn story is simply: build
once, cache the object, and call `.run_stream(task=prompt, ...)` again on
turn 2+ against that SAME object, exactly mirroring how the Google ADK
runner caches its bound `InMemoryRunner`. A fresh `RunSession()` (or
`session=None`) rebuilds fresh every call, same as every other runner in
this codebase. One caveat verified only at the level stated here, not
deeper: a `Swarm` this adapter returns has `max_turns=len(reachable)` set
at construction (`autogen_adapter.py`'s own documented heuristic) --
`autogen_agentchat.teams._group_chat._base_group_chat.BaseGroupChat` passes
`self._max_turns` into a freshly-constructed group-chat-manager on each
`run_stream()` call (`_base_group_chat.py:225`, inside the per-call
team-creation path), which reads as a PER-CALL budget, not one that
depletes across the whole session -- not independently reproduced with a
live multi-turn run, so documented as the read taken, not as verified fact.
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


class AutoGenRunner(BaseRunner):
    target = "autogen"

    async def run(
        self,
        project: "Project",
        agent_name: str,
        prompt: str,
        *,
        session: Optional[RunSession] = None,
        hooks: Optional[HookRegistry] = None,
    ) -> Trace:
        from autogen_agentchat.base import TaskResult
        from autogen_agentchat.messages import (
            HandoffMessage,
            ToolCallExecutionEvent,
            ToolCallRequestEvent,
        )

        trace = Trace()
        run_id = _uuid.uuid4().hex
        t0 = time.monotonic()
        model_cache: dict[str, Optional[str]] = {}

        # Same reasoning as the Google ADK / OpenAI Agents runners: a build
        # failure (missing provider API key, unbuildable graph, the
        # `anthropic>=1` guard in autogen_adapter.py) means the run never
        # started at all, so this stays outside the try/except below.
        state = session.native.setdefault(self.target, {}) if session is not None else {}
        built = state.get("built")
        if built is None:
            built = project.build(agent_name, target=self.target)
            if session is not None:
                state["built"] = built

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

        current_agent_name: Optional[str] = None
        pending_calls: dict[str, dict[str, Any]] = {}
        final_text: Optional[str] = None

        try:
            async for item in built.run_stream(task=prompt, output_task_messages=False):
                if isinstance(item, TaskResult):
                    final_text = self._final_text(item)
                    continue

                source = getattr(item, "source", None)
                if source and source != current_agent_name:
                    if current_agent_name is not None:
                        self._emit(
                            trace, hooks, AgentFinished(run_id=run_id, agent_name=current_agent_name)
                        )
                    current_agent_name = source
                    self._emit(
                        trace, hooks, AgentStarted(run_id=run_id, agent_name=current_agent_name)
                    )

                if isinstance(item, ToolCallRequestEvent):
                    for call in item.content:
                        pending_calls[call.id] = {
                            "name": call.name,
                            "arguments": self._parse_arguments(call.arguments),
                            "start": time.monotonic(),
                            "agent": source,
                        }
                elif isinstance(item, ToolCallExecutionEvent):
                    for result in item.content:
                        started = pending_calls.pop(result.call_id, None) if result.call_id else None
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
                                agent_name=(started or {}).get("agent") or current_agent_name or agent_name,
                                tool_name=(started or {}).get("name") or result.name or "<unknown>",
                                arguments=(started or {}).get("arguments"),
                                result_summary=self._summarize(result.content),
                                duration_ms=duration_ms,
                                error=result.content if result.is_error else None,
                            ),
                        )
                elif isinstance(item, HandoffMessage):
                    self._emit(
                        trace,
                        hooks,
                        Transfer(
                            run_id=run_id,
                            from_agent=item.source,
                            to_agent=item.target,
                            transfer_kind="autogen:handoff",
                        ),
                    )

                usage = getattr(item, "models_usage", None)
                if usage is not None:
                    # See module docstring, "The 0-not-None wrinkle, AutoGen
                    # edition" -- an all-zero RequestUsage is a client-side
                    # "no usage reported" default, not a confident zero.
                    reported = bool(usage.prompt_tokens or usage.completion_tokens)
                    call_agent = source or current_agent_name
                    model_name = self._model_for(project, call_agent, model_cache)
                    prompt_tokens = usage.prompt_tokens if reported else None
                    completion_tokens = usage.completion_tokens if reported else None
                    self._emit(
                        trace,
                        hooks,
                        LLMCall(
                            run_id=run_id,
                            agent_name=call_agent,
                            model=model_name,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=(
                                prompt_tokens + completion_tokens if reported else None
                            ),
                            cost_usd=estimate_cost_usd(model_name, prompt_tokens, completion_tokens),
                            duration_ms=None,  # RequestUsage carries no timing -- see module docstring
                        ),
                    )

            if current_agent_name is not None:
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
    def _final_text(result: Any) -> Optional[str]:
        """Mirrors `cli.py`'s `_run_autogen`: the content of the last
        message in the `TaskResult`, or `None` for a run that produced no
        messages at all (the task's own echo is suppressed via
        `output_task_messages=False`, so an empty list is possible only for
        a genuinely empty run)."""
        if not result.messages:
            return None
        return str(result.messages[-1].content)

    @staticmethod
    def _model_for(
        project: "Project", agent_name: Optional[str], cache: dict[str, Optional[str]]
    ) -> Optional[str]:
        """Resolve `agent_name`'s model, memoized per `run()` call -- see
        module docstring, "Per-agent model resolution" for why this is
        computed per message source rather than once for the whole run."""
        if agent_name is None:
            return None
        if agent_name in cache:
            return cache[agent_name]

        resolved: Optional[str] = None
        spec = project.agents.get(agent_name)
        if spec is not None:
            override = spec.config.targets.get("autogen", {})
            if "model" in override:
                resolved = str(override["model"])
            else:
                try:
                    resolved = project.resolve_model(agent_name)
                except ValueError:
                    resolved = None
        cache[agent_name] = resolved
        return resolved

    @staticmethod
    def _parse_arguments(raw: Any) -> Optional[dict[str, Any]]:
        if not isinstance(raw, str):
            return raw if isinstance(raw, dict) else None
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {"_raw": raw}
        return parsed if isinstance(parsed, dict) else {"_raw": raw}

    @staticmethod
    def _summarize(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value)
        return text if len(text) <= 500 else text[:500] + "..."
