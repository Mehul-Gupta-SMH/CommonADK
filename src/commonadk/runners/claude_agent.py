"""Claude Agent SDK runner: drives `claude_agent_sdk.ClaudeSDKClient` and
normalizes its `AsyncIterator[Message]` (from `.receive_response()`) into
commonadk's runner events.

Verified against the installed package: claude-agent-sdk 0.2.144 --
`claude_agent_sdk/client.py` (`ClaudeSDKClient.connect`/`.query`/
`.receive_response`/`.disconnect`), `claude_agent_sdk/_internal/message_parser.py`
(exactly which JSON keys populate each dataclass field -- read directly,
not assumed from `types.py`'s annotations alone), `claude_agent_sdk/types.py`
(`AssistantMessage`, `UserMessage`, `ResultMessage`, `TextBlock`,
`ThinkingBlock`, `ToolUseBlock`, `ToolResultBlock`, `ServerToolUseBlock`,
`ServerToolResultBlock`, `ModelUsage`). See docs/runner-design.md, "Claude
Agent SDK" for the full mapping table and two corrections this
implementation makes to that table's original (pre-implementation) plan:

1. `AssistantMessage.usage` IS populated per-message (`message_parser.py`
   line 215: `usage=data["message"].get("usage")`) -- the original plan's
   claim that Claude only reports usage "at the end of the whole turn" is
   not quite right. This runner still emits exactly **one** `LLMCall` per
   `run()` call anyway -- see "Why one coarse LLMCall, not several" in the
   design doc for why splitting tokens finer than the SDK's own per-call
   *cost* granularity (turn-wide only, from `ResultMessage`) would produce
   a trace where `Trace.rollup()`'s cost/token totals disagree with each
   other, which is worse than one honestly coarse number.
2. Subagent invocation genuinely is observable as `AssistantMessage.
   parent_tool_use_id` correlating back to the `ToolUseBlock.id` of the
   `"Agent"` tool call that spawned it (verified in `message_parser.py`,
   lines 98/135/142/213/349) -- this runner uses that correlation for
   `AgentStarted`/`AgentFinished` boundaries and `Transfer`, a real SDK
   signal, not a guess.

`RunSession` support: unlike the two shipped runners, this one *does* use
`ClaudeSDKClient` (not the one-shot `query()` function) for the reason the
original design doc note already gave -- `ClaudeSDKClient` keeps one
subprocess/connection alive across `.query()` calls, which is this SDK's
actual multi-turn primitive. A `RunSession`'s connected client is cached in
`session.native["claude"]["client"]` and reused verbatim on every later
turn, mirroring the Google ADK runner's "bound to the first turn's build"
caching pattern exactly (a later turn's freshly-built `options` are
discarded once a client already exists). A one-off call (`session=None`)
connects and disconnects its own client every time -- the SDK's client owns
a real subprocess, so this runner always disconnects an ad hoc client, in a
`finally`, even on a failed run.
"""

from __future__ import annotations

import os
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
from .trace import Trace


class ClaudeAgentSDKRunner(BaseRunner):
    target = "claude"

    async def run(
        self,
        project: "Project",
        agent_name: str,
        prompt: str,
        *,
        session: Optional[RunSession] = None,
        hooks: Optional[HookRegistry] = None,
    ) -> Trace:
        from claude_agent_sdk import ClaudeSDKClient
        from claude_agent_sdk.types import (
            AssistantMessage,
            ResultMessage,
            ServerToolResultBlock,
            ServerToolUseBlock,
            ToolResultBlock,
            ToolUseBlock,
            UserMessage,
        )

        # Mirrors cli.py's `_run_claude` preflight exactly: the SDK's
        # bundled CLI needs this to authenticate, but nothing in the SDK
        # itself declares or checks for it (unlike `requires.env` in
        # agent-config.yaml, which `_check_env` already validated inside
        # `project.build`). Left outside the try/except below, alongside
        # `project.build`, for the same reason as both shipped runners: a
        # precondition failure here means the run never actually started.
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise OSError(
                "commonadk: missing required environment variable for target "
                "'claude': ANTHROPIC_API_KEY (the Claude Agent SDK's bundled "
                "CLI needs it to authenticate with the Anthropic API)"
            )

        trace = Trace()
        run_id = _uuid.uuid4().hex
        t0 = time.monotonic()

        # Always built, matching both shipped runners -- discarded below on
        # a session turn that reuses an already-connected client (see
        # module docstring, "RunSession support").
        options = project.build(agent_name, target="claude")

        state = session.native.setdefault(self.target, {}) if session is not None else {}
        client = state.get("client")
        if client is None:
            client = ClaudeSDKClient(options=options)
            await client.connect()
            if session is not None:
                state["client"] = client

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

        current_agent_name = agent_name
        self._emit(trace, hooks, AgentStarted(run_id=run_id, agent_name=current_agent_name))

        # ToolUseBlock.id -> {"name", "arguments", "start", "agent"},
        # popped when the matching ToolResultBlock arrives -- same
        # self-timed pairing pattern both shipped runners use (see their
        # docstrings); this SDK doesn't timestamp a tool call either.
        pending_calls: dict[str, dict[str, Any]] = {}
        # ToolUseBlock.id of an "Agent" (subagent) call -> the subagent
        # name it invoked, keyed off `input["subagent_type"]" -- see
        # adapters/claude_agent.py's own docstring for why that argument
        # name is the SDK-documented one. Every later message whose
        # `parent_tool_use_id` matches a key here is attributed to that
        # subagent instead of the root.
        subagent_of_call: dict[str, str] = {}
        final_text: Optional[str] = None

        def author_for(parent_tool_use_id: Optional[str]) -> str:
            if parent_tool_use_id is None:
                return agent_name
            return subagent_of_call.get(parent_tool_use_id, agent_name)

        try:
            try:
                await client.query(prompt)

                async for message in client.receive_response():
                    if isinstance(message, (AssistantMessage, UserMessage)):
                        author = author_for(message.parent_tool_use_id)
                        if author != current_agent_name:
                            self._emit(
                                trace,
                                hooks,
                                AgentFinished(run_id=run_id, agent_name=current_agent_name),
                            )
                            current_agent_name = author
                            self._emit(
                                trace,
                                hooks,
                                AgentStarted(run_id=run_id, agent_name=current_agent_name),
                            )

                    if isinstance(message, AssistantMessage):
                        if message.error:
                            # Inline, non-fatal -- see google_adk.py's identical
                            # pattern and design doc, "Fatal vs. inline RunError".
                            self._emit(
                                trace,
                                hooks,
                                RunError(
                                    run_id=run_id,
                                    message=str(message.error),
                                    error_type=f"claude-agent-sdk:{message.error}",
                                ),
                            )
                        for block in message.content:
                            if isinstance(block, (ToolUseBlock, ServerToolUseBlock)):
                                pending_calls[block.id] = {
                                    "name": block.name,
                                    "arguments": dict(block.input or {}),
                                    "start": time.monotonic(),
                                    "agent": current_agent_name,
                                }
                                if isinstance(block, ToolUseBlock) and block.name == "Agent":
                                    to_agent = block.input.get("subagent_type") if block.input else None
                                    if to_agent:
                                        subagent_of_call[block.id] = str(to_agent)
                                        self._emit(
                                            trace,
                                            hooks,
                                            Transfer(
                                                run_id=run_id,
                                                from_agent=current_agent_name,
                                                to_agent=str(to_agent),
                                                transfer_kind="claude-agent-sdk:agent_tool",
                                            ),
                                        )

                    elif isinstance(message, UserMessage):
                        content = message.content
                        blocks = content if isinstance(content, list) else []
                        for block in blocks:
                            if isinstance(block, (ToolResultBlock, ServerToolResultBlock)):
                                started = pending_calls.pop(block.tool_use_id, None)
                                duration_ms = (
                                    (time.monotonic() - started["start"]) * 1000.0
                                    if started is not None
                                    else None
                                )
                                error = getattr(block, "is_error", None)
                                self._emit(
                                    trace,
                                    hooks,
                                    ToolCall(
                                        run_id=run_id,
                                        agent_name=(started or {}).get("agent", current_agent_name),
                                        tool_name=(started or {}).get("name") or "<unknown>",
                                        arguments=(started or {}).get("arguments"),
                                        result_summary=self._summarize(block.content),
                                        duration_ms=duration_ms,
                                        error=str(error) if error else None,
                                    ),
                                )

                    elif isinstance(message, ResultMessage):
                        if message.is_error:
                            # Fatal -- the SDK's own turn ended in error
                            # (blocked, rate-limited, aborted, ...); the
                            # async iterator still completed normally (no
                            # Python exception), but the contract every
                            # runner follows is "RunFinished or RunError,
                            # never a RunFinished dressed up as success" --
                            # see base.py's docstring. Raising here routes
                            # this through the same fatal try/except below
                            # as a real SDK exception would.
                            raise RuntimeError(
                                "claude-agent-sdk: turn ended in error "
                                f"(subtype={message.subtype!r}, "
                                f"stop_reason={message.stop_reason!r}, "
                                f"api_error_status={message.api_error_status!r})"
                            )
                        final_text = message.result
                        self._emit(trace, hooks, self._llm_call_for(project, agent_name, run_id, message))
                        break
                    # SystemMessage / StreamEvent / RateLimitEvent /
                    # ConversationResetMessage / task-notification messages:
                    # no normalized equivalent in v1 -- see design doc.

                self._emit(trace, hooks, AgentFinished(run_id=run_id, agent_name=current_agent_name))
                if session is not None:
                    session.turns += 1
            finally:
                if session is None:
                    # Ad hoc (session=None) call -- always release the
                    # subprocess this client owns, even on a failed run. A
                    # session-bound client is intentionally never closed
                    # here -- it's cached in session.native for later turns.
                    # Best-effort: a failure to disconnect must never mask
                    # the real exception (if any) still propagating below.
                    try:
                        await client.disconnect()
                    except Exception:
                        pass

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

    # -- LLMCall: one per run(), sourced from the turn-final ResultMessage --

    def _llm_call_for(
        self, project: "Project", agent_name: str, run_id: str, message: Any
    ) -> LLMCall:
        """Build the run's one `LLMCall` from `ResultMessage`.

        Tokens come from `ResultMessage.model_usage` (a `dict[str,
        ModelUsage]`, one entry per distinct model actually used this
        turn) rather than the looser `ResultMessage.usage: dict[str, Any]`
        -- `ModelUsage` is a typed dict with guaranteed `inputTokens`/
        `outputTokens` keys (see `claude_agent_sdk.types.ModelUsage`),
        while `usage`'s shape is whatever the CLI subprocess happened to
        send and isn't documented at this layer. An empty/missing
        `model_usage` means unreported -- summing an empty dict would
        silently produce `0`, which is exactly the None-vs-0 rule this
        project exists to avoid, so that case is handled explicitly below,
        never by `sum()`-of-nothing.

        `cost_usd` comes straight from `ResultMessage.total_cost_usd` --
        this SDK computes cost itself (see docs/runner-design.md, "Cost
        estimation"); `runners/pricing.py` is never consulted for this
        target, even when tokens are known, since a per-model `costUSD`
        the SDK itself computed is strictly more authoritative than this
        project's own static table.
        """
        model_usage = message.model_usage or {}
        if model_usage:
            prompt_tokens = sum(int(mu.get("inputTokens", 0)) for mu in model_usage.values())
            completion_tokens = sum(int(mu.get("outputTokens", 0)) for mu in model_usage.values())
            total_tokens = prompt_tokens + completion_tokens
            model = next(iter(model_usage)) if len(model_usage) == 1 else self._resolved_model(
                project, agent_name
            )
        else:
            prompt_tokens = completion_tokens = total_tokens = None
            model = self._resolved_model(project, agent_name)

        return LLMCall(
            run_id=run_id,
            agent_name=agent_name,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=message.total_cost_usd,
            # ResultMessage.duration_api_ms is the one genuinely per-LLM-call
            # (well, per-turn) timing figure any of the three runners built
            # so far can report -- Google ADK and OpenAI Agents both always
            # report None here (see their docstrings); this SDK actually
            # timestamps the API-side portion of the turn separately from
            # tool-execution wall time.
            duration_ms=float(message.duration_api_ms)
            if message.duration_api_ms is not None
            else None,
        )

    @staticmethod
    def _resolved_model(project: "Project", agent_name: str) -> Optional[str]:
        """Best-effort model label for an `LLMCall` -- never raises, unlike
        the adapter's own `_model_for` (which must fail loudly at build
        time for a non-Anthropic model). Labeling an already-completed
        call should never turn a successful run into a failed one."""
        spec = project.agents.get(agent_name)
        if spec is None:
            return None
        override = spec.config.targets.get("claude", {})
        if "model" in override:
            return str(override["model"])
        try:
            resolved = project.resolve_model(agent_name)
        except ValueError:
            return None
        provider, sep, rest = resolved.partition("/")
        return rest if sep and provider == "anthropic" else resolved

    @staticmethod
    def _summarize(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value)
        return text if len(text) <= 500 else text[:500] + "..."


__all__ = ["ClaudeAgentSDKRunner"]
