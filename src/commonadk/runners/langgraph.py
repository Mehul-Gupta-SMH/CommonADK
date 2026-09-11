"""LangGraph runner: drives a `CompiledStateGraph.astream` and normalizes
its per-node state-value stream into commonadk's runner events.

Verified against the installed packages: langgraph 1.2.11
(`langgraph.graph.state.CompiledStateGraph.astream`, `langgraph.types.
Command`), langchain-core 1.6.0 (`langchain_core.messages.AIMessage`/
`ToolMessage`, `.usage_metadata`, `.tool_calls`, `.status`). Unlike both
shipped runners and the AutoGen runner above, this module's event mapping
was NOT taken from reading the SDK source alone -- `docs/runner-design.md`'s
own LangGraph row flagged several "not yet confirmed against the installed
version" gaps (which `stream_mode` actually surfaces tool calls; whether a
`Command`-driven handoff is visible in the stream at all; whether a
checkpointer can even be attached to the graph this codebase's adapter
returns). Every one of those was resolved by constructing a real,
offline `StateGraph` with `langchain_core.language_models.fake_chat_models.
GenericFakeChatModel` standing in for the model client (same shape
`langgraph_adapter.py` builds via `create_agent`, minus the network-bound
chat model), streaming it, and reading the actual chunks -- not guessed.
This docstring documents what that investigation found; the corrections
this makes to the design doc's original LangGraph row are called out
explicitly below and folded back into `docs/runner-design.md` itself.

WHAT THE STREAM LOOKS LIKE, verified by direct construction (see above) --
this is the load-bearing finding this whole runner is built on:

    graph.astream({"messages": [...]}, stream_mode="values", subgraphs=True)

yields `(namespace: tuple[str, ...], values: dict)` pairs. For the LEAF
case (`build()` returns a bare `create_agent(...)` react graph -- see
`adapters/langgraph_adapter.py`'s "WHAT build() RETURNS"), `namespace` is
always `()`: there is no subgraph nesting, so every one of that agent's own
internal model/tool steps streams directly at the top level. For the
MULTI-AGENT case (`build()` returns a parent `StateGraph` with one node per
reachable agent, each node itself a `CompiledStateGraph` -- LangGraph's own
"a compiled graph used as a node is automatically a subgraph" behavior),
`namespace` is `(f"{agent_name}:{run_uuid}",)` while a step is executing
INSIDE that agent's own subgraph, and reverts to `()` for the handful of
values the OUTER graph itself observes directly (the very first chunk,
carrying just the input `HumanMessage`; the moment a `Command(graph=
Command.PARENT)`-returning handoff tool resolves, which visibly SKIPS the
handed-off-FROM agent's own namespace entirely -- see "Handoffs" below; and
the very last chunk, a duplicate of whatever the final agent's own last
chunk already was). `namespace[0].split(":", 1)[0]` recovers the agent
name -- verified to equal exactly the node name this codebase's own
`langgraph_adapter.py` used in `builder.add_node(name, node)`, i.e. the
same commonadk agent name everywhere else in this codebase, not an SDK-
internal id. This runner never needs a SECOND level of nesting: verified
directly that `create_agent`'s own internal model/tools nodes do NOT add a
further namespace segment (they are plain node functions inside the one
subgraph `create_agent` already returned, not themselves compiled graphs
added as nodes) -- `namespace` here is at most one element long.

Per-message agent attribution -- two independent, agreeing signals,
verified together: every `AIMessage` a `create_agent`-built node produces
ALSO carries `.name` set to that `create_agent(..., name=...)` call's own
`name` (confirmed directly: `create_agent(model, ..., name="writer")`
yields `AIMessage(..., name="writer", ...)` with no further wiring needed
from this runner). `ToolMessage.name` is the TOOL's name, not the calling
agent's, so it carries no agent attribution of its own -- this runner
resolves a `ToolMessage`'s agent from the `pending_calls` entry the
matching `AIMessage.tool_calls` request queued (see "Tool calls" below),
falling back to whichever agent is currently open. Net effect: this
runner's `LLMCall.agent_name` is essentially never `None` -- a strictly
better attribution story than the OpenAI Agents SDK runner's documented
multi-handoff gap (see docs/runner-design.md), made possible by LangGraph's
per-message `.name` tagging and per-namespace subgraph attribution working
in agreement rather than needing to be inferred from either alone.

Message identity and de-duplication, verified not assumed: every
`BaseMessage` carries a stable `.id` (default-factoried once, at
construction) that is PRESERVED as the same message threads from a
subgraph's own namespace up into the parent graph's `()`-namespaced view
(confirmed directly: the exact same `AIMessage.id` appears in both the
`('writer:<uuid>',)`-namespaced chunk where it was produced and the later
`()`-namespaced chunk that surfaces it at the outer level). Since
`stream_mode="values"` yields the FULL accumulated message list on every
chunk (not a delta), a naive "process every message in every chunk" loop
would re-process and double-count every message once per namespace level
it passes through. This runner instead tracks `seen_ids: set[str]` and
processes a message exactly once, the first time its `.id` is observed --
which also means the chunk it's *first* seen in is always the most
specific (innermost) one, so no separate resolution logic is needed for
"which namespace does this message really belong to".

Tool calls -- request/response pairing by id, same pattern as the two
shipped runners and the AutoGen runner above: `AIMessage.tool_calls` (a
`list[ToolCall]` TypedDict, `.name`/`.args`/`.id` -- `.args` is ALREADY a
parsed dict, unlike AutoGen's/OpenAI's raw JSON-string arguments, verified
directly) queues a `pending_calls` entry per call, keyed by `.id`;
the matching `ToolMessage.tool_call_id` pops it to close the pairing and
compute `duration_ms` as this runner's own wall-clock delta (LangGraph, like
every other target here, does not timestamp tool execution itself).
`ToolMessage.status: Literal["success", "error"] = "success"`
(`langchain_core/messages/tool.py`, verified via `model_fields`) is the
one thing here with a REAL typed error signal, unlike ADK's dict-convention
`"error"` key or OpenAI's absence of one -- `error` is set to
`ToolMessage.content` when `status == "error"`, `None` otherwise.

Handoffs -- NOT a dedicated message type, verified directly (a correction
to the design doc's original "not yet confirmed" note): a
`langgraph_adapter.py`-built handoff tool's `Command(goto=..., graph=
Command.PARENT)` return value is consumed entirely by the graph engine --
the `ToolMessage` it also constructs (`content="Successfully transferred to
<dest>"`, per the adapter's own `_make_handoff_tool`) is INDISTINGUISHABLE,
by type, from any other tool's result message. The only observable signals
are (1) the tool's own name, which `langgraph_adapter.py` always spells
`transfer_to_<destination>` (verified: this runner is built specifically
against graphs THIS adapter produces, so trusting its own naming
convention here is a fair, documented coupling, not a guess about
LangGraph's API in general), and (2) empirically, that handoff's
`ToolMessage` surfaces in the OUTER graph's own `()` namespace directly,
never inside the handing-off-FROM agent's own subgraph namespace (verified:
`Command(graph=Command.PARENT)` causes the state update to apply at the
parent level, so the source agent's own subgraph never sees its own
handoff tool's result). This runner treats any `ToolMessage` whose `.name`
starts with `"transfer_to_"` as a `Transfer(transfer_kind=
"langgraph:command_handoff")` instead of a `ToolCall` -- `from_agent` is
whichever agent's `pending_calls` entry the matching `tool_call_id` queued
(the true source, captured before the Command-driven jump loses its own
namespace), `to_agent` is the suffix after `"transfer_to_"`. Deliberately
NOT also emitted as a `ToolCall` for the same call -- matching how neither
shipped runner double-reports its own SDK-native handoff signal as a tool
call, even though LangGraph's handoff genuinely IS implemented as an
ordinary tool call under the hood (unlike ADK's `EventActions.
transfer_to_agent` or OpenAI's dedicated `HandoffOutputItem`).

LLM usage -- per-message, via `AIMessage.usage_metadata: UsageMetadata |
None` (`langchain_core/messages/ai.py`, a `TypedDict` with required
`input_tokens`/`output_tokens`/`total_tokens` int fields when present).
THE `0`-NOT-`None` WRINKLE, LangGraph edition -- verified directly against
`langchain_openai/chat_models/base.py:2001-2009` (the non-streaming
`_create_chat_result` path this codebase's adapter always exercises,
`create_agent`'s model node never streams by default): `usage_metadata` is
set on the `AIMessage` ONLY `if token_usage` (the raw response's own
`"usage"` dict) is truthy at all -- so the outer "did the provider report
usage" question is already answered honestly at the `None`-vs-present level
for this provider. This runner still applies the SAME conservative
all-zero check this codebase uses everywhere else
(`reported = bool(usage["input_tokens"] or usage["output_tokens"] or
usage["total_tokens"])`) rather than trusting a present-but-possibly-
degenerate `UsageMetadata` blindly, for two reasons stated plainly: (1) the
individual sub-fields inside `_create_usage_metadata` themselves default
to `0` when a present-but-incomplete `usage` dict is missing one of them
(`langchain_openai/chat_models/base.py:4339` onward), and (2) per
docs/runner-design.md's own original caution here, `ChatAnthropic`'s and
`ChatGoogleGenerativeAI`'s own `usage_metadata`-population paths were not
independently re-verified field-by-field in this investigation (only
`ChatAnthropic`'s call site being unconditional, `chat_models.py:2086`, was
confirmed -- Anthropic's real API always returns a `usage` block, so this
is not expected to matter in practice for that provider, but the
conservative check costs nothing and protects against every provider
uniformly rather than special-casing one). A model_id is still always
attached to `LLMCall.model` when resolvable, exactly like every other
runner in this codebase -- `model` is descriptive metadata about which
model produced the call, independent of whether ITS usage happened to be
reported (see docs/runner-design.md, "The None-vs-0 rule": the rule is
about token/cost fields, not the model identifier).

Per-agent model resolution -- same reasoning and same shape as the AutoGen
runner above (see its module docstring, "Per-agent model resolution"):
`_resolved_model(project, name)` is resolved PER MESSAGE SOURCE (memoized
per `run()` call), not once for the whole run using only the build root,
since a multi-agent graph's participants can each be on a different
model/provider and this runner has real per-message attribution to use.

Not available: per-call duration (`UsageMetadata` carries no timing field);
an inline, SDK-recovered-from run error the way ADK's `LlmResponse.
error_code` is -- nothing in the "values" stream distinguishes "a node
raised and the graph kept going" from "nothing went wrong", so (like the
OpenAI Agents SDK and AutoGen runners) this runner only ever emits the one,
fatal `RunError` from its own top-level `try/except`.

SESSION/MULTI-TURN -- a genuine correction to the design doc's original
guess, not merely an elaboration of it: the design doc speculated a
`langgraph.checkpoint.*` checkpointer (e.g. `MemorySaver`) would be
"attached" for session continuity. Investigated directly and found
UNREACHABLE from this runner: `project.build(agent_name, target=
"langgraph")` -- a function this runner must call unmodified, per
`docs/runner-design.md`'s "Why a separate layer from adapters/" -- returns
an ALREADY-`builder.compile()`d graph with no checkpointer attached, and
`CompiledStateGraph` exposes no supported way to retrofit one after the
fact (`langgraph_adapter.py` is explicitly out of scope for this change --
see the task's "Do not touch" list). Instead, this runner implements
session continuity itself, entirely outside the graph: `RunSession.
native["langgraph"]` holds `{"messages": list[BaseMessage]}`, the full
accumulated conversation history as plain LangChain message objects. Each
turn's input is `{"messages": history + [{"role": "user", "content":
prompt}]}` (mixing already-constructed `BaseMessage` objects with a fresh
plain dict is fine -- `MessagesState`'s `add_messages` reducer normalizes
either form, verified via the same offline construction used throughout
this module), and after the run, `history` is replaced with the FULL final
message list this runner already tracked for `final_text` (see below) --
so turn N+1 starts from exactly the state turn N ended on. This reproduces
what a checkpointer would provide (the graph's only actual job for a
checkpointer here is replaying `messages` state across calls) without
needing one, and without touching the adapter. A fresh `RunSession()` (or
`session=None`) starts a new, empty history, same as every other runner
here. The graph itself is rebuilt fresh via `project.build(...)` on EVERY
turn regardless of session (it is stateless -- all continuity lives in
`RunSession.native`, not in the compiled graph object), mirroring exactly
how the OpenAI Agents SDK runner rebuilds its `Agent` fresh every turn
because history lives in the `SQLiteSession`, not the `Agent`.
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

_HANDOFF_PREFIX = "transfer_to_"


class LangGraphRunner(BaseRunner):
    target = "langgraph"

    async def run(
        self,
        project: "Project",
        agent_name: str,
        prompt: str,
        *,
        session: Optional[RunSession] = None,
        hooks: Optional[HookRegistry] = None,
    ) -> Trace:
        from langchain_core.messages import AIMessage, ToolMessage

        trace = Trace()
        run_id = _uuid.uuid4().hex
        t0 = time.monotonic()
        model_cache: dict[str, Optional[str]] = {}

        # Same reasoning as every other runner here: a build failure means
        # the run never started, so this stays outside the try/except
        # below. The graph itself is rebuilt fresh every turn -- see module
        # docstring, "Session/multi-turn" -- history lives in `session`,
        # never in the graph object.
        graph = project.build(agent_name, target=self.target)

        state = session.native.setdefault(self.target, {}) if session is not None else {}
        history: list[Any] = state.get("messages", [])

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
        seen_ids: set[str] = set()
        pending_calls: dict[str, dict[str, Any]] = {}
        final_messages: list[Any] = []

        try:
            input_messages = [*history, {"role": "user", "content": prompt}]
            async for namespace, values in graph.astream(
                {"messages": input_messages}, stream_mode="values", subgraphs=True
            ):
                msgs = values.get("messages", [])
                if not namespace:
                    final_messages = msgs

                node_agent = namespace[0].split(":", 1)[0] if namespace else None

                for msg in msgs:
                    msg_id = getattr(msg, "id", None)
                    if msg_id is None or msg_id in seen_ids:
                        continue
                    seen_ids.add(msg_id)

                    if isinstance(msg, AIMessage):
                        source = (
                            node_agent
                            or getattr(msg, "name", None)
                            or current_agent_name
                            or agent_name
                        )
                        if source != current_agent_name:
                            if current_agent_name is not None:
                                self._emit(
                                    trace,
                                    hooks,
                                    AgentFinished(run_id=run_id, agent_name=current_agent_name),
                                )
                            current_agent_name = source
                            self._emit(
                                trace, hooks, AgentStarted(run_id=run_id, agent_name=current_agent_name)
                            )

                        for call in msg.tool_calls or []:
                            pending_calls[call["id"]] = {
                                "name": call.get("name") or "<unknown>",
                                "arguments": call.get("args"),
                                "start": time.monotonic(),
                                "agent": source,
                            }

                        usage = getattr(msg, "usage_metadata", None)
                        if usage is not None:
                            # See module docstring, "The 0-not-None wrinkle,
                            # LangGraph edition".
                            reported = bool(
                                usage.get("input_tokens")
                                or usage.get("output_tokens")
                                or usage.get("total_tokens")
                            )
                            model_name = self._model_for(project, source, model_cache)
                            prompt_tokens = usage.get("input_tokens") if reported else None
                            completion_tokens = usage.get("output_tokens") if reported else None
                            total_tokens = usage.get("total_tokens") if reported else None
                            self._emit(
                                trace,
                                hooks,
                                LLMCall(
                                    run_id=run_id,
                                    agent_name=source,
                                    model=model_name,
                                    prompt_tokens=prompt_tokens,
                                    completion_tokens=completion_tokens,
                                    total_tokens=total_tokens,
                                    cost_usd=estimate_cost_usd(
                                        model_name, prompt_tokens, completion_tokens
                                    ),
                                    duration_ms=None,  # UsageMetadata carries no timing
                                ),
                            )

                    elif isinstance(msg, ToolMessage):
                        call_id = msg.tool_call_id
                        started = pending_calls.pop(call_id, None) if call_id else None
                        tool_name = (started or {}).get("name") or msg.name or "<unknown>"

                        if tool_name.startswith(_HANDOFF_PREFIX):
                            destination = tool_name[len(_HANDOFF_PREFIX) :]
                            self._emit(
                                trace,
                                hooks,
                                Transfer(
                                    run_id=run_id,
                                    from_agent=(started or {}).get("agent")
                                    or current_agent_name
                                    or agent_name,
                                    to_agent=destination,
                                    transfer_kind="langgraph:command_handoff",
                                ),
                            )
                        else:
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
                                    agent_name=(started or {}).get("agent")
                                    or current_agent_name
                                    or agent_name,
                                    tool_name=tool_name,
                                    arguments=(started or {}).get("arguments"),
                                    result_summary=self._summarize(msg.content),
                                    duration_ms=duration_ms,
                                    error=str(msg.content) if msg.status == "error" else None,
                                ),
                            )
                    # HumanMessage / SystemMessage / anything else: no
                    # normalized equivalent -- not a step in this model's
                    # vocabulary (matches how both shipped runners skip
                    # their own SDKs' non-mapped stream items).

            if current_agent_name is not None:
                self._emit(trace, hooks, AgentFinished(run_id=run_id, agent_name=current_agent_name))

            final_text = str(final_messages[-1].content) if final_messages else None

            if session is not None:
                state["messages"] = final_messages
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
    def _model_for(
        project: "Project", agent_name: Optional[str], cache: dict[str, Optional[str]]
    ) -> Optional[str]:
        """Resolve `agent_name`'s model, memoized per `run()` call -- see
        module docstring, "Per-agent model resolution"."""
        if agent_name is None:
            return None
        if agent_name in cache:
            return cache[agent_name]

        resolved: Optional[str] = None
        spec = project.agents.get(agent_name)
        if spec is not None:
            override = spec.config.targets.get("langgraph", {})
            if "model" in override:
                # Per-target overrides here are langchain-native
                # "provider:model" strings (see langgraph_adapter.py's own
                # docstring, "Per-target override"), not the LiteLLM
                # "provider/model" form `pricing.py.estimate_cost_usd`
                # strips on "/" -- normalize to the bare id here so a
                # priced override still gets costed.
                resolved = str(override["model"]).rsplit(":", 1)[-1]
            else:
                try:
                    resolved = project.resolve_model(agent_name)
                except ValueError:
                    resolved = None
        cache[agent_name] = resolved
        return resolved

    @staticmethod
    def _summarize(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value)
        return text if len(text) <= 500 else text[:500] + "..."
