"""The normalized runtime event model every runner produces.

See docs/runner-design.md ("The normalized event model") for why each event
exists and how each of the six SDKs' native execution surface maps onto it.
The short version: eight frozen, keyword-only dataclasses, every one
carrying a process-wide monotonic `seq` and a wall-clock `ts`, so events
from concurrent runs interleave unambiguously in a hook stream or a written
trace file.

**The None-vs-0 rule** (see the design doc section of the same name): every
token/cost/duration field on `LLMCall`/`RunFinished` defaults to `None` and
means "this SDK did not report it for this call" -- a runner must never
substitute `0` or an estimate. `trace.py`'s rollups enforce the same rule
at the aggregate level.

This module has no agent-SDK import at all -- constructing, serializing, and
round-tripping events never requires any of the six SDKs to be installed.
"""

from __future__ import annotations

import dataclasses
import itertools
import time
from typing import Any, ClassVar, Optional

_seq_counter = itertools.count(1)


def next_seq() -> int:
    """The next process-wide monotonic sequence number.

    Never reused, never reset mid-process. Sort events by `seq`, not `ts`
    -- two events can share a `ts` at whatever clock resolution the host
    platform gives `time.time()`, but `seq` is always a strict total order.
    """
    return next(_seq_counter)


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Event:
    """Base of every normalized runtime event. Never instantiated directly."""

    seq: int = dataclasses.field(default_factory=next_seq)
    ts: float = dataclasses.field(default_factory=time.time)
    run_id: str = ""

    kind: ClassVar[str] = "event"
    """Discriminator used by `to_dict()`/`event_from_dict()` -- a ClassVar,
    so it is never treated as a dataclass field (excluded from `asdict()`,
    equality, and the constructor)."""

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict form, with a `"type"` key added from `kind`."""
        d = dataclasses.asdict(self)
        d["type"] = self.kind
        return d


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class RunStarted(Event):
    """Marks the start of exactly one `run()` call.

    `session_id` is the owning `RunSession.session_id` when the call is
    part of a multi-turn conversation, else `None` -- the field that lets a
    trace consumer tell a one-off turn from turn N of an ongoing session.
    """

    kind: ClassVar[str] = "run_started"

    target: str
    agent_name: str
    prompt: str
    session_id: Optional[str] = None


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class AgentStarted(Event):
    """A specific agent began producing output.

    Multi-agent runs emit more than one `AgentStarted`/`AgentFinished` pair
    -- see each runner module's docstring for exactly what native signal
    (an explicit SDK event vs. an author-change heuristic) triggers this.
    """

    kind: ClassVar[str] = "agent_started"

    agent_name: str


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class AgentFinished(Event):
    """Closes the `AgentStarted` for the same `agent_name`."""

    kind: ClassVar[str] = "agent_finished"

    agent_name: str
    output_summary: Optional[str] = None


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class LLMCall(Event):
    """One model round-trip. Every cost/usage rollup is built from these.

    `agent_name` is `Optional` (not just the other four) because at least
    one shipped runner (OpenAI Agents, on a run that spans a handoff) has
    no reliable way to attribute a given call to a specific agent -- see
    docs/runner-design.md. `model`/`prompt_tokens`/`completion_tokens`/
    `total_tokens`/`cost_usd`/`duration_ms` are all `Optional`, defaulting
    to `None` to mean "not reported by this SDK for this call" -- see the
    module docstring's None-vs-0 rule.
    """

    kind: ClassVar[str] = "llm_call"

    agent_name: Optional[str] = None
    model: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    duration_ms: Optional[float] = None


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class ToolCall(Event):
    """One tool invocation the underlying SDK reports as a discrete step.

    `arguments` is the parsed call arguments when the native SDK gave them
    to us in a recoverable form (see each runner's docstring for exactly
    how); `result_summary` is a truncated `str()` of the tool's return
    value, never the raw object (which may not be JSON-serializable).
    `duration_ms` and `error` are `Optional` -- most SDKs pair a call/result
    across two separate native events with no duration of their own, so a
    runner times the pairing itself (documented per-runner, not implied to
    be SDK-reported); `error` is `None` unless the SDK's own result payload
    signals a tool failure.
    """

    kind: ClassVar[str] = "tool_call"

    agent_name: str
    tool_name: str
    arguments: Optional[dict[str, Any]] = None
    result_summary: Optional[str] = None
    duration_ms: Optional[float] = None
    error: Optional[str] = None


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Transfer(Event):
    """One agent routed work to another, observed at run time.

    `transfer_kind` names the *native* mechanism observed (e.g.
    `"google-adk:transfer_to_agent"`, `"openai-agents:handoff"`) -- not
    commonadk's own `delegate`/`handoff` `interactions.yaml` vocabulary, since
    no adapter distinguishes those two at build time either (see
    docs/HLD.md, "v1 edge-semantics intersection") and a runner has no
    ground truth to map onto that distinction.
    """

    kind: ClassVar[str] = "transfer"

    from_agent: str
    to_agent: str
    transfer_kind: str


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class RunFinished(Event):
    """The terminal, successful event -- a `run()` call ends in exactly
    one of `RunFinished` or `RunError`, never both, never neither (see
    `runners/base.py`'s `BaseRunner.run` docstring).

    The token/cost totals mirror `Trace.rollup()`'s `llm_calls` block at
    the moment this event is emitted, so a `--stream` consumer sees the
    final numbers without separately re-deriving them from the trace.
    `usage_complete` is `False` whenever any `LLMCall` in the run did not
    report usage -- see docs/runner-design.md, "The None-vs-0 rule".
    """

    kind: ClassVar[str] = "run_finished"

    final_text: Optional[str] = None
    total_prompt_tokens: Optional[int] = None
    total_completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    total_cost_usd: Optional[float] = None
    duration_ms: Optional[float] = None
    usage_complete: bool = True


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class RunError(Event):
    """The terminal, failed event -- see docs/runner-design.md, "Fatal vs.
    inline RunError" for the distinction between this ending a `run()` call
    (fatal, exactly one, always last) and an inline `RunError` a runner may
    emit mid-run for a native per-turn error the SDK itself recovered from
    (Google ADK only, see `runners/google_adk.py`).
    """

    kind: ClassVar[str] = "run_error"

    message: str
    error_type: str


_EVENT_TYPES: dict[str, type[Event]] = {
    cls.kind: cls
    for cls in (
        RunStarted,
        AgentStarted,
        AgentFinished,
        LLMCall,
        ToolCall,
        Transfer,
        RunFinished,
        RunError,
    )
}


def event_from_dict(data: dict[str, Any]) -> Event:
    """Inverse of `Event.to_dict()` -- reconstructs the concrete subclass
    named by the dict's `"type"` key. Raises `ValueError` for an unknown
    `"type"` (e.g. a trace file from a future, incompatible version).
    """
    data = dict(data)
    kind = data.pop("type", None)
    cls = _EVENT_TYPES.get(kind)  # type: ignore[arg-type]
    if cls is None:
        raise ValueError(
            f"commonadk: unknown event type {kind!r} in trace data. Known "
            f"types: {sorted(_EVENT_TYPES)}"
        )
    return cls(**data)


__all__ = [
    "Event",
    "RunStarted",
    "AgentStarted",
    "AgentFinished",
    "LLMCall",
    "ToolCall",
    "Transfer",
    "RunFinished",
    "RunError",
    "event_from_dict",
    "next_seq",
]
