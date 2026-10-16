"""Common interface every per-SDK runner implements, plus `RunSession`.

Mirrors `adapters/base.py`'s shape deliberately -- `BaseAdapter` is
build-time (`Project` + agent name -> one live SDK object), `BaseRunner` is
run-time (that object, driven for one turn, normalized into events). This
module stays free of any agent SDK import, exactly like `adapters/base.py`
-- constructing a `RunSession` or subclassing `BaseRunner` never requires
any SDK to be installed; only a concrete runner's `run()` body does, and
only for its own target.
"""

from __future__ import annotations

import asyncio
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from .hooks import HookRegistry
from .trace import Trace

if TYPE_CHECKING:
    from ..models import Project


@dataclass
class RunSession:
    """Per-target conversation state, so a second `run()` call continues
    the first instead of silently starting over.

    Create one `RunSession()` per logical conversation and pass it to every
    `run()` call that should continue it; a fresh `RunSession()` (or
    `session=None`) starts an unrelated conversation. Each runner reads and
    writes only its own slice of `native`, keyed by its own `target` string
    -- see each runner module's docstring, "Session/multi-turn", for
    exactly what it stores there (an ADK session id and a bound
    `InMemoryRunner`; an `agents.SQLiteSession`; ...).

    Where a future runner's SDK has no multi-turn story beyond "replay the
    whole history into the next prompt yourself", the contract is: raise a
    clear `NotImplementedError` naming the target and the specific gap
    when `session` is passed, rather than silently discarding it and
    starting a blank conversation -- see docs/runner-design.md, "Session /
    multi-turn model". Neither shipped runner needs this path; both have a
    real session primitive in their SDK.
    """

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    turns: int = 0
    native: dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    """Who this conversation is with, when the caller knows. Carried onto
    every `RunStarted` of the session, and used as the SDK's own user id
    where the SDK has one (Google ADK's session service)."""


CONTEXT_SEPARATOR = "\n\n"


def with_context(project: "Project", agent_name: str, context: Optional[str]) -> "Project":
    """A copy of `project` whose `agent_name` instructions end with `context`.

    The one mechanism every runner uses to put per-run context (memory,
    retrieved documents, ...) into an agent's instructions: every adapter
    reads `AgentSpec.instructions` at build time, so building from this
    copy delivers the context on all six targets without any SDK-specific
    code. Only the entry agent changes; sub-agents keep their own
    instructions. `project` itself is never mutated. Empty or `None`
    context returns `project` unchanged, and so does an unknown
    `agent_name` -- `build()` then raises its own, specific error for it.
    """
    if not context or agent_name not in project.agents:
        return project
    spec = project.agents[agent_name]
    joined = f"{spec.instructions}{CONTEXT_SEPARATOR}{context}" if spec.instructions else context
    agents = dict(project.agents)
    agents[agent_name] = spec.model_copy(update={"instructions": joined})
    return project.model_copy(update={"agents": agents})


def prompt_with_context(prompt: str, context: Optional[str]) -> str:
    """`prompt` preceded by `context` (or `prompt` unchanged when there is
    none)."""
    if not context:
        return prompt
    return f"{context}{CONTEXT_SEPARATOR}{prompt}"


@dataclass(frozen=True)
class Delivery:
    """How one run's `context` and `turn_context` reach the agent."""

    prompt: str
    """What to send as the turn's input."""
    build_context: Optional[str]
    """What to build the agent's instructions with (`with_context`), or
    None when this turn reuses an agent built earlier."""
    context_via: Optional[str]
    turn_context_via: Optional[str]


def plan_delivery(
    prompt: str,
    *,
    context: Optional[str],
    turn_context: Optional[str],
    reusing: bool,
    built_context: Optional[str] = None,
) -> Delivery:
    """The one place every runner decides where context goes.

    `context` is **stable** (it changes rarely), so it belongs in the
    instructions -- built in on a turn that builds the agent. A reused agent
    already carries the context it was built with: the same context again
    needs nothing, a changed one rides on the prompt. `turn_context` changes
    every turn and always rides on the prompt, after any stable context and
    before the caller's prompt.
    """
    prefix = []
    if not context:
        via = None
    elif not reusing or context == built_context:
        via = "instructions"
    else:
        via = "prompt"
        prefix.append(context)
    turn_via = "prompt" if turn_context else None
    if turn_context:
        prefix.append(turn_context)
    return Delivery(
        prompt=CONTEXT_SEPARATOR.join([*prefix, prompt]),
        build_context=None if reusing else context,
        context_via=via,
        turn_context_via=turn_via,
    )


class BaseRunner(ABC):
    """One runner per target SDK, registered in `commonadk.runners.get_runner`.

    Async-first (`run`), since every SDK this targets is async underneath
    (`google.adk.runners.Runner.run_async`, `agents.Runner.run_streamed`'s
    `stream_events()`, ...); `run_sync` is a thin `asyncio.run` wrapper for
    callers -- like the CLI's default, non-async path -- that don't need to
    be async themselves.
    """

    target: str

    @abstractmethod
    async def run(
        self,
        project: "Project",
        agent_name: str,
        prompt: str,
        *,
        session: Optional[RunSession] = None,
        hooks: Optional[HookRegistry] = None,
        context: Optional[str] = None,
        turn_context: Optional[str] = None,
    ) -> Trace:
        """Run one turn and return its `Trace` (events + rollup).

        Two kinds of per-run text for the agent (see `plan_delivery`):

        - `context` -- **stable** text (memory about the user, standing
          constraints): delivered in the entry agent's *instructions*
          (`with_context`) whenever this turn builds the agent. A session
          turn that reuses an SDK object built earlier (Google ADK's bound
          runner, the Claude client, AutoGen's team) already carries the
          context it was built with; a *changed* context then rides on the
          prompt instead.
        - `turn_context` -- **per-turn** text (what is relevant to this
          request): always in front of the prompt, so it never changes the
          instructions and never defeats prompt caching.

        Never silently: `RunStarted.context_via` / `turn_context_via` say
        where each went (`"instructions"`, `"prompt"`, or `None`), and
        `RunStarted.prompt` is always the caller's own prompt -- so a
        consumer recording what the user said does not record its own
        injected memory back as user input.

        Contract every implementation must follow (see docs/runner-design.md,
        "Fatal vs. inline RunError" for the full rationale):

        - Emit `RunStarted` first.
        - Route every event through both `trace.append(event)` and, if
          `hooks` is given, `hooks.fire(event)` -- use `self._emit(...)`,
          never append/fire separately, so the two never drift.
        - End in exactly one of `RunFinished` or `RunError` -- never both,
          never neither.
        - On any exception from the underlying SDK (or from
          `project.build(...)` itself, e.g. a missing env var surfacing as
          `OSError`), emit `RunError` as the last event and then RE-RAISE
          the original exception -- a `Trace` ending in `RunError` is a
          complete, honestly-reported failed run, not a swallowed one; the
          caller (the CLI, a test, a library consumer) must still see the
          real exception through normal Python control flow.
        """
        raise NotImplementedError

    def run_sync(
        self,
        project: "Project",
        agent_name: str,
        prompt: str,
        *,
        session: Optional[RunSession] = None,
        hooks: Optional[HookRegistry] = None,
        context: Optional[str] = None,
        turn_context: Optional[str] = None,
    ) -> Trace:
        """Synchronous convenience wrapper around `run` (`asyncio.run`)."""
        return asyncio.run(
            self.run(
                project, agent_name, prompt, session=session, hooks=hooks,
                context=context, turn_context=turn_context,
            )
        )

    @staticmethod
    def _emit(trace: Trace, hooks: Optional[HookRegistry], event: Any) -> None:
        """Append `event` to `trace` and fire it through `hooks` (if given).

        The one place both happen together -- every runner routes every
        event through this helper instead of calling `trace.append`/
        `hooks.fire` separately, so the two can never drift out of sync.
        """
        trace.append(event)
        if hooks is not None:
            hooks.fire(event)


__all__ = [
    "BaseRunner",
    "RunSession",
    "with_context",
    "prompt_with_context",
    "plan_delivery",
    "Delivery",
    "CONTEXT_SEPARATOR",
]
