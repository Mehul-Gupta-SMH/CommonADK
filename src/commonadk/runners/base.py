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
    ) -> Trace:
        """Run one turn and return its `Trace` (events + rollup).

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
    ) -> Trace:
        """Synchronous convenience wrapper around `run` (`asyncio.run`)."""
        return asyncio.run(
            self.run(project, agent_name, prompt, session=session, hooks=hooks)
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


__all__ = ["BaseRunner", "RunSession"]
