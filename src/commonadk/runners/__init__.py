"""Runner registry -- turns a live, SDK-native build (from `adapters/`) into
a driven, normalized-event run.

Mirrors `adapters/__init__.py`'s registry shape on purpose: each target's
actual SDK is imported lazily, only when that target is requested via
`get_runner`. Importing `commonadk` (or `commonadk.runners` itself) must
keep working with no agent SDK installed at all -- this module has zero SDK
imports at module scope, exactly like `adapters/__init__.py`.

Every target has a runner as of issue #22, so `known_unported_targets()`
is empty today -- but the distinction is kept deliberately, because a new
adapter can land (see issue #11) before its runner does, and that target
deserves the honest "buildable but not yet traceable" message rather than
"unknown target". `known_targets()` lists targets that have a runner;
`known_unported_targets()` the real, buildable adapter targets that do not.
See docs/runner-design.md's per-SDK mapping table for how each plugs in.
`get_runner` distinguishes a target
that is simply unrecognized (`ValueError`, same message shape as
`adapters.get_adapter`) from one that is a real, buildable adapter target
with no runner *yet* (`NotImplementedError` pointing at the design doc) --
two different problems that deserve two different, honest error messages.
"""

from __future__ import annotations

from importlib import import_module

from .base import BaseRunner, RunSession
from .events import (
    AgentFinished,
    AgentStarted,
    Event,
    LLMCall,
    RunError,
    RunFinished,
    RunStarted,
    ToolCall,
    Transfer,
    event_from_dict,
)
from .hooks import HookRegistry
from .trace import Trace

# target -> (module to import, class name to instantiate, pip extra to suggest)
_REGISTRY: dict[str, tuple[str, str, str]] = {
    "google-adk": ("commonadk.runners.google_adk", "GoogleADKRunner", "google"),
    "openai": ("commonadk.runners.openai_agents", "OpenAIAgentsRunner", "openai"),
    "autogen": ("commonadk.runners.autogen", "AutoGenRunner", "autogen"),
    "langgraph": ("commonadk.runners.langgraph", "LangGraphRunner", "langgraph"),
    "claude": ("commonadk.runners.claude_agent", "ClaudeAgentSDKRunner", "claude"),
    "crewai": ("commonadk.runners.crewai_runner", "CrewAIRunner", "crewai"),
}

# Real `adapters/` build targets that don't have a runner yet. Empty since
# issue #22 ported all six, and kept on purpose: a new adapter (issue #11)
# can land before its runner, and listing it here is what earns it the
# specific "buildable, not yet traceable" error instead of "unknown
# target". See docs/runner-design.md's per-SDK mapping table.
_UNPORTED_TARGETS: set[str] = set()


def known_targets() -> list[str]:
    """Sorted list of targets with a registered runner -- no SDK import required."""
    return sorted(_REGISTRY)


def known_unported_targets() -> list[str]:
    """Sorted list of real adapter targets that don't have a runner yet."""
    return sorted(_UNPORTED_TARGETS)


def get_runner(target: str) -> BaseRunner:
    """Look up and instantiate the runner for `target`.

    Raises:
        NotImplementedError: `target` is a real, buildable adapter target
            (see `adapters.known_targets()`) but has no runner yet --
            naming the targets that do, and pointing at the design doc's
            mapping table for what `target`'s would look like.
        ValueError: `target` isn't a recognized target at all.
        ImportError: `target` has a runner but its SDK isn't installed --
            with a `pip install "commonadk[<extra>]"` hint.
    """
    if target in _UNPORTED_TARGETS:
        raise NotImplementedError(
            f"commonadk: tracing/streaming for target {target!r} is not "
            f"available yet -- only {known_targets()} have a runner today. "
            "See docs/runner-design.md, 'What the remaining unported SDKs "
            f"will map to', for the mapping planned for {target!r}. "
            "`commonadk run` without --stream/--trace still works for it "
            "via the original build-and-print path."
        )

    if target not in _REGISTRY:
        from ..adapters import known_targets as adapter_known_targets

        raise ValueError(
            f"Unknown run target {target!r}. Known targets: "
            f"{sorted(adapter_known_targets())}"
        )

    module_path, class_name, extra = _REGISTRY[target]
    try:
        module = import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"target {target!r} requires its SDK to be installed. "
            f'Install it with: pip install "commonadk[{extra}]" '
            f"(underlying import error: {e})"
        ) from e

    runner_cls = getattr(module, class_name)
    return runner_cls()


__all__ = [
    "BaseRunner",
    "RunSession",
    "HookRegistry",
    "Trace",
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
    "get_runner",
    "known_targets",
    "known_unported_targets",
]
