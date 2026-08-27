"""Adapter registry -- turns a framework-neutral `Project` into a live SDK agent.

Each target's actual SDK is imported lazily, only when that target is
requested via `get_adapter`. Importing `commonadk` (or calling
`commonadk.load()`) must keep working with no agent SDK installed at all --
this module and `base.py` have zero SDK imports at module scope to guarantee
that.
"""

from __future__ import annotations

from importlib import import_module

from .base import BaseAdapter

# target -> (module to import, class name to instantiate, pip extra to suggest)
_REGISTRY: dict[str, tuple[str, str, str]] = {
    "google-adk": ("commonadk.adapters.google_adk", "GoogleADKAdapter", "google"),
    "openai": ("commonadk.adapters.openai_agents", "OpenAIAgentsAdapter", "openai"),
    "claude": ("commonadk.adapters.claude_agent", "ClaudeAgentSDKAdapter", "claude"),
    "crewai": ("commonadk.adapters.crewai_adapter", "CrewAIAdapter", "crewai"),
    "autogen": ("commonadk.adapters.autogen_adapter", "AutoGenAdapter", "autogen"),
    "langgraph": ("commonadk.adapters.langgraph_adapter", "LangGraphAdapter", "langgraph"),
}


def get_adapter(target: str) -> BaseAdapter:
    """Look up and instantiate the adapter for `target`.

    Raises `ValueError` naming the known targets if `target` is unrecognized,
    or `ImportError` with a `pip install "commonadk[<extra>]"` hint if the
    target is recognized but its underlying SDK is not installed.
    """
    if target not in _REGISTRY:
        raise ValueError(
            f"Unknown build target {target!r}. Known targets: "
            f"{sorted(_REGISTRY)}"
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

    adapter_cls = getattr(module, class_name)
    return adapter_cls()


__all__ = ["BaseAdapter", "get_adapter"]
