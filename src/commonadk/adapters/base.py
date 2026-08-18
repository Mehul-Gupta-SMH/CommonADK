"""Common interface every per-SDK adapter implements.

See plan.md ("Adapt") for the architecture this slots into: one adapter per
target SDK, each turning a framework-neutral `Project` + agent name into a
live, SDK-native agent object. Kept intentionally minimal -- no speculative
hooks beyond what M2 (Google ADK) actually needs; M3 (OpenAI Agents) can grow
the contract if it turns out to need more.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..models import Project


class BaseAdapter(ABC):
    """One adapter per target SDK, registered in `commonadk.adapters.get_adapter`."""

    target: str

    @abstractmethod
    def build(self, project: "Project", agent_name: str) -> Any:
        """Build and return a live, SDK-native agent object for `agent_name`."""
        raise NotImplementedError
