"""Common interface every per-SDK adapter implements.

See plan.md ("Adapt") for the architecture this slots into: one adapter per
target SDK, each turning a framework-neutral `Project` + agent name into a
live, SDK-native agent object.

`_reachable_agents` and `_check_env` started life in M2's Google ADK adapter
and were hoisted here in M3 so both the Google ADK and OpenAI Agents adapters
share one implementation of "which agents does this build touch" and "fail
loudly, up front, if any of them is missing a required env var" -- the logic
is identical across targets, only the target name in the error message
differs (via `self.target`). This module stays free of any agent SDK
import -- `commonadk.load()` must keep working with no agent SDK installed
at all.
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

    # -- env preflight (shared across targets) ---------------------------

    def _check_env(self, project: "Project", agent_name: str) -> None:
        """Fail loudly, up front, if any reachable agent is missing a required env var.

        Checks `agent_name` and every agent reachable from it via
        `interactions.yaml` edges (not just its direct sub_agents/handoffs)
        -- building the graph can transitively depend on any of them.
        """
        missing_lines: list[str] = []
        for name in self._reachable_agents(project, agent_name):
            agent = project.agents[name]
            missing_names = set(project.check_env(name))
            if not missing_names:
                continue
            for req in agent.config.requires.env:
                if req.name in missing_names:
                    detail = f"{req.name} ({req.description})" if req.description else req.name
                    missing_lines.append(f"  - {name}: {detail}")

        if missing_lines:
            raise OSError(
                "commonadk: missing required environment variable(s) for "
                f"target {self.target!r} (building '{agent_name}'):\n"
                + "\n".join(missing_lines)
            )

    def _reachable_agents(self, project: "Project", start: str) -> list[str]:
        """Every agent name reachable from `start` via ANY edge (delegate or
        handoff), including `start`. A thin wrapper over `_reachable_via`
        with both edge types allowed -- kept as its own method since it is
        the traversal every adapter's env preflight and flat-registry
        construction has always used, unaffected by the delegate/handoff
        distinction (issue #10): an agent's required env vars matter
        whether it's reached by a call-and-return or a control transfer.
        """
        return self._reachable_via(project, start, {"delegate", "handoff"})

    def _reachable_via(
        self, project: "Project", start: str, edge_types: "set[str]"
    ) -> list[str]:
        """Every agent name reachable from `start`, including `start`,
        following only edges whose `type` is in `edge_types`.

        Added for issue #10 (honor the delegate/handoff distinction): an
        adapter that builds a different SDK-native structure for the two
        edge types (e.g. AutoGen's `Swarm` participants, which must only be
        `handoff`-reachable agents, never `delegate`-only destinations
        that are wired in as `AgentTool`s instead) calls this with just the
        one edge type it cares about, rather than filtering
        `_reachable_agents`'s all-edges result after the fact.
        """
        seen = [start]
        seen_set = {start}
        i = 0
        while i < len(seen):
            current = seen[i]
            i += 1
            for edge in project.graph.edges:
                if edge.from_ == current and edge.type in edge_types and edge.to not in seen_set:
                    seen_set.add(edge.to)
                    seen.append(edge.to)
        return seen
