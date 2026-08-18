"""Google ADK adapter: `AgentSpec` -> live `google.adk` agent.

Edge semantics (v1 intersection decision, plan.md "Edge semantics v1"):
both `delegate` and `handoff` edges map to ADK `sub_agents` -- ADK's runtime
picks the actual transfer mechanism (`transfer_to_agent`) regardless of which
of the two the edge declares in `interactions.yaml`. commonadk does not yet
distinguish them for Google ADK; the distinction exists in the neutral model
so it *can* be honored by an adapter that supports it (or by a future ADK
feature) without changing `interactions.yaml`.

Sub-agent tree constraint: `google.adk.agents.base_agent.BaseAgent` enforces
a strict tree -- `model_post_init` -> `__set_parent_agent_for_sub_agents`
raises `ValueError` if a sub-agent instance already has a `parent_agent` set
(google-adk 2.7.1, `base_agent.py`). That guard only fires for a *shared
instance*; building a second, independent instance of the same logical agent
under a second parent would sail right past it and silently duplicate the
agent instead of erroring. So this adapter tracks which logical agent names
have already been claimed by a parent while it walks `interactions.yaml`
itself, *before* constructing anything, and raises a clear error naming the
conflicting edge if the same agent is reachable from two parents. ADK
sub_agents are a tree; `interactions.yaml` is a graph -- v1 requires the
subgraph reachable from the build root to actually be a tree.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from google.adk.agents import Agent

if TYPE_CHECKING:
    from ..models import AgentSpec, Project

from .base import BaseAdapter

# agent-config.yaml `model_params` key -> google.genai.types.GenerateContentConfig field
_MODEL_PARAM_MAP = {
    "temperature": "temperature",
    "max_tokens": "max_output_tokens",
}


class GoogleADKAdapter(BaseAdapter):
    target = "google-adk"

    def build(self, project: "Project", agent_name: str) -> Any:
        self._check_env(project, agent_name)
        claimed: dict[str, str] = {}
        return self._build_agent(project, agent_name, claimed, ancestors=(), parent=None)

    # -- tree construction ------------------------------------------------

    def _build_agent(
        self,
        project: "Project",
        name: str,
        claimed: dict[str, str],
        ancestors: tuple[str, ...],
        parent: "str | None",
    ) -> Agent:
        if name in ancestors:
            chain = " -> ".join((*ancestors, name))
            raise ValueError(
                f"commonadk: cycle detected in interactions.yaml reachable "
                f"from the build root ({chain}). Google ADK's sub_agents "
                f"cannot represent a cycle."
            )
        if name in claimed:
            raise ValueError(
                f"commonadk: agent '{name}' is reachable from two different "
                f"parents in interactions.yaml -- it is already a sub_agent "
                f"of '{claimed[name]}'. Google ADK's sub_agents form a tree "
                f"(an agent can only have one parent), so this project's "
                f"interaction graph is not representable as an ADK "
                f"sub_agents tree. Conflicting edge: '{parent}' -> '{name}'."
            )

        spec = project.agents[name]
        claimed[name] = parent if parent is not None else "<build root>"
        child_ancestors = (*ancestors, name)

        sub_agents = [
            self._build_agent(project, edge.to, claimed, child_ancestors, parent=name)
            for edge in project.graph.edges
            if edge.from_ == name
        ]

        return Agent(
            name=spec.name,
            description=spec.config.description,
            instruction=spec.instructions,
            model=self._model_for(project, spec),
            tools=[tool.func for tool in spec.tools],
            generate_content_config=self._generate_content_config(spec),
            sub_agents=sub_agents,
        )

    # -- model routing ------------------------------------------------------

    def _model_for(self, project: "Project", spec: "AgentSpec") -> Any:
        override = spec.config.targets.get("google-adk", {})
        if "model" in override:
            # Per-target override: already SDK-native form, passed through as-is.
            return override["model"]

        resolved = project.resolve_model(spec.name)  # LiteLLM-format string
        provider, sep, rest = resolved.partition("/")
        if sep and provider == "gemini":
            return rest  # bare native model id, e.g. "gemini-2.5-pro"

        from google.adk.models.lite_llm import LiteLlm

        return LiteLlm(model=resolved)

    def _generate_content_config(self, spec: "AgentSpec") -> Any:
        params = spec.config.model_params
        if not params:
            return None

        kwargs: dict[str, Any] = {}
        for key, value in params.items():
            mapped = _MODEL_PARAM_MAP.get(key)
            if mapped is None:
                warnings.warn(
                    f"{spec.name}: model_params key '{key}' is not supported "
                    f"by the Google ADK adapter and will be ignored",
                    stacklevel=2,
                )
                continue
            kwargs[mapped] = value

        if not kwargs:
            return None

        from google.genai import types as genai_types

        return genai_types.GenerateContentConfig(**kwargs)
