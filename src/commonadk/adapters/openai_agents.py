"""OpenAI Agents SDK adapter: `AgentSpec` -> live `agents.Agent`.

Edge semantics (v1 intersection decision, plan.md "Edge semantics v1"):
both `delegate` and `handoff` edges map to OpenAI Agents `handoffs` -- the
SDK exposes a single mechanism (an agent's `handoffs` list) regardless of
which of the two an edge declares in `interactions.yaml`. commonadk does not
yet distinguish them for OpenAI Agents; the distinction exists in the
neutral model so it *can* be honored by an adapter that supports it (or by a
future SDK feature) without changing `interactions.yaml`.

KEY DIFFERENCE from the Google ADK adapter -- handoffs are references, not a
tree: `agents.Agent.handoffs` is a plain `list[Agent | Handoff]` field on a
dataclass, with no parent-tracking and no "already has a parent" guard (see
`agents.Agent.__post_init__`, which only type-checks fields). The *same*
agent *instance* can legitimately sit in more than one parent's `handoffs`
list -- e.g. both `coordinator` and `researcher` can hand off to the same
`writer`. So, unlike the Google ADK adapter, this adapter does not reject
multi-parent graphs: it builds a `Agent` instance once per logical agent
name (memoized in a `dict[str, Agent]`) and reuses that same instance
everywhere it's referenced.

That same reference-not-tree property means cycles are not a construction
hazard the way they are for Google ADK's sub_agents tree: an `Agent`
dataclass can be created with an empty `handoffs=[]` and have handoffs
appended to it afterward, so a cycle (A -> B -> A) can be wired up *after*
both instances already exist, with no recursion and no partially-built
state. This adapter therefore builds every reachable agent once (a plain
two-pass construct-then-wire), and a cyclic `interactions.yaml` graph BUILDS
SUCCESSFULLY here -- verified against the installed SDK (openai-agents
0.21.1): `Agent.__post_init__` only validates field *types*, never handoff
graph shape, and assigning/extending `.handoffs` post-construction is a
plain list mutation. If a future SDK version starts rejecting cycles at
construction time, this adapter should raise the same style of clear error
the Google ADK adapter uses for its tree violation -- there is no such
rejection to catch today.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from agents import Agent, ModelSettings, function_tool

if TYPE_CHECKING:
    from ..models import AgentSpec, Project

from .base import BaseAdapter

# agent-config.yaml `model_params` key -> agents.ModelSettings field.
# `dataclasses.fields(ModelSettings)` (openai-agents 0.21.1) was introspected
# directly: `top_p`, `frequency_penalty`, and `presence_penalty` are real
# fields and map straight across, exactly like `temperature`/`max_tokens`.
# `top_k`, `stop`, and `seed` are NOT fields on this dataclass at all (no
# `stop_sequences` either) -- verified absent, not assumed -- so those three
# keys stay unmapped and fall through to the warn-and-ignore path below.
_MODEL_PARAM_MAP = {
    "temperature": "temperature",
    "max_tokens": "max_tokens",
    "top_p": "top_p",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
}


class OpenAIAgentsAdapter(BaseAdapter):
    target = "openai"

    def build(self, project: "Project", agent_name: str) -> Any:
        self._check_env(project, agent_name)
        memo: dict[str, Agent] = {}
        self._get_or_build(project, agent_name, memo)
        return memo[agent_name]

    # -- graph construction (memoized: one shared instance per agent name) --

    def _get_or_build(self, project: "Project", name: str, memo: dict[str, Agent]) -> Agent:
        """Return the single shared `Agent` instance for `name`, building it
        (and everything reachable from it) if this is the first visit.

        Two-pass per agent: construct the `Agent` with `handoffs=[]`, record
        it in `memo` *before* recursing into its own handoff targets, then
        fill in `handoffs` afterward. Recording before recursing is what
        makes a cycle safe -- if agent B's handoff graph loops back to A,
        the recursive call for A finds A already in `memo` and reuses it
        instead of recursing forever.
        """
        if name in memo:
            return memo[name]

        spec = project.agents[name]
        agent = Agent(
            name=spec.name,
            handoff_description=spec.config.description or None,
            instructions=spec.instructions,
            model=self._model_for(project, spec),
            model_settings=self._model_settings(spec),
            tools=[function_tool(tool.func) for tool in spec.tools],
            handoffs=[],
        )
        memo[name] = agent

        agent.handoffs = [
            self._get_or_build(project, edge.to, memo)
            for edge in project.graph.edges
            if edge.from_ == name
        ]
        return agent

    # -- model routing ------------------------------------------------------

    def _model_for(self, project: "Project", spec: "AgentSpec") -> Any:
        override = spec.config.targets.get("openai", {})
        if "model" in override:
            # Per-target override: already SDK-native form, passed through as-is.
            return override["model"]

        resolved = project.resolve_model(spec.name)  # LiteLLM-format string
        provider, sep, rest = resolved.partition("/")
        if sep and provider == "openai":
            return rest  # bare native model id, e.g. "gpt-4o"

        from agents.extensions.models.litellm_model import LitellmModel

        return LitellmModel(model=resolved)

    def _model_settings(self, spec: "AgentSpec") -> ModelSettings:
        params = spec.config.model_params
        kwargs: dict[str, Any] = {}
        for key, value in params.items():
            mapped = _MODEL_PARAM_MAP.get(key)
            if mapped is None:
                warnings.warn(
                    f"{spec.name}: model_params key '{key}' is not supported "
                    f"by the OpenAI Agents adapter and will be ignored",
                    stacklevel=2,
                )
                continue
            kwargs[mapped] = value

        return ModelSettings(**kwargs)
