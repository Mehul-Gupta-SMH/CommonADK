"""OpenAI Agents SDK adapter: `AgentSpec` -> live `agents.Agent`.

Edge semantics -- THIS ADAPTER HONORS THE DELEGATE/HANDOFF DISTINCTION
(GitHub issue #10's first checkbox), not the v1 collapsed mapping every
other adapter in this codebase still uses: the installed SDK (openai-agents
0.21.1) has two genuinely distinct mechanisms, and `Agent.as_tool`'s own
docstring states the difference in exactly these terms (verified via
`inspect.getdoc(agents.Agent.as_tool)`, not assumed):

    "This is different from handoffs in two ways:
    1. In handoffs, the new agent receives the conversation history. In
       this tool, the new agent receives generated input.
    2. In handoffs, the new agent takes over the conversation. In this
       tool, the new agent is called as a tool, and the conversation is
       continued by the original agent."

That second point is exactly this project's delegate/handoff split (models.py
`InteractionEdge.type`): "takes over the conversation" (control transfers
and never returns) is `handoff`; "called as a tool, conversation is
continued by the original agent" (a sub-call that returns) is `delegate`.
So:

- `handoff` edges map to `agent.handoffs` (unchanged from before this
  feature) -- the destination `Agent` is appended to the source's
  `handoffs` list, and the SDK's own `Runner` hands the whole conversation
  over to it when the model calls the corresponding built-in handoff tool.
- `delegate` edges map to `dest_agent.as_tool(tool_name=f"delegate_to_
  {dest}", tool_description=...)` (new), appended to the source's `tools`
  list instead -- a plain `FunctionTool` that runs the destination agent to
  completion on a generated input and returns its output as this tool
  call's result, with the SOURCE agent's own run continuing right after
  (verified via `inspect.getsource(Agent.as_tool)`: it builds a
  `FunctionTool` whose `on_invoke_tool` calls `Runner.run(self, input, ...)`
  and returns the extracted output as a string -- no conversation-transfer
  primitive involved at all).

Both mechanisms build on the same memoized `dict[str, Agent]` this adapter
already maintains (see "KEY DIFFERENCE" below), so a `delegate` edge to a
destination also reachable via a `handoff` edge elsewhere in the graph wraps
the SAME shared `Agent` instance in `as_tool()` -- no duplicate construction.

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

        Two-pass per agent: construct the `Agent` with `handoffs=[]` (its
        own tools list already includes its `tools.py` functions), record it
        in `memo` *before* recursing into its own outgoing edges, then fill
        in `handoffs` and append delegate tools afterward. Recording before
        recursing is what makes a cycle safe -- if agent B's outgoing edges
        loop back to A (via either edge type), the recursive call for A
        finds A already in `memo` and reuses it instead of recursing
        forever.
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

        # See module docstring, "Edge semantics" -- `handoff` edges transfer
        # the conversation (agents.handoffs); `delegate` edges are a sub-call
        # that returns (agent.as_tool(...), appended to agent.tools).
        agent.handoffs = [
            self._get_or_build(project, edge.to, memo)
            for edge in project.graph.edges
            if edge.from_ == name and edge.type == "handoff"
        ]
        agent.tools = agent.tools + [
            self._get_or_build(project, edge.to, memo).as_tool(
                tool_name=f"delegate_to_{edge.to}",
                tool_description=(
                    f"Delegate a task to the '{edge.to}' agent and receive "
                    f"its result back into this conversation."
                ),
            )
            for edge in project.graph.edges
            if edge.from_ == name and edge.type == "delegate"
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
