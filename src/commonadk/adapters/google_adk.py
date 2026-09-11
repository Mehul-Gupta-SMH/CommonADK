"""Google ADK adapter: `AgentSpec` -> live `google.adk` agent.

Edge semantics -- THIS ADAPTER HONORS THE DELEGATE/HANDOFF DISTINCTION
(GitHub issue #10's first checkbox), not the v1 collapsed mapping every
other adapter in this codebase still uses. Investigated directly against
the installed SDK (google-adk 2.7.1), not assumed: `google.adk.tools.
agent_tool.AgentTool` "wraps an agent" so it "allows an agent to be called
as a tool within a larger application" -- its own docstring states "The
agent's input schema is used to define the tool's input parameters, and the
agent's output is returned as the tool's result" (verified via `inspect.
getsource`). That is exactly this project's `delegate` semantic: the caller
invokes the callee and gets its result back, then keeps running. `sub_agents`
+ ADK's own `transfer_to_agent` mechanism, by contrast, is exactly `handoff`:
control transfers to the callee and does not return (see "Sub-agent tree
constraint" below, unchanged from before this feature). So:

- `handoff` edges map to ADK `sub_agents`, unchanged from before this
  feature -- the destination becomes a genuine sub-agent of the source, and
  ADK's runtime routes control to it via `transfer_to_agent` with no return.
- `delegate` edges map to a `google.adk.tools.AgentTool(agent=<destination>)`
  appended to the source's own `tools` list (new) -- the destination is
  built as its own, fully independent agent (recursively, with its own
  `sub_agents`/`AgentTool`s for whatever edges IT has), never added to
  `sub_agents` anywhere. `AgentTool.__init__` never touches `parent_agent`
  (verified via `inspect.getsource`: it only sets `self.agent`/
  `self.skip_summarization`/etc. and calls `BaseTool.__init__`), so a
  delegate destination is exempt from ADK's sub_agents tree/parent
  constraint entirely -- it can be `AgentTool`-wrapped from as many
  different sources as `interactions.yaml` likes, and can independently
  ALSO be some other agent's `handoff` sub_agent, with no conflict, since
  those are two structurally separate `Agent` instances.

Sub-agent tree constraint (applies to `handoff` edges only, now): `google.
adk.agents.base_agent.BaseAgent` enforces a strict tree -- `model_post_init`
-> `__set_parent_agent_for_sub_agents` raises `ValueError` if a sub-agent
instance already has a `parent_agent` set (google-adk 2.7.1, `base_agent.
py`). That guard only fires for a *shared instance*; building a second,
independent instance of the same logical agent under a second parent would
sail right past it and silently duplicate the agent instead of erroring. So
this adapter tracks which logical agent names have already been claimed by
a `handoff` parent while it walks `interactions.yaml` itself, *before*
constructing anything, and raises a clear error naming the conflicting edge
if the same agent is `handoff`-reachable from two parents. ADK sub_agents
are a tree; the `handoff` subgraph of `interactions.yaml` is a graph -- this
adapter requires the *handoff-only* subgraph reachable from the build root
to actually be a tree (a `delegate` edge to the same destination never
counts against this, since AgentTool doesn't join the tree at all -- see
`test_delegate_edge_bypasses_the_sub_agents_tree_constraint` in
test_adapter_google.py). Each `AgentTool`-wrapped delegate subtree gets its
OWN fresh tree-tracking state, independent of the outer build's, since it is
a structurally separate object graph with no shared `parent_agent`
bookkeeping to conflict over.

Cycle detection now covers BOTH edge types uniformly: a `delegate` edge
recurses into a fresh, fully independent build of its destination (see
above), so a cycle through delegate edges (or a mix of the two types) is
just as much an unbounded-recursion hazard at construction time as a cycle
through handoff edges always was -- this adapter threads one `ancestors`
path (the chain of agent names currently under construction) through both
the `sub_agents` and `AgentTool` recursion, and raises before ever
recursing past a repeat, regardless of which edge type closes the loop.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from google.adk.agents import Agent
from google.adk.tools import AgentTool

if TYPE_CHECKING:
    from ..models import AgentSpec, Project

from .base import BaseAdapter

# agent-config.yaml `model_params` key -> google.genai.types.GenerateContentConfig
# field. `GenerateContentConfig.model_fields` (google-adk 2.7.1) was
# introspected directly to confirm every key below is a real field -- this is
# the one adapter in this codebase where every candidate sampling param
# (top_p, top_k, stop, presence_penalty, frequency_penalty, seed) genuinely
# has a matching field, since GenerateContentConfig is a single flat config
# object with no per-provider client split (see module docstring).
_MODEL_PARAM_MAP = {
    "temperature": "temperature",
    "max_tokens": "max_output_tokens",
    "top_p": "top_p",
    "top_k": "top_k",
    "stop": "stop_sequences",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
    "seed": "seed",
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
                f"from the build root ({chain}). Neither Google ADK's "
                f"sub_agents (handoff edges) nor a nested AgentTool "
                f"(delegate edges) can represent a cycle -- both would "
                f"recurse without ever terminating at construction time."
            )
        if name in claimed:
            raise ValueError(
                f"commonadk: agent '{name}' is reachable via a `handoff` "
                f"edge from two different parents in interactions.yaml -- "
                f"it is already a sub_agent of '{claimed[name]}'. Google "
                f"ADK's sub_agents form a tree (an agent can only have one "
                f"parent), so this project's `handoff` subgraph is not "
                f"representable as an ADK sub_agents tree. Conflicting "
                f"edge: '{parent}' -> '{name}'. (A `delegate` edge to the "
                f"same destination would not conflict -- delegate targets "
                f"are wrapped in an independent AgentTool, not added to "
                f"sub_agents; see google_adk.py's module docstring.)"
            )

        spec = project.agents[name]
        claimed[name] = parent if parent is not None else "<build root>"
        child_ancestors = (*ancestors, name)

        # `handoff` edges join the sub_agents tree (control transfers, never
        # returns) -- subject to `claimed`'s one-parent tree constraint,
        # same as before this feature (see module docstring, "Sub-agent
        # tree constraint").
        sub_agents = [
            self._build_agent(project, edge.to, claimed, child_ancestors, parent=name)
            for edge in project.graph.edges
            if edge.from_ == name and edge.type == "handoff"
        ]

        # `delegate` edges become AgentTool-wrapped sub-calls (control
        # returns to `name` afterward) -- each gets its own fresh `claimed`
        # dict, since AgentTool never sets `parent_agent` and so is exempt
        # from the sub_agents tree constraint entirely (see module
        # docstring, "Edge semantics"). Cycles are still guarded via the one
        # shared `ancestors` chain threaded through both branches.
        delegate_tools = [
            AgentTool(agent=self._build_agent(project, edge.to, {}, child_ancestors, parent=None))
            for edge in project.graph.edges
            if edge.from_ == name and edge.type == "delegate"
        ]

        return Agent(
            name=spec.name,
            description=spec.config.description,
            instruction=spec.instructions,
            model=self._model_for(project, spec),
            tools=[tool.func for tool in spec.tools] + delegate_tools,
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
