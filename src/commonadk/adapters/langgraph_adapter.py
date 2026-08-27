"""LangGraph adapter: `AgentSpec` -> a live, compiled `langgraph` graph.

WHAT `build()` RETURNS -- read this first, verified not assumed: every
reachable agent is built as its own prebuilt react agent -- a
`langgraph.graph.state.CompiledStateGraph` from `langchain.agents.
create_agent(model, tools=..., system_prompt=..., name=...)`. This is the
CURRENT idiomatic entry point, not `langgraph.prebuilt.create_react_agent`:
that function still exists and still works in the installed stack
(langgraph 1.2.11), but calling it emits `LangGraphDeprecatedSinceV10:
create_react_agent has been moved to langchain.agents. Please update your
import to from langchain.agents import create_agent` -- verified directly by
calling it -- so this adapter uses `create_agent` (from the `langchain`
package, not `langgraph`), its documented replacement, throughout.

- The build root has NO outgoing edges (a leaf, e.g. `writer` in the shipped
  example): this adapter returns that agent's own compiled react graph
  directly -- there is nothing else in the picture, so the simplest, most
  directly runnable object is exactly what `create_agent` already produced.
- The build root HAS at least one outgoing edge (e.g. `coordinator` or
  `researcher`): every agent reachable from the build root (`BaseAdapter.
  _reachable_agents`) is built the same way, each wired with a HANDOFF TOOL
  per outgoing `interactions.yaml` edge (see "Edge mapping" below), and all
  of them are wired together as NODES of one parent `langgraph.graph.
  StateGraph(MessagesState)`, with a single `START -> <build root>` edge as
  the entry point. `builder.compile()` returns the final, ready-to-run
  `CompiledStateGraph` for the WHOLE multi-agent graph -- the build root's
  handoff tools (and every downstream agent's own handoff tools) are what
  actually drive routing between nodes at run time, not any edges added to
  `builder` beyond that one entry edge (verified: a `Command(goto=...,
  graph=Command.PARENT)` returned from a node's tool call is LangGraph's own
  documented mechanism for jumping to a *sibling* node in the parent graph,
  no static `builder.add_edge(a, b)` needed or even meaningful between
  agent nodes here).

Usage (mirroring cli.py's `_run_langgraph`):

    from commonadk import load

    project = load("common/")
    graph = project.build("coordinator", target="langgraph")  # CompiledStateGraph

    result = graph.invoke({"messages": [{"role": "user", "content": "Research EV adoption"}]})
    print(result["messages"][-1].content)

    # or, async:
    result = await graph.ainvoke({"messages": [{"role": "user", "content": "..."}]})

Verified against the installed packages: langgraph 1.2.11 (`langgraph.
graph.StateGraph`/`MessagesState`, `langgraph.types.Command`, `langgraph.
prebuilt.InjectedState`), langchain 1.3.17 (`langchain.agents.create_agent`,
`langchain.chat_models.init_chat_model`), langchain-core 1.6.0
(`langchain_core.tools.tool`/`InjectedToolCallId`), langchain-google-genai
4.3.6, langchain-anthropic 1.6.1, langchain-openai 1.6.0 -- all introspected
directly via `inspect.signature`/`inspect.getsource` and exercised during
M8, not taken from memory of the API (the installed stack has moved past
what older LangGraph tutorials describe -- see the `create_react_agent`
deprecation above).

Edge mapping -- PER-EDGE TARGETING, unlike every coarser adapter in this
codebase: LangGraph is the one target here where `interactions.yaml`'s edge
*targets* are fully, precisely expressible. Both `delegate` and `handoff`
edges map to the SAME mechanism in v1 (plan.md's stated intersection
decision, matching every other adapter) -- a HANDOFF TOOL, one per distinct
outgoing edge destination, named `transfer_to_<destination>` (see
`_make_handoff_tool` below) and added to the SOURCE agent's own tool list.
Unlike CrewAI's `allow_delegation=True` (crew-wide -- any member can reach
any other member once delegation is on at all) or AutoGen's `Swarm`
(name-string handoffs resolved against a flat team-wide participant list,
functionally global once inside the graph), a LangGraph handoff tool is a
distinct, individually-named, individually-invocable tool scoped to
EXACTLY the one agent it was built for -- an agent with edges to `x` and
`y` gets exactly two handoff tools, `transfer_to_x` and `transfer_to_y`, and
literally cannot reach any other reachable agent unless a matching edge (and
therefore a matching tool) exists for it. This is the most precise
`interactions.yaml` <-> SDK-native mapping of any adapter in this codebase.

Handoff mechanism, investigated not assumed: this adapter hand-rolls the
handoff tool itself using LangGraph's own `Command` primitive, rather than
depending on the separate `langgraph-supervisor` or `langgraph-swarm`
packages -- neither is installed, and per LangGraph's own current
multi-agent documentation, a `Command`-returning tool is the DOCUMENTED,
dependency-free way to hand off between agent nodes in a parent
`StateGraph`, not a workaround: a tool annotated to receive `InjectedState`
(the parent graph's running `MessagesState`) and `InjectedToolCallId`
returns `Command(goto=<destination node name>, update={"messages": [...]},
graph=Command.PARENT)` -- `graph=Command.PARENT` is what makes the jump
target a SIBLING node in the parent `StateGraph` rather than a node inside
the calling agent's own react-agent subgraph (verified via `Command`'s own
docstring, `langgraph.types.Command`: "graph: ... Command.PARENT: closest
parent graph"). The tool call itself never executes tool logic beyond
building this `Command` and a `ToolMessage` acknowledging the transfer, so
handoffs cost no extra model round-trip beyond the one that already decided
to call the tool.

KEY PROPERTY, investigated -- multi-parent graphs and cycles both build
successfully, verified directly (see test_adapter_langgraph.py): every
reachable agent is built exactly ONCE into a `dict[str, CompiledStateGraph]`
keyed by logical agent name (mirroring `_reachable_agents`'s own dedup, like
every other adapter's flat-registry construction), and `StateGraph.add_node`
is called once per dict entry. A destination reachable from two different
sources is simply the same node referenced by two different `transfer_to_*`
tools on two different source nodes -- no duplication, no special-casing. A
cycle back to the build root (e.g. `writer -> coordinator`) is just another
`transfer_to_coordinator` tool on `writer`, targeting a node that already
exists in the same `StateGraph` -- `Command(goto="coordinator", graph=
Command.PARENT)` resolves it by name at run time with no construction-time
recursion hazard whatsoever (LangGraph's own execution loop, not this
adapter, is what actually re-enters the coordinator node).

Model routing -- LiteLLM "provider/model" strings map onto langchain's
`init_chat_model` "provider:model" convention (verified via `inspect.
getsource` of `langchain.chat_models.base._parse_model`: it splits on the
first `:` and looks the left half up in a fixed `_BUILTIN_PROVIDERS` table,
which includes `google_genai`, `openai`, and `anthropic` among many others
this adapter does NOT ship a package for). Only the three providers this
project's `langgraph` extra actually installs an integration package for
get a real, verified path -- everything else is a clear unsupported-provider
error, matching the Claude Agent SDK and AutoGen adapters' policy (no
LiteLLM fallback here, unlike CrewAI):

- `gemini/<model>` -> `init_chat_model(f"google_genai:{model}", ...)`
  (needs `langchain-google-genai`, shipped in this extra). Verified this
  provider's chat model (`ChatGoogleGenerativeAI`) constructs EAGERLY and
  raises immediately (a pydantic `ValidationError`, "API key required for
  Gemini Developer API") if neither `GOOGLE_API_KEY` nor `GEMINI_API_KEY`
  is set and no `api_key` kwarg is given -- no network call either way, but
  a key-shaped string must exist somewhere. See "Offline construction"
  below.
- `openai/<model>` -> `init_chat_model(f"openai:{model}", ...)` (needs
  `langchain-openai`). Verified this provider's chat model (`ChatOpenAI`)
  is ALSO eager: raises `openai.OpenAIError: "Missing credentials..."`
  immediately if `OPENAI_API_KEY` isn't set and no `api_key` kwarg is given.
- `anthropic/<model>` -> `init_chat_model(f"anthropic:{model}", ...)` (needs
  `langchain-anthropic`). Verified this provider's chat model
  (`ChatAnthropic`) is LAZY, unlike the other two -- it constructs
  successfully with no `ANTHROPIC_API_KEY` set at all and no error until an
  actual API call is made. Documented here as an asymmetry across this
  adapter's three providers, not a bug: this adapter does not paper over it
  (each SDK client's own eagerness, or lack of it, is left exactly as the
  installed package defines it, matching every other adapter's restraint
  around errors it doesn't specifically own).
- Anything else (azure, bedrock, ollama, cohere, mistral, ...) raises a
  clear `ValueError` naming the agent, its resolved model string, and the
  fix options (a `gemini/`, `openai/`, or `anthropic/`-prefixed model; a
  different alias; or a `targets.langgraph.model` override) -- mirroring
  the Claude Agent SDK and AutoGen adapters' error style exactly.

The shipped research-crew example builds UNMODIFIED for this target, same
as AutoGen and CrewAI (and unlike Claude, which needs per-agent overrides):
`fast` (coordinator, writer) resolves to `gemini/gemini-2.5-flash` and
researcher's own model is `gemini/gemini-2.5-pro` directly -- both native
`google_genai` paths, no override needed.

Per-target override (`targets.langgraph.model` in `agent-config.yaml`):
always wins, and is passed through AS-IS to `init_chat_model(...)` -- so its
EXPECTED FORM is already langchain-native `"provider:model"` (e.g.
`"anthropic:claude-opus-4"`), not a bare model id and not a LiteLLM
`"provider/model"` string, matching every other adapter's "already SDK-
native form, passed through as-is" policy for overrides. If the provider
half isn't one `init_chat_model` recognizes (or its package isn't
installed), the SDK's own clear error surfaces unwrapped.

model_params: verified via direct construction that `temperature` and
`max_tokens` are accepted as constructor/`init_chat_model` kwargs
IDENTICALLY across all three shipped providers -- `ChatOpenAI` and
`ChatAnthropic` both declare a real `max_tokens` field, and
`ChatGoogleGenerativeAI` declares `max_output_tokens` but with a pydantic
`validation_alias="max_tokens"` AND `populate_by_name=True` (verified via
`model_fields`/`model_config`), so passing the keyword `max_tokens=...`
resolves correctly on all three -- no per-provider kwarg remapping is
needed, unlike AutoGen's shared-but-separately-typed clients (`_MODEL_
PARAM_MAP` below is a single flat map for exactly this reason). Any other
`model_params` key is warned-and-ignored, per this codebase's established
policy.

Tool wiring: each `ToolSpec`'s plain, typed, documented `tools.py` function
is passed straight through as a bare callable in the `tools=[...]` list --
verified via direct construction that `create_agent`/`langgraph.prebuilt.
ToolNode` wraps a bare callable into a `StructuredTool` itself, introspecting
its signature and docstring exactly like `tools.py`'s own contract already
guarantees (enforced upstream by `validation.py`), so no separate wrapping
is needed here -- the same simplicity as the AutoGen adapter's tool wiring.

Offline construction: like AutoGen's `OpenAIChatCompletionClient`/
`AnthropicChatCompletionClient`, the `google_genai` and `openai` chat model
classes construct their underlying provider client EAGERLY (see "Model
routing" above) -- `build()` for an agent on either of those providers fails
immediately, with the SDK's own unwrapped error text, if the matching env
var isn't set. `anthropic` is the one provider of the three that does NOT
require this. Tests in this codebase set fake `GOOGLE_API_KEY`/
`OPENAI_API_KEY`/`ANTHROPIC_API_KEY` values up front for exactly this
reason (see test_adapter_langgraph.py's `provider_keys_env` fixture,
mirroring test_adapter_autogen.py's) -- construction never makes a network
call, but a key-shaped string must exist for the two eager providers.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Annotated, Any

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool as lc_tool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

if TYPE_CHECKING:
    from ..models import AgentSpec, Project

from .base import BaseAdapter

# agent-config.yaml `model_params` key -> init_chat_model kwarg. One flat
# map for all three providers -- see module docstring, "model_params": the
# google_genai provider resolves the same `max_tokens` keyword via a
# pydantic validation_alias, so no per-provider remapping is needed here,
# unlike autogen_adapter.py.
_MODEL_PARAM_MAP = {
    "temperature": "temperature",
    "max_tokens": "max_tokens",
}

# LiteLLM provider prefix -> langchain `init_chat_model` provider name.
# Only providers this adapter's `langgraph` extra ships an integration
# package for -- see module docstring, "Model routing".
_PROVIDER_MAP = {
    "gemini": "google_genai",
    "openai": "openai",
    "anthropic": "anthropic",
}


class LangGraphAdapter(BaseAdapter):
    target = "langgraph"

    def build(self, project: "Project", agent_name: str) -> CompiledStateGraph:
        self._check_env(project, agent_name)

        reachable = self._reachable_agents(project, agent_name)  # agent_name first

        nodes: dict[str, CompiledStateGraph] = {}
        for name in reachable:
            nodes[name] = self._build_agent_node(project, name)

        has_outgoing = any(edge.from_ == agent_name for edge in project.graph.edges)
        if not has_outgoing:
            # Nothing for the build root to hand off to -- its own compiled
            # react agent IS the whole graph (see module docstring, "WHAT
            # build() RETURNS").
            return nodes[agent_name]

        builder: StateGraph = StateGraph(MessagesState)
        for name, node in nodes.items():
            builder.add_node(name, node)
        builder.add_edge(START, agent_name)
        return builder.compile()

    # -- per-agent node construction -----------------------------------

    def _build_agent_node(self, project: "Project", name: str) -> CompiledStateGraph:
        spec = project.agents[name]

        # One handoff tool per DISTINCT outgoing edge destination -- this is
        # the per-edge targeting this target is documented to honor
        # precisely (see module docstring, "Edge mapping"). `dict.fromkeys`
        # dedupes while keeping first-seen order (a source can have two
        # edges, e.g. one delegate and one handoff, to the same
        # destination -- that must still produce exactly one tool, not a
        # duplicate-named one).
        destinations = dict.fromkeys(
            edge.to for edge in project.graph.edges if edge.from_ == name
        )
        handoff_tools = [self._make_handoff_tool(dest) for dest in destinations]

        return create_agent(
            self._model_for(project, spec),
            tools=[t.func for t in spec.tools] + handoff_tools,
            system_prompt=spec.instructions,
            name=name,
        )

    @staticmethod
    def _make_handoff_tool(destination: str) -> Any:
        """Build a `transfer_to_<destination>` tool (see module docstring,
        "Handoff mechanism"): calling it returns a `Command` that jumps
        execution to the `destination` node of the closest PARENT graph.
        """
        tool_name = f"transfer_to_{destination}"

        @lc_tool(
            tool_name,
            description=f"Transfer the conversation to the '{destination}' agent.",
        )
        def handoff_tool(
            state: Annotated[MessagesState, InjectedState],
            tool_call_id: Annotated[str, InjectedToolCallId],
        ) -> Command:
            tool_message = ToolMessage(
                content=f"Successfully transferred to {destination}",
                name=tool_name,
                tool_call_id=tool_call_id,
            )
            return Command(
                goto=destination,
                update={"messages": [*state["messages"], tool_message]},
                graph=Command.PARENT,
            )

        return handoff_tool

    # -- model routing ------------------------------------------------------

    def _model_for(self, project: "Project", spec: "AgentSpec") -> Any:
        kwargs = self._model_param_kwargs(spec)

        override = spec.config.targets.get("langgraph", {})
        if "model" in override:
            # Per-target override: already langchain-native "provider:model"
            # form, passed through as-is -- see module docstring, "Per-
            # target override".
            return init_chat_model(override["model"], **kwargs)

        resolved = project.resolve_model(spec.name)  # LiteLLM-format string
        provider, sep, rest = resolved.partition("/")
        lc_provider = _PROVIDER_MAP.get(provider) if sep else None
        if lc_provider is not None:
            return init_chat_model(f"{lc_provider}:{rest}", **kwargs)

        raise ValueError(
            f"commonadk: agent {spec.name!r} resolves to model {resolved!r}, "
            f"but the LangGraph target ('langgraph') only ships native "
            f"model providers for 'gemini/...', 'openai/...', and "
            f"'anthropic/...' (see langgraph_adapter.py's module docstring, "
            f"'Model routing'). Fix this by either: using one of those "
            f"providers (e.g. 'openai/gpt-4o'), changing {spec.name}'s "
            f"model alias in config.yaml to one that resolves to a "
            f"supported provider, or adding a `targets.langgraph.model` "
            f"override to {spec.name}/agent-config.yaml with a langchain-"
            f"native 'provider:model' string (e.g. 'openai:gpt-4o') "
            f"understood by `init_chat_model`."
        )

    def _model_param_kwargs(self, spec: "AgentSpec") -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        for key, value in spec.config.model_params.items():
            mapped = _MODEL_PARAM_MAP.get(key)
            if mapped is None:
                warnings.warn(
                    f"{spec.name}: model_params key '{key}' is not supported "
                    f"by the LangGraph adapter and will be ignored",
                    stacklevel=2,
                )
                continue
            kwargs[mapped] = value
        return kwargs
