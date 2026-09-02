"""Claude Agent SDK adapter: `AgentSpec` -> a `claude_agent_sdk.ClaudeAgentOptions`.

WHAT `build()` RETURNS -- read this first: unlike the Google ADK and OpenAI
Agents adapters, this SDK has no persistent "agent object" to instantiate.
It is session/query-based: you call `claude_agent_sdk.query(prompt=...,
options=...)` (or open a `ClaudeSDKClient(options=...)`) and the SDK's
bundled Claude Code CLI runs the whole turn -- built-in tools, subagent
dispatch, everything -- as a subprocess driven by that one `options` object.
So `build()` returns a fully-wired `claude_agent_sdk.ClaudeAgentOptions`: the
requested agent's instructions as `system_prompt`, its resolved model as
`model`, its own `tools.py` functions registered as an in-process MCP server
and exposed through `allowed_tools`/`disallowed_tools`, and every other
agent reachable from it wired into `options.agents` as
`claude_agent_sdk.AgentDefinition` subagents (see "Edge mapping" below for
exactly how). Usage:

    import asyncio
    from claude_agent_sdk import query

    options = project.build("coordinator", target="claude")

    async def main():
        async for message in query(prompt="Research EV adoption", options=options):
            ...  # see cli.py's _run_claude for pulling out the final text

    asyncio.run(main())

Verified against the installed SDK: claude-agent-sdk 0.2.144
(`claude_agent_sdk.types`, `claude_agent_sdk.__init__`, and
https://code.claude.com/docs/en/agent-sdk/subagents, fetched during M5).

Tool wiring -- the SDK's in-process MCP mechanism: each reachable agent's
plain `tools.py` functions are wrapped with `claude_agent_sdk.tool(...)` (an
async handler + a JSON Schema built from the function's introspected
parameters -- see models.py's `ToolSpec.parameters`) and bundled into one
`claude_agent_sdk.create_sdk_mcp_server(...)` per agent, named
`"<agent>_tools"`. Every such server -- the build root's own and every
reachable subagent's -- is registered once in `options.mcp_servers` (a flat,
session-wide registry: `AgentDefinition.mcpServers` and the main session's
tool visibility both resolve server names against this one dict), and each
agent's own tools are addressed on the wire as `mcp__<agent>_tools__<name>`.

Per-agent tool restriction -- verified supported: `AgentDefinition.tools`
(confirmed via the SDK's docs, "AgentDefinition configuration" table) is a
restrictive allowlist when set -- "if omitted, inherits every tool available
to subagents" implies that when *given*, it is the agent's *entire* tool
set, not an addition to some base set. So every subagent in `options.agents`
gets `tools=[<its own mcp__..__ names>] + (["Agent"] if it has outgoing
edges, else nothing)`, and `mcpServers=["<agent>_tools"]` (or `[]` if it
declares no tools.py functions) to scope its MCP visibility to just its own
server. `ClaudeAgentOptions` has no equivalent restrictive field for the
*main* session (`tools` there only toggles the base set of *built-in*
Claude Code tools -- Read/Write/Bash/... -- off; MCP tool visibility is
governed separately and, per the SDK's docs, is not narrowed by it), so the
build root is restricted the other way: `tools=[]` turns off every built-in
tool (commonadk agents are pure custom-tool agents, not coding agents), and
`disallowed_tools` explicitly removes every *other* reachable agent's own
`mcp__..__` names from the root's tool set -- `disallowed_tools` is
documented to actually remove a tool "from the model's context", unlike
`allowed_tools`, which only pre-approves without restricting availability.

Edge mapping and nesting depth -- investigated, not assumed: `AgentDefinition`
has no nested "agents" field of its own -- subagents are declared exactly
once, in a single flat `dict[str, AgentDefinition]` on `ClaudeAgentOptions`
(`options.agents`), so nesting *declarations* really are one level deep
(main session + a flat set), matching the M5 plan's premise. But per the
SDK's docs (https://code.claude.com/docs/en/agent-sdk/subagents,
"Subagents can also spawn subagents of their own" -- true up to
`CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH`, default 3), any agent whose own
`tools` includes `"Agent"` can invoke *any* name registered in that same
flat `options.agents` dict via the Agent tool's `subagent_type` argument --
invocation is a name lookup against the whole flat registry, not scoped to
a parent-declared child list. So a deeper `interactions.yaml` edge (e.g.
`researcher -> writer` when building `coordinator`) IS honestly
representable: this adapter registers every agent *reachable* from the
build root (the full transitive closure, not just root's direct children)
into that one flat `options.agents` dict, and grants `"Agent"` tool access
only to the reachable agents that actually have an outgoing edge in
`interactions.yaml` -- so an agent can delegate exactly when the graph says
it can, even though the underlying mechanism (a global name lookup) is
technically capable of more. This also means multi-parent graphs and cycles
in the reachable subgraph -- both rejected or specially handled by the other
two adapters -- need no special handling here at all: `options.agents` is a
plain `dict[str, AgentDefinition]` keyed by logical agent name (mirroring
`_reachable_agents`'s own dedup), so a name reachable by two paths, or a
path that cycles back to the build root, is simply the same dict entry
visited twice; the build root itself is excluded from `options.agents` (it
*is* `options`), so a cycle back to it is a no-op.

Model routing: Anthropic-native only -- there is no LiteLLM path in this
SDK (`ClaudeAgentOptions.model` / `AgentDefinition.model` take a bare Claude
model id or alias such as `"sonnet"`/`"opus"`/`"haiku"`/`"inherit"`, per
`claude_agent_sdk.types.AgentDefinition`'s `model` field comment -- there is
no wrapper analogous to google-adk's `LiteLlm` or openai-agents'
`LitellmModel`). A resolved `anthropic/...` LiteLLM-format string maps to
the bare id after the slash (e.g. `anthropic/claude-sonnet-5` ->
`claude-sonnet-5`). Any other provider raises a clear `ValueError` naming
the offending agent and its resolved model string, since this target simply
cannot run it. A per-target `targets.claude.model` override in
`agent-config.yaml` always wins and is passed through as-is (already
SDK-native form) -- this is the escape hatch for a project whose
`default_model`/aliases target other providers, exactly like the shipped
research-crew example (gemini models): building it for target `"claude"`
with no override fails loudly by design (tested); adding
`targets.claude.model: claude-sonnet-5` to each agent's `agent-config.yaml`
is what makes the example buildable here too.

model_params: the installed SDK exposes no per-request sampling controls
analogous to `temperature`/`max_tokens` anywhere in `claude_agent_sdk.types`.
Re-verified directly against installed claude-agent-sdk 0.2.144 via
`dataclasses.fields(ClaudeAgentOptions)` and `dataclasses.fields(
AgentDefinition)` for this project's full candidate list (`temperature`,
`max_tokens`, `top_p`, `top_k`, `stop`/`stop_sequences`, `presence_penalty`,
`frequency_penalty`, `seed`) -- none exist on either dataclass; the closest
fields, `thinking`/`effort`/`max_thinking_tokens`/`max_turns`, are
reasoning-effort and turn-budget controls, not sampling parameters. So every
`model_params` key is unsupported here and this adapter warns-and-ignores
all of them, per the same policy the other five adapters apply to whichever
keys *they* don't recognize.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from claude_agent_sdk import (
    AgentDefinition,
    ClaudeAgentOptions,
    create_sdk_mcp_server,
)
from claude_agent_sdk import tool as sdk_tool

if TYPE_CHECKING:
    from ..models import AgentSpec, Project, ToolSpec

from .base import BaseAdapter

# agent-config.yaml `model_params` keys -> claude_agent_sdk field: none exist
# for temperature, max_tokens, top_p, top_k, stop, presence_penalty,
# frequency_penalty, or seed (see module docstring, "model_params") -- every
# key is warned-and-ignored.
_MODEL_PARAM_MAP: dict[str, str] = {}

# models.py ToolParameter.type strings -> JSON Schema "type" values. Anything
# not covered falls back to "string", matching the SDK's own
# `_python_type_to_json_schema` fallback for unrecognized annotations.
_JSON_SCHEMA_TYPE = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
}


class ClaudeAgentSDKAdapter(BaseAdapter):
    target = "claude"

    def build(self, project: "Project", agent_name: str) -> ClaudeAgentOptions:
        self._check_env(project, agent_name)

        reachable = self._reachable_agents(project, agent_name)  # includes agent_name
        has_children = {
            name: any(edge.from_ == name for edge in project.graph.edges)
            for name in reachable
        }

        mcp_servers: dict[str, Any] = {}
        own_tool_names: dict[str, list[str]] = {}
        for name in reachable:
            spec = project.agents[name]
            if spec.tools:
                server_key = f"{name}_tools"
                mcp_servers[server_key] = create_sdk_mcp_server(
                    name=server_key,
                    tools=[self._sdk_tool_for(t) for t in spec.tools],
                )
                own_tool_names[name] = [
                    f"mcp__{server_key}__{t.name}" for t in spec.tools
                ]
            else:
                own_tool_names[name] = []

        agents: dict[str, AgentDefinition] = {}
        for name in reachable:
            if name == agent_name:
                continue
            spec = project.agents[name]
            self._warn_unsupported_model_params(spec)
            tools = list(own_tool_names[name])
            if has_children[name]:
                tools.append("Agent")
            agents[name] = AgentDefinition(
                description=spec.config.description,
                prompt=spec.instructions,
                tools=tools,
                model=self._model_for(project, spec),
                mcpServers=[f"{name}_tools"] if spec.tools else [],
            )

        root_spec = project.agents[agent_name]
        self._warn_unsupported_model_params(root_spec)
        root_tools = list(own_tool_names[agent_name])
        if has_children[agent_name]:
            root_tools.append("Agent")

        disallowed = [
            tool_name
            for name in reachable
            if name != agent_name
            for tool_name in own_tool_names[name]
        ]

        return ClaudeAgentOptions(
            system_prompt=root_spec.instructions,
            model=self._model_for(project, root_spec),
            tools=[],  # no built-in Claude Code tools -- see module docstring
            allowed_tools=root_tools,
            disallowed_tools=disallowed,
            mcp_servers=mcp_servers,
            agents=agents,
        )

    # -- tool wiring (SDK in-process MCP mechanism) ------------------------

    def _sdk_tool_for(self, tool_spec: "ToolSpec") -> Any:
        properties: dict[str, Any] = {}
        required: list[str] = []
        for param in tool_spec.parameters:
            properties[param.name] = {"type": _JSON_SCHEMA_TYPE.get(param.type, "string")}
            if param.required:
                required.append(param.name)
        input_schema = {"type": "object", "properties": properties, "required": required}

        func = tool_spec.func

        async def handler(args: dict[str, Any]) -> dict[str, Any]:
            result = func(**args)
            return {"content": [{"type": "text", "text": str(result)}]}

        return sdk_tool(tool_spec.name, tool_spec.description, input_schema)(handler)

    # -- model routing ------------------------------------------------------

    def _model_for(self, project: "Project", spec: "AgentSpec") -> Any:
        override = spec.config.targets.get("claude", {})
        if "model" in override:
            # Per-target override: already SDK-native form, passed through as-is.
            return override["model"]

        resolved = project.resolve_model(spec.name)  # LiteLLM-format string
        provider, sep, rest = resolved.partition("/")
        if sep and provider == "anthropic":
            return rest  # bare native model id, e.g. "claude-sonnet-5"

        raise ValueError(
            f"commonadk: agent {spec.name!r} resolves to model {resolved!r}, "
            f"but the Claude Agent SDK target ('claude') runs Anthropic "
            f"models only -- there is no LiteLLM path for this target. Fix "
            f"this by either: using an 'anthropic/...' model (e.g. "
            f"'anthropic/claude-sonnet-5'), changing {spec.name}'s model "
            f"alias in config.yaml to one that resolves to an "
            f"'anthropic/...' string, or adding a `targets.claude.model` "
            f"override to {spec.name}/agent-config.yaml with an SDK-native "
            f"model id or alias (e.g. 'claude-sonnet-5', 'sonnet', 'opus', "
            f"'haiku')."
        )

    def _warn_unsupported_model_params(self, spec: "AgentSpec") -> None:
        for key in spec.config.model_params:
            if key in _MODEL_PARAM_MAP:
                continue
            warnings.warn(
                f"{spec.name}: model_params key '{key}' is not supported "
                f"by the Claude Agent SDK adapter and will be ignored",
                stacklevel=2,
            )
