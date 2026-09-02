"""AutoGen adapter: `AgentSpec` -> live `autogen_agentchat` objects.

This targets Microsoft's current AutoGen stack (`autogen-agentchat` /
`autogen-core` / `autogen-ext`, 0.4+), NOT the community `ag2` fork -- the
two forked from the same original project and now have unrelated APIs.
Verified against the installed packages: autogen-agentchat 0.7.5,
autogen-core 0.7.5, autogen-ext 0.7.5 (`autogen_agentchat.agents`,
`autogen_agentchat.teams`, `autogen_agentchat.base`, and
`autogen_ext.models.{openai,anthropic}`, introspected via `inspect.signature`
/ `inspect.getsource` and exercised directly during M7 -- not taken from
memory of the API, which is unreliable here given the fork).

WHAT `build()` RETURNS -- read this first: like the Google ADK and OpenAI
Agents adapters, this SDK has real persistent agent objects
(`autogen_agentchat.agents.AssistantAgent`), each wired with a
`model_client`, its own `tools.py` functions, and a `handoffs` list of
target agent *names* (see "Edge mapping" below). But an `AssistantAgent`
with handoffs configured needs a *team* to actually route those handoffs at
run time -- AutoGen's own mechanism for this is `autogen_agentchat.teams.
Swarm`, "a group chat team that selects the next speaker based on handoff
message[s]" (its own docstring). Investigated, not assumed, whether a lone
`AssistantAgent` with handoffs is runnable on its own: it is NOT -- handoffs
only take effect inside a `Swarm` (or another team); a bare `AssistantAgent.
run()` just answers once and never consults `.handoffs` at all.

So this adapter builds every reachable agent once (see "Edge mapping"), then
picks the return shape based on whether the build root actually has
somewhere to hand off to:

- The build root has NO outgoing edges (a leaf, e.g. `writer` in the shipped
  example): there is nothing to route to, so this adapter returns the bare
  `AssistantAgent` -- the simplest, most directly runnable object for that
  case. Usage: `result = await agent.run(task="...")`.
- The build root HAS at least one outgoing edge: this adapter returns a
  ready-to-run `autogen_agentchat.teams.Swarm` whose `participants` are
  every reachable agent, build root first (`BaseAdapter._reachable_agents`
  already returns `agent_name` at index 0, which is also exactly the
  property `Swarm` requires: verified via `Swarm.__init__` -- the first
  participant becomes the initial speaker). Usage:
  `result = await team.run(task="...")`.

Usage (mirroring cli.py's `_run_autogen`):

    import asyncio
    from commonadk import load

    project = load("common/")
    built = project.build("coordinator", target="autogen")  # a Swarm here

    async def main():
        result = await built.run(task="Research EV adoption")
        print(result.messages[-1].content)

    asyncio.run(main())

`max_turns` on the returned `Swarm` -- a deliberate, documented default, not
an SDK requirement: investigated directly against `Swarm`'s own docs and
`BaseGroupChat.__init__` -- with neither a `termination_condition` nor a
`max_turns` set, a group chat "will run indefinitely": if the current
speaker doesn't send a handoff message, `SwarmGroupChatManager` just lets
the same speaker go again, and there is no other built-in stop condition
(unlike a bare `AssistantAgent.run()`, which always returns after exactly
one turn). Since `commonadk run` needs a single execution that reliably
terminates (matching every other adapter's `_run_*` in cli.py), and
`max_turns`/`termination_condition` are `Swarm` CONSTRUCTOR-only fields with
no equivalent per-call override on `run`/`run_stream` (verified via
`inspect.signature`), this adapter sets `max_turns=len(reachable)` on every
`Swarm` it returns: exactly enough speaker-turns for one full pass down a
linear delegate/handoff chain (root speaks and hands off, ..., the final
agent speaks its answer and the budget is exhausted right after). This is a
heuristic, not a guarantee, for branchier graphs (multi-parent, cycles) --
documented here as a known v1 limitation, not silently assumed correct. A
caller who wants a different turn budget or an interactive, longer-running
conversation should not rely on the `Swarm` this adapter returns for that;
building one directly from `AssistantAgent`s (this adapter's own approach,
above) with an explicit `termination_condition` is the escape hatch.

Edge mapping (v1 intersection decision, plan.md "Edge semantics v1", same
call as openai_agents.py): both `delegate` and `handoff` edges map to
AutoGen's one handoff mechanism -- `AssistantAgent(handoffs=[...])`. commonadk
does not yet distinguish them for this target either.

KEY PROPERTY, investigated not assumed -- handoff targets are plain NAME
STRINGS, not object references: `AssistantAgent.__init__` accepts
`handoffs: List[HandoffBase | str] | None`, and a bare `str` is wrapped as
`HandoffBase(target=that_string)` (verified via `inspect.getsource`) --
`Swarm` resolves those names against its own `participants` list by name at
run time, there is no parent-tracking or "already referenced" guard
anywhere in construction. This makes multi-parent graphs and cycles even
more trivially fine here than in openai_agents.py (which at least memoizes
live object references): this adapter builds one `AssistantAgent` per
logical agent name (memoized in a `dict[str, AssistantAgent]`, matching
`_reachable_agents`'s own dedup) and each agent's `handoffs` list is just
`[edge.to for edge in ... if edge.from_ == name]` -- plain strings, so a name
reachable by two paths or a path that cycles back to the build root needs no
special handling at all: it is simply the same dict entry, and a cycle back
to `agent_name` is just another string in some other agent's `handoffs`
list, not a construction hazard (`agent_name` itself is never excluded from
`memo`, unlike the Claude/CrewAI adapters' flat registries, since here
"being referenced by name" carries no risk of infinite recursion or
double-registration).

Tool wiring: `AssistantAgent(tools=[...])` accepts PLAIN CALLABLES directly
(verified via `inspect.getsource` of `AssistantAgent.__init__`) -- it wraps
each with `autogen_core.tools.FunctionTool(tool, description=tool.__doc__)`
itself, introspecting the function's signature and docstring exactly like
every `tools.py` function already provides (enforced upstream by
validation.py). So this adapter passes `[t.func for t in spec.tools]`
straight through with no wrapping of its own -- simpler than every other
adapter in this codebase.

Model routing -- investigated against the installed `autogen_ext.models`
package tree (`anthropic`, `azure`, `ollama`, `openai`, ... submodules;
`importlib.metadata.metadata("autogen-ext").get_all("Requires-Dist")` for
the full extras list). Three providers get a real, verified path; anything
else is a clear unsupported-provider error:

- `openai/<model>` -> `autogen_ext.models.openai.OpenAIChatCompletionClient
  (model=<bare id>)` -- the native OpenAI client, per plan.md's explicit
  instruction for this provider.
- `anthropic/<model>` -> `autogen_ext.models.anthropic.
  AnthropicChatCompletionClient(model=<bare id>, model_info=...)` -- a real,
  separately-shipped native client module (needs the `anthropic` package,
  already a transitive dependency of this project's `claude`/`crewai`
  extras and pinned directly in this adapter's own `autogen` extra).
  CRITICAL LANDMINE, verified not assumed: this client's bundled model-name
  table (`autogen_ext.models.anthropic._model_info._MODEL_INFO`) only knows
  a handful of hardcoded, DATED model ids (e.g. `claude-opus-4-20250514`)
  and falls back to fuzzy prefix-matching for anything else -- and that
  fallback is buggy for exactly the kind of aliased model id this project's
  own examples use: `"claude-sonnet-5".startswith("claude-2.0".split("-2")
  [0])` == `"claude-sonnet-5".startswith("claude")` == True, so an unrelated
  legacy entry (`claude-2.0`, `function_calling: False`) silently wins the
  match -- verified directly: constructing the client on `"claude-sonnet-5"`
  with no explicit `model_info` returns `function_calling: False`, and then
  `AssistantAgent.__init__` raises "The model does not support function
  calling" as soon as this adapter passes any tools/handoffs. This adapter
  works around it by ALWAYS passing an explicit `model_info` for this
  provider (`_ANTHROPIC_MODEL_INFO` below, `function_calling: True`),
  bypassing the stale table entirely rather than trusting it for a model id
  it clearly does not know about.
- `gemini/<model>` -> also `OpenAIChatCompletionClient(model=<bare id>,
  model_info=...)`. There is no separate native Gemini client class shipped
  anywhere in `autogen_ext.models` (the `autogen-ext[gemini]` extra only
  pulls in `google-genai`, used by the unrelated `semantic-kernel` optional
  integration, not by any model-client class) -- but `OpenAIChatCompletionClient.
  __init__` itself special-cases Gemini: verified via `inspect.getsource`,
  when the model name starts with `"gemini-"` it automatically points
  `base_url` at Gemini's OpenAI-compatible endpoint
  (`GEMINI_OPENAI_BASE_URL`) and reads `GEMINI_API_KEY` from the
  environment if no `api_key` is given -- this genuinely IS "the shipped
  Gemini path", it just lives inside the OpenAI client rather than a
  dedicated module. Its bundled model-info table is INCOMPLETE, not buggy
  (verified: `gemini-2.5-flash` is listed, but `gemini-2.5-pro` -- used
  directly by the shipped example's `researcher` agent -- is not, and
  raises `"model_info is required when model name is not a valid OpenAI
  model"`), so this adapter always passes an explicit `model_info` here
  too (`_GEMINI_MODEL_INFO` below), for the same reason as the Anthropic
  path: don't trust an incomplete/stale table for a model id it might not
  recognize. The base-url/api-key special-casing runs unconditionally
  before that table is even consulted, so passing `model_info` explicitly
  does not disable it -- verified by constructing the client both ways.
- Anything else (azure, bedrock, ollama, cohere, mistral, ...) raises a
  clear `ValueError` naming the agent, its resolved model string, and the
  fix options (an `openai/`, `anthropic/`, or `gemini/`-prefixed model; a
  different alias; or a `targets.autogen.model` override).

Unlike the Claude Agent SDK adapter (M5), this target needs NO per-target
model overrides added to the shipped research-crew example to make it
buildable: the example's `fast` alias resolves to `gemini/gemini-2.5-flash`
and `smart` to `anthropic/claude-sonnet-5`, and researcher's own
`gemini/gemini-2.5-pro` is used directly -- all three are covered by the
native Gemini/Anthropic paths above with no escape hatch needed.

Per-target override (`targets.autogen.model` in `agent-config.yaml`):
always wins, and is passed through as the bare model id to
`OpenAIChatCompletionClient` -- the "default client" of this adapter (the
same one the `openai/...` provider branch above uses), with NO explicit
`model_info` (unlike the `anthropic/gemini` provider branches): an override
is assumed to already be a valid, SDK-native identifier the project author
vouches for, exactly like every other adapter's override handling ("already
SDK-native form, passed through as-is"). If that id isn't in
`OpenAIChatCompletionClient`'s own known-model table, the SDK's own clear
`model_info is required` error surfaces -- at which point the fix is the
same escape hatch every other adapter documents for its overrides: it needs
to be a model this client actually knows, or the caller composes their own
client outside `project.build(...)`.

model_params: `OpenAIChatCompletionClient` and `AnthropicChatCompletionClient`
DO NOT share one parameter set -- investigated at the level that actually
matters (the runtime whitelist each client filters constructor kwargs
through before building its `create_args`, `_create_args_from_config` in
each client's own module), not just each client's `CreateArguments`
`TypedDict` type hints (which turn out to be a red herring here: passing an
unsupported kwarg like `seed` to `AnthropicChatCompletionClient` does NOT
raise at construction time -- it is silently accepted and then silently
DROPPED by that filter, never reaching the Anthropic API, which is worse
than an error if this adapter mapped it blindly). Verified directly against
both real whitelists: `autogen_ext.models.openai._openai_client.
create_kwargs` contains `temperature`, `max_tokens`, `top_p`, `stop`,
`presence_penalty`, `frequency_penalty`, `seed` (no `top_k`);
`autogen_ext.models.anthropic._anthropic_client.anthropic_message_params`
contains `temperature`, `max_tokens`, `top_p`, `top_k`, `stop_sequences` (no
`presence_penalty`, `frequency_penalty`, `seed`, and the key is
`stop_sequences`, not `stop`). So this adapter maps two SEPARATE dicts,
`_OPENAI_MODEL_PARAM_MAP` (used for the `openai`/`gemini` provider branches
and the per-target override, all three of which build an
`OpenAIChatCompletionClient`) and `_ANTHROPIC_MODEL_PARAM_MAP` (the
`anthropic` provider branch only) -- unlike every flat-single-map adapter in
this codebase. Any key absent from whichever map applies is
warned-and-ignored, per the same policy every other adapter applies to keys
it doesn't map.

Offline construction -- a real difference from every other adapter here,
investigated not assumed: `OpenAIChatCompletionClient`/
`AnthropicChatCompletionClient.__init__` EAGERLY construct the underlying
`openai.AsyncOpenAI`/`anthropic.AsyncAnthropic` client right there in
`build()` -- and that raises immediately (`openai.OpenAIError: "Missing
credentials..."`) if no `api_key` is given AND the matching env var
(`OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY`) isn't set --
verified directly, with the relevant env vars cleared. This is unlike
openai-agents' `Agent` (no client touched until `Runner.run` actually
executes) and unlike this project's `requires.env` mechanism (which is for
an agent's own tool-level env vars, e.g. `TAVILY_API_KEY` -- model-provider
auth has never been part of that contract for any target). Net effect:
`build()` for this target fails loudly on a missing provider API key all by
itself, with the SDK's own error text, un-wrapped -- this adapter does not
catch or re-word it, the same restraint every other adapter shows for
errors it doesn't specifically own. Tests in this codebase set fake
`OPENAI_API_KEY`/`ANTHROPIC_API_KEY`/`GEMINI_API_KEY` values up front for
exactly this reason (see test_adapter_autogen.py) -- construction never
makes a network call, but it does require *a* key-shaped string to exist
somewhere.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import Swarm
from autogen_core.models import ModelFamily, ModelInfo
from autogen_ext.models.anthropic import AnthropicChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient

if TYPE_CHECKING:
    from ..models import AgentSpec, Project

from .base import BaseAdapter

# agent-config.yaml `model_params` key -> OpenAIChatCompletionClient
# constructor kwarg. Used for the `openai`/`gemini` provider branches and the
# per-target override (see module docstring, "model_params").
_OPENAI_MODEL_PARAM_MAP = {
    "temperature": "temperature",
    "max_tokens": "max_tokens",
    "top_p": "top_p",
    "stop": "stop",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
    "seed": "seed",
}

# agent-config.yaml `model_params` key -> AnthropicChatCompletionClient
# constructor kwarg. Used for the `anthropic` provider branch only -- this
# client's genuinely-accepted parameter set is smaller and differently named
# (`stop` -> `stop_sequences`) than the OpenAI-family client above (see
# module docstring, "model_params").
_ANTHROPIC_MODEL_PARAM_MAP = {
    "temperature": "temperature",
    "max_tokens": "max_tokens",
    "top_p": "top_p",
    "top_k": "top_k",
    "stop": "stop_sequences",
}

# Explicit model_info for the `anthropic/...` provider branch -- bypasses
# autogen_ext's stale/buggy bundled Anthropic model-name table entirely (see
# module docstring, "Model routing"). `family: ModelFamily.UNKNOWN` is
# deliberate: this adapter doesn't know or assert which Claude generation a
# given alias/model id maps to, only that it is Anthropic-native and
# supports function calling (a requirement of every commonadk agent that
# has tools or outgoing edges).
_ANTHROPIC_MODEL_INFO: ModelInfo = {
    "vision": False,
    "function_calling": True,
    "json_output": True,
    "family": ModelFamily.UNKNOWN,
    "structured_output": False,
    "multiple_system_messages": False,
}

# Explicit model_info for the `gemini/...` provider branch -- bypasses
# autogen_ext's incomplete bundled Gemini model-name table (see module
# docstring, "Model routing"). Vision/structured_output reflect what every
# modern Gemini model actually supports; family is intentionally generic
# for the same reason as the Anthropic table above.
_GEMINI_MODEL_INFO: ModelInfo = {
    "vision": True,
    "function_calling": True,
    "json_output": True,
    "family": ModelFamily.UNKNOWN,
    "structured_output": True,
    "multiple_system_messages": False,
}


class AutoGenAdapter(BaseAdapter):
    target = "autogen"

    def build(self, project: "Project", agent_name: str) -> Any:
        self._check_env(project, agent_name)

        reachable = self._reachable_agents(project, agent_name)  # agent_name first

        agents: dict[str, AssistantAgent] = {}
        for name in reachable:
            spec = project.agents[name]
            handoff_targets = [
                edge.to for edge in project.graph.edges if edge.from_ == name
            ]
            agents[name] = AssistantAgent(
                name=spec.name,
                model_client=self._client_for(project, spec),
                tools=[t.func for t in spec.tools],
                handoffs=handoff_targets,
                system_message=spec.instructions,
                description=spec.config.description,
            )

        has_outgoing = any(edge.from_ == agent_name for edge in project.graph.edges)
        if not has_outgoing:
            # Nothing for the build root to hand off to -- handoffs only do
            # anything inside a team, so the bare agent is the honest,
            # directly runnable object here (see module docstring, "WHAT
            # build() RETURNS").
            return agents[agent_name]

        participants = [agents[name] for name in reachable]  # root first
        return Swarm(participants, max_turns=len(reachable))

    # -- model routing ------------------------------------------------------

    def _client_for(self, project: "Project", spec: "AgentSpec") -> Any:
        override = spec.config.targets.get("autogen", {})
        if "model" in override:
            # Per-target override: passed through as the bare model id to
            # the default client, no explicit model_info -- see module
            # docstring, "Per-target override". Always the OpenAI-family
            # client, so the OpenAI param map applies.
            kwargs = self._model_param_kwargs(spec, _OPENAI_MODEL_PARAM_MAP)
            return OpenAIChatCompletionClient(model=override["model"], **kwargs)

        resolved = project.resolve_model(spec.name)  # LiteLLM-format string
        provider, sep, rest = resolved.partition("/")
        if sep and provider == "openai":
            kwargs = self._model_param_kwargs(spec, _OPENAI_MODEL_PARAM_MAP)
            return OpenAIChatCompletionClient(model=rest, **kwargs)
        if sep and provider == "anthropic":
            kwargs = self._model_param_kwargs(spec, _ANTHROPIC_MODEL_PARAM_MAP)
            return AnthropicChatCompletionClient(
                model=rest, model_info=_ANTHROPIC_MODEL_INFO, **kwargs
            )
        if sep and provider == "gemini":
            kwargs = self._model_param_kwargs(spec, _OPENAI_MODEL_PARAM_MAP)
            return OpenAIChatCompletionClient(
                model=rest, model_info=_GEMINI_MODEL_INFO, **kwargs
            )

        raise ValueError(
            f"commonadk: agent {spec.name!r} resolves to model {resolved!r}, "
            f"but the AutoGen target ('autogen') only ships native model "
            f"clients for 'openai/...', 'anthropic/...', and 'gemini/...' "
            f"providers (see autogen_adapter.py's module docstring, 'Model "
            f"routing'). Fix this by either: using one of those providers "
            f"(e.g. 'openai/gpt-4o'), changing {spec.name}'s model alias in "
            f"config.yaml to one that resolves to a supported provider, or "
            f"adding a `targets.autogen.model` override to "
            f"{spec.name}/agent-config.yaml with a bare model id understood "
            f"by autogen_ext's OpenAIChatCompletionClient."
        )

    def _model_param_kwargs(
        self, spec: "AgentSpec", param_map: dict[str, str]
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        for key, value in spec.config.model_params.items():
            mapped = param_map.get(key)
            if mapped is None:
                warnings.warn(
                    f"{spec.name}: model_params key '{key}' is not supported "
                    f"by the AutoGen adapter and will be ignored",
                    stacklevel=2,
                )
                continue
            kwargs[mapped] = value
        return kwargs
