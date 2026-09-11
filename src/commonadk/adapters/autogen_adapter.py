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
picks the return shape based on whether the build root actually has a
`handoff` edge to route -- NOT "any outgoing edge" (a build root with only
`delegate` edges, e.g. coordinator in the shipped example, needs no `Swarm`
at all: its delegate targets are wired directly into its own `tools` via
`AgentTool`, so the bare agent is already fully self-contained and
runnable -- see "Edge mapping" above):

- The build root has NO outgoing HANDOFF edges (either a true leaf with no
  edges at all, e.g. `writer` in the shipped example, OR a root whose edges
  are all `delegate`, e.g. `coordinator`): there is nothing for a `Swarm` to
  route, so this adapter returns the bare `AssistantAgent` -- the simplest,
  most directly runnable object for that case, its own `AgentTool`-wrapped
  delegate calls included. Usage: `result = await agent.run(task="...")`.
- The build root HAS at least one outgoing HANDOFF edge: this adapter
  returns a ready-to-run `autogen_agentchat.teams.Swarm` whose
  `participants` are every agent reachable via `handoff` edges ONLY, build
  root first (`BaseAdapter._reachable_via(..., {"handoff"})` returns
  `agent_name` at index 0, which is also exactly the property `Swarm`
  requires: verified via `Swarm.__init__` -- the first participant becomes
  the initial speaker). Usage: `result = await team.run(task="...")`.

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

Edge mapping -- THIS ADAPTER HONORS THE DELEGATE/HANDOFF DISTINCTION
(GitHub issue #10's first checkbox), not the v1 collapsed mapping. The
issue only asked this adapter to be "investigated" (unlike LangGraph/Google
ADK/OpenAI Agents, named explicitly) -- investigation found a second, real
mechanism installed alongside `AssistantAgent(handoffs=[...])`:
`autogen_agentchat.tools.AgentTool` (autogen-agentchat 0.7.5, verified via
`inspect.getsource`). Its own docstring: "Tool that can be used to run a
task using an agent. The tool returns the result of the task execution... as
a TaskResult object" -- wraps a `BaseChatAgent`, is added to another agent's
own `tools=[...]` list like any other tool, and runs the wrapped agent to
completion and returns its result INTO THE CALLING AGENT'S OWN turn. That is
exactly this project's `delegate` (a sub-call that returns), matching
`AssistantAgent(handoffs=[...])`'s `Swarm`-routed conversation TRANSFER
(never returns) being exactly `handoff`. So:

- `handoff` edges map to `AssistantAgent(handoffs=[...])` (unchanged) --
  target names as plain strings, resolved by `Swarm` at run time.
- `delegate` edges map to a tool wrapping the destination's OWN independent
  build, appended to the source's `tools` list (new) -- see "Recursive
  construction, and the AgentTool/TeamTool split" below for why it is
  sometimes `AgentTool` and sometimes `TeamTool`.

Recursive construction, and the AgentTool/TeamTool split: a `delegate`
destination is built by recursing into THIS ADAPTER'S OWN `build()` (not a
separate code path) -- exactly like Google ADK's and LangGraph's adapters
after this same feature, and for the same reason: a delegate destination
can itself have further outgoing edges of either type, and `build()`
already knows how to decide "bare agent, or does this need a `Swarm`" for
any agent, root or not. That recursive `build()` call returns one of two
shapes (see "WHAT build() RETURNS" below, re-scoped to handoff edges only):

- A bare `AssistantAgent` (destination has no outgoing HANDOFF edges of its
  own) -- wrapped in `autogen_agentchat.tools.AgentTool(agent=<that
  AssistantAgent>)`. Verified via `inspect.getsource`: `AgentTool` "wraps
  an agent" so it "allows an agent to be called as a tool"; "the agent's
  output is returned as the tool's result" -- a sub-call that returns,
  exactly `delegate`.
- A `Swarm` team (destination itself has outgoing handoff edges, so its own
  build wires a whole routable sub-team) -- wrapped in `autogen_agentchat.
  tools.TeamTool(team=<that Swarm>, name=f"delegate_to_{dest}",
  description=...)` instead, since `AgentTool.__init__` only accepts a
  `BaseChatAgent`, not a `Team`/`BaseGroupChat` (verified via `inspect.
  signature`) -- `TeamTool` is the SDK's own team-shaped counterpart,
  same "runs to completion, returns the result, caller's turn continues"
  contract, just for a multi-agent sub-team instead of a single agent.
  Delegating to a destination that itself needs to internally hand off
  is exactly the case a plain `AgentTool` cannot represent, and `TeamTool`
  exists in this SDK for precisely that reason.

`AgentTool` derives its exposed tool name from `agent.name` itself
(verified via `inspect.getsource` of `TaskRunnerTool.__init__`, which
`AgentTool` calls with `agent.name`/`agent.description` and no override
parameter at all) -- so an `AgentTool`-wrapped delegate tool is named
exactly the destination's agent name (e.g. `"researcher"`), while a
`TeamTool`-wrapped one is named `f"delegate_to_{dest}"` (this adapter's own
explicit choice, since `TeamTool` requires a `name` argument). This is a
real, SDK-imposed naming asymmetry between the two branches, not an
oversight -- documented here rather than worked around, matching this
project's restraint around every other SDK-owned quirk in this file (e.g.
the OpenAI-vs-Anthropic `model_params` map split above).

Recursion and cycles: since a `delegate` edge now recurses into a fresh,
independent `build()` call for its destination (rather than referencing an
already-built object the way `handoffs` name-strings do), a cycle closed
purely by `delegate` edges (or a mix of the two types) is an unbounded-
recursion hazard at construction time, exactly like Google ADK's and
LangGraph's adapters after this same feature -- `build()` therefore threads
one `_delegate_ancestors` chain through this recursion and raises a clear
`ValueError` before ever recursing past a repeated name, rather than
overflowing the stack. A cycle closed ENTIRELY by `handoff` edges is still
no hazard at all, exactly as before this feature (see "KEY PROPERTY"
above) -- `Swarm` resolves those by name at run time, with no construction-
time recursion involved.

WHAT build() RETURNS is now scoped to HANDOFF edges specifically, not "any
outgoing edge": a build root with only `delegate` edges (e.g.
`coordinator` in the shipped example) needs no `Swarm` at all -- its
delegate targets are wired directly into its own `tools` via
`AgentTool`/`TeamTool`, so the bare `AssistantAgent` is already fully
self-contained and runnable. See the updated "WHAT build() RETURNS" section
above for the precise decision.

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

VERIFIED UPSTREAM INCOMPATIBILITY: `anthropic>=1` breaks the `anthropic/...`
provider branch entirely, and commonadk cannot prevent it by omitting
`model_params` -- confirmed live (M8's first live run, all six targets,
model `anthropic/claude-haiku-4-5`): the AutoGen target was the only one of
six that did not execute, failing in 0.22s with `TypeError:
AsyncMessages.create() got an unexpected keyword argument 'temperature'`,
never reaching the API. Root cause, read directly from the installed
source, not inferred: `autogen_ext.models.anthropic._anthropic_client`
(still true as of autogen-ext 0.7.5, both `create` and `create_stream`)
builds every request as `request_args = {..., "temperature":
create_args.get("temperature", 1.0)}` -- this key is ALWAYS present, with a
default of `1.0`, regardless of whether the caller (this adapter, via
`model_params`) ever set `temperature`; there is no branch that omits it.
That was harmless against `anthropic<1`: this dev box's installed
`anthropic` 0.122.0 still accepts `temperature` as a real kwarg on
`AsyncMessages.create` (checked directly via `inspect.signature`), so every
offline test in this codebase passes here. But `anthropic>=1` (confirmed at
both the 1.0.0 boundary and 1.5.0 -- the version CI's `autogen` extra leg
actually resolves, since it is the only CI leg installing this extra with
no other extra around to pull in an older transitive `anthropic` pin)
removed `temperature` from `AsyncMessages.create`'s signature entirely
(confirmed via `inspect.signature` against a real `anthropic==1.5.0`
install), so that hard-coded kwarg becomes a raw `TypeError` raised from
inside `anthropic`'s own generated method wrapper, at Python
argument-binding time -- before any HTTP request, reproduced directly in a
clean venv with no network call and no real API key needed. This project's
`autogen` extra now pins `anthropic<1` for exactly this reason (see
pyproject.toml's `autogen` extra comment for what that pin does and does
not conflict with). That pin AVOIDS the bug for anyone who installs via
this project's own extras; it does not fix it -- `autogen_ext`'s client
still hard-codes the kwarg, so anyone who independently upgrades
`anthropic` past `1.0` in the same environment (or installs this adapter's
dependencies by hand, ignoring the extras) hits the same `TypeError` again.
So `_client_for`'s `anthropic/...` branch also runs a build-time guard,
`_check_anthropic_temperature_compat()`: it reads the installed
`anthropic` package's version via `importlib.metadata` (no import of
`anthropic` itself needed) and raises a clear commonadk `RuntimeError` --
naming the installed `anthropic` and `autogen-ext` versions, the exact
mechanism above, and the fix (`pip install 'anthropic<1'`) -- rather than
letting a raw SDK `TypeError` surface at `run()` time. This is the
project's standing policy (a target that cannot do something errors loudly
at build time, not obscurely at run time) applied to a case this adapter
cannot route around: unlike the model-info workarounds above, there is no
commonadk-side kwarg to add or omit that changes what `autogen_ext` sends.

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

import importlib.metadata as importlib_metadata
import warnings
from typing import TYPE_CHECKING, Any

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import Swarm
from autogen_agentchat.tools import AgentTool, TeamTool
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


def _installed_version(package: str) -> str | None:
    """`importlib.metadata.version`, `None` if `package` isn't installed --
    never imports `package` itself. Split out so tests can monkeypatch a
    single, narrow seam instead of the whole `importlib.metadata` module
    (see module docstring, "VERIFIED UPSTREAM INCOMPATIBILITY").
    """
    try:
        return importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        return None


def _check_anthropic_temperature_compat() -> None:
    """Build-time guard for the verified `anthropic>=1` incompatibility
    documented in the module docstring ("VERIFIED UPSTREAM
    INCOMPATIBILITY") -- called once, from `_client_for`'s `anthropic/...`
    branch, before constructing `AnthropicChatCompletionClient`. Raises a
    clear commonadk error instead of letting a raw `TypeError` surface out
    of `anthropic`'s own generated method wrapper the first time the built
    agent/team is actually run.
    """
    installed = _installed_version("anthropic")
    if installed is None:
        # No `anthropic` package at all -- `autogen_ext.models.anthropic`
        # itself fails to import with its own clear error first; not this
        # guard's job to duplicate that.
        return

    major = installed.split(".", 1)[0]
    try:
        major_num = int(major)
    except ValueError:
        # An unparseable leading version segment (unexpected for a real
        # PyPI release) -- don't block on a guess neither confirmed safe
        # nor confirmed broken.
        return
    if major_num < 1:
        return  # anthropic<1 -- the compatible range this adapter verified.

    autogen_ext_version = _installed_version("autogen-ext") or "<not installed>"
    raise RuntimeError(
        f"commonadk: the AutoGen target ('autogen') cannot use the "
        f"anthropic/... provider with anthropic=={installed} installed. "
        f"autogen-ext=={autogen_ext_version}'s AnthropicChatCompletionClient "
        f"unconditionally sends a `temperature` kwarg to "
        f"anthropic.AsyncMessages.create() on every request -- "
        f"'\"temperature\": create_args.get(\"temperature\", 1.0)' in "
        f"autogen_ext/models/anthropic/_anthropic_client.py, present even "
        f"when no model_params.temperature was ever set -- but anthropic>=1 "
        f"removed the `temperature` parameter from `messages.create()` "
        f"entirely, so every call would fail with `TypeError: "
        f"AsyncMessages.create() got an unexpected keyword argument "
        f"'temperature'` (verified directly, both at construction-adjacent "
        f"inspection and a live reproduction -- see autogen_adapter.py's "
        f"module docstring). This is an upstream autogen-ext bug, not "
        f"something commonadk's model_params mapping can route around. "
        f"Fix: `pip install 'anthropic<1'` in this environment (this "
        f"project's own `autogen` extra already pins this; something else "
        f"upgraded it past that pin here), or track "
        f"https://github.com/microsoft/autogen for an upstream fix."
    )


class AutoGenAdapter(BaseAdapter):
    target = "autogen"

    def build(
        self,
        project: "Project",
        agent_name: str,
        _delegate_ancestors: tuple[str, ...] = (),
    ) -> Any:
        """Build `agent_name`. `_delegate_ancestors` is an internal-only
        parameter (not part of `BaseAdapter`'s public contract) used when
        this method recurses into a `delegate` destination's own build --
        see module docstring, "Recursion and cycles".
        """
        if agent_name in _delegate_ancestors:
            chain = " -> ".join((*_delegate_ancestors, agent_name))
            raise ValueError(
                f"commonadk: cycle detected in delegate edges reachable "
                f"from the build root ({chain}). Each `delegate` edge "
                f"recurses into an independent build of its destination "
                f"(see autogen_adapter.py's module docstring, 'Recursion "
                f"and cycles'), which would recurse forever around this "
                f"cycle rather than terminating."
            )
        self._check_env(project, agent_name)

        # Only handoff-reachable agents join this build's Swarm/participant
        # set -- a `delegate`-only destination is invoked as a standalone
        # tool call (AgentTool/TeamTool, below) and never receives
        # conversation history or becomes a Swarm speaker (see module
        # docstring, "WHAT build() RETURNS").
        handoff_reachable = self._reachable_via(project, agent_name, {"handoff"})  # root first

        agents: dict[str, AssistantAgent] = {
            name: self._build_assistant_agent(project, name, _delegate_ancestors)
            for name in handoff_reachable
        }

        has_outgoing_handoff = any(
            edge.from_ == agent_name and edge.type == "handoff"
            for edge in project.graph.edges
        )
        if not has_outgoing_handoff:
            # Nothing for the build root to hand off to -- handoffs only do
            # anything inside a team, so the bare agent (its own delegate
            # tools already wired in by _build_assistant_agent) is the
            # honest, directly runnable object here.
            return agents[agent_name]

        participants = [agents[name] for name in handoff_reachable]  # root first
        return Swarm(participants, max_turns=len(handoff_reachable))

    # -- per-agent construction -------------------------------------------

    def _build_assistant_agent(
        self,
        project: "Project",
        name: str,
        ancestors: tuple[str, ...],
    ) -> AssistantAgent:
        """Build one `AssistantAgent`, its `handoffs` list set from `name`'s
        `handoff` edges and its own `delegate` edges wired in as
        AgentTool/TeamTool-wrapped tools (see module docstring, "Edge
        mapping" and "Recursive construction").
        """
        spec = project.agents[name]
        handoff_targets = [
            edge.to
            for edge in project.graph.edges
            if edge.from_ == name and edge.type == "handoff"
        ]
        agent = AssistantAgent(
            name=spec.name,
            model_client=self._client_for(project, spec),
            tools=[t.func for t in spec.tools],
            handoffs=handoff_targets,
            system_message=spec.instructions,
            description=spec.config.description,
        )

        delegate_edges = [
            edge for edge in project.graph.edges if edge.from_ == name and edge.type == "delegate"
        ]
        if delegate_edges:
            child_ancestors = (*ancestors, name)
            delegate_tools = [
                self._make_delegate_tool(project, edge.to, child_ancestors)
                for edge in delegate_edges
            ]
            # `AssistantAgent` exposes no public `tools` attribute/setter at
            # all (verified via `inspect.getsource`/`dir`: only the private
            # `self._tools: List[BaseTool[Any, Any]]`, appended to at
            # construction and read directly wherever the agent builds its
            # LLM tool schema) -- so appending post-construction, the same
            # way `AssistantAgent.__init__` itself populates it, is the only
            # way to add a tool after the fact; there is no cleaner public
            # seam to prefer here.
            agent._tools.extend(delegate_tools)
        return agent

    def _make_delegate_tool(
        self, project: "Project", dest_name: str, ancestors: tuple[str, ...]
    ) -> Any:
        """Build `dest_name` via a fresh, independent `build()` call and
        wrap the result as a tool -- `AgentTool` for a bare `AssistantAgent`,
        `TeamTool` for a `Swarm` (see module docstring, "Recursive
        construction, and the AgentTool/TeamTool split").
        """
        built = self.build(project, dest_name, _delegate_ancestors=ancestors)
        if isinstance(built, Swarm):
            return TeamTool(
                team=built,
                name=f"delegate_to_{dest_name}",
                description=(
                    f"Delegate a task to the '{dest_name}' team and "
                    f"receive its result back into this conversation."
                ),
            )
        return AgentTool(agent=built)

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
            _check_anthropic_temperature_compat()
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
