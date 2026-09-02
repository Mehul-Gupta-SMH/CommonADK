"""CrewAI adapter: `AgentSpec` -> a live `crewai.Crew`.

WHAT `build()` RETURNS -- read this first: unlike the Claude Agent SDK
adapter (session/query-based, no persistent object) and like the Google ADK
and OpenAI Agents adapters, CrewAI has real persistent agent objects. But
`crewai.Crew` requires a `tasks` list at construction (`tasks: list[Task]` is
a required field with no default) -- investigated directly against the
installed SDK: `Crew(agents=[...], tasks=[], process=...)` DOES construct
successfully with an *empty* `tasks` list (verified: a `Task` needs a
`description`/`expected_output` that only exist once the caller knows the
actual prompt, which `build()` never sees -- that's `commonadk run`'s job,
not the adapter's). So this adapter always returns a fully-wired
`crewai.Crew` with `tasks=[]`: every reachable agent is built and attached,
and the caller supplies the one task for the turn at hand at kickoff time.
Usage:

    project = commonadk.load("common/")
    crew = project.build("coordinator", target="crewai")

    from crewai import Task
    task = Task(
        description="Research EV adoption",
        expected_output="A short written summary",
        agent=crew.agents[0] if crew.process.value == "sequential" else None,
    )
    crew.tasks = [task]
    result = crew.kickoff()
    print(result.raw)

(see cli.py's `_run_crewai` for the same pattern, wired to argv.)

Verified against the installed package: crewai 1.15.16 (`crewai.Agent`,
`crewai.Crew`, `crewai.Task`, `crewai.Process`, `crewai.LLM`, and
`crewai.tools.tool`, introspected via `pydantic`'s `model_fields` and
`inspect.getsource` during M6 -- not taken from memory of the API).

Manager-or-solo-member decision -- investigated, not assumed: `crewai.Crew`
supports two processes, `Process.sequential` and `Process.hierarchical`
(`crewai.Process` is a two-value enum; nothing else exists). Hierarchical
needs a `manager_agent` that is explicitly NOT a member of `agents`
(`Crew.__init__` raises a pydantic `ValidationError`, "Manager agent should
not be included in agents list", if it is -- verified). This adapter builds
the requested `agent_name` as manager whenever it has at least one other
agent reachable from it (`process=Process.hierarchical`, `manager_agent=`
the root, `agents=` every *other* reachable agent). When the build root has
no reachable agents at all (a leaf built directly, e.g. `writer` in the
shipped example), there is no one to manage, so this adapter falls back to
`process=Process.sequential` with the root as the crew's one and only
member -- verified both shapes construct successfully (an *empty*
non-manager `agents=[]` combined with an empty `tasks=[]` is rejected by a
different pydantic validator, "Either 'agents' and 'tasks' need to be set",
so the empty-crew shape is not viable and the sequential/solo-member
fallback is the correct honest choice, not merely a convenient one).

Manager tools -- investigated, not assumed, and a real behavioral
constraint: `Crew._create_manager_agent` (called at `kickoff()`, not at
`build()`) does `if manager.tools is not None and len(manager.tools) > 0:
... raise Exception("Manager agent should not have tools")` -- verified by
calling it directly against a manager built with tools attached. So when
this adapter puts `agent_name` in the manager role, it builds that agent
with `tools=[]` regardless of what `agent-config.yaml` declares, and warns
(same warn-and-ignore policy as unsupported `model_params` keys below) if
that agent actually has declared tools -- they are silently unusable in the
manager role otherwise, and kickoff would hard-crash if this adapter passed
them through. This only affects the root when it becomes a manager (i.e.
when it has outgoing edges); every other agent in the crew, and the root
itself when it ends up as the lone sequential member, keeps its own tools.

Edge mapping -- investigated, not assumed, and COARSENED versus every other
adapter in this codebase: both `delegate` and `handoff` edges map to CrewAI
delegation (there is exactly one delegation mechanism in this SDK -- an
agent with `allow_delegation=True` gets delegation tools automatically,
whether the crew runs hierarchical or sequential; see
`Crew._prepare_tools`/`Crew._add_delegation_tools`). CRITICALLY, CrewAI
delegation is CREW-WIDE, not per-edge: `_add_delegation_tools` builds its
target list as `[agent for agent in self.agents if agent != task.agent]` --
literally every OTHER agent in the crew, with no concept of "only the agents
`interactions.yaml` actually points this agent at." So `interactions.yaml`'s
edge *targets* are NOT representable here -- an agent that can delegate can
delegate to any crew member, not just its declared out-edges. What this
adapter DOES still honor from the graph: (1) *whether* an agent can delegate
at all -- `allow_delegation=True` only for agents with at least one outgoing
edge in `interactions.yaml`, `False` for agents with none; and (2) *scope*
-- only agents actually reachable from the build root join the crew at all
(via `BaseAdapter._reachable_agents`), so an unrelated part of a larger
`interactions.yaml` graph never becomes a delegation target just because it
happens to exist. This is a real, documented semantic loss versus Google
ADK's sub_agents tree, OpenAI Agents' per-agent `handoffs` list, and the
Claude Agent SDK's per-agent tool allowlist -- all three of which *can*
represent "agent A may only reach agent B, not C." Multi-parent graphs and
cycles need no special handling as a result: `crew.agents` is a flat list
built once per logical agent name from `_reachable_agents`'s already-deduped
result, so an agent reachable by two paths, or a path that cycles back to
the build root, is simply the same `Agent` instance appearing once (a cycle
back to the root is a no-op, same as the other two flat-registry adapters --
the root is never added to its own `agents`/`manager_agent` twice).

AgentSpec mapping: `role=spec.name`, `goal=spec.config.description`,
`backstory=spec.instructions` (the agent's `skill.md` content). This mirrors
how the other adapters use these three fields (Google ADK: `name`/
`description`/`instruction`; OpenAI Agents: `name`/`handoff_description`/
`instructions`) -- `role` is CrewAI's only agent-identifying string field,
`goal` is its short one-line purpose (matching `description`'s role
elsewhere), and `backstory` is where CrewAI expects an agent's fuller
persona/task instructions to live, matching `instructions`' role elsewhere.

Model routing: CrewAI's `LLM` class speaks LiteLLM-format `"provider/model"`
strings NATIVELY -- `crewai.LLM.__new__` (a factory, not a plain
constructor; verified via `inspect.getsource`) parses the `"provider/..."`
prefix itself, routes a short list of major providers (openai, anthropic,
azure, bedrock, gemini, openrouter, deepseek, ollama, cerebras, ...) to a
native provider client, and FALLS BACK to litellm's `completion()` for any
other provider string -- either way the exact `"provider/model"` string
`project.resolve_model(agent)` produces is exactly what `LLM(model=...)`
expects, unchanged. This is the point of this target: there is NO
unsupported-provider error here, unlike the Google ADK, OpenAI Agents, and
Claude Agent SDK adapters, each of which special-cases one native provider
and only reaches LiteLLM (or raises) for everything else. A per-target
`targets.crewai.model` override in `agent-config.yaml` always wins and is
passed through as-is to `LLM(model=...)`, exactly like every other adapter's
override handling.

model_params: `crewai.LLM` (and every native provider subclass it can
resolve to -- verified against `GeminiCompletion`, `AnthropicCompletion`,
`OpenAICompletion`, AND the base `crewai.llm.LLM` class used for the litellm
fallback path, via each class's `model_fields`) exposes `temperature`,
`max_tokens`, `top_p`, `stop`, `presence_penalty`, `frequency_penalty`, and
`seed` as real pydantic fields on EVERY one of those four classes -- so all
seven map directly (`_MODEL_PARAM_MAP`), each verified by actually
constructing an `LLM(...)` with all seven kwargs set and reading the
resulting instance's attributes back (not just checking `model_fields`
exists). `top_k` is deliberately NOT mapped: only `GeminiCompletion` among
the four classes declares it (`OpenAICompletion`, `AnthropicCompletion`, and
the base `LLM` class do not), so it isn't safe to route uniformly for a
`model_params` key that can resolve to any of CrewAI's supported providers --
it falls through to the warn-and-ignore path below, same as any other
adapter's genuinely-unsupported keys.

Tool wiring: each `ToolSpec`'s plain, typed, documented `tools.py` function
is wrapped with `crewai.tools.tool(func)` -- the SDK's own function-tool
decorator, which builds a pydantic `args_schema` from the function's
signature and requires a docstring and type hints (both already enforced
upstream by `validation.py` before an `AgentSpec` can exist, so this never
fails here). `tool_spec.name` is always `func.__name__` (see
`ToolSpec.from_function` in models.py), which is exactly what the decorator
derives the wrapped tool's name from, so no separate name/description
plumbing is needed.

Telemetry / offline construction: CrewAI ships anonymous, opt-out telemetry
(`crewai.telemetry.Telemetry`) that can spin up an OpenTelemetry exporter on
first use. It is gated at *every* emission call by
`Telemetry._should_execute_telemetry()`, which re-checks
`CREWAI_DISABLE_TELEMETRY` / `OTEL_SDK_DISABLED` / `CREWAI_DISABLE_TRACKING`
dynamically (not just once at import) -- verified by reading
`telemetry.py`. Building a `Crew` (this adapter's `build()`) does not itself
emit any telemetry event (`kickoff()` does); the test suite still sets
`CREWAI_DISABLE_TELEMETRY=1` and `OTEL_SDK_DISABLED=1` (see
`tests/test_adapter_crewai.py`) so nothing in this codebase's test run ever
depends on that being true only by construction-time accident.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from crewai import Agent, Crew, Process, LLM
from crewai.tools import tool as crewai_tool

if TYPE_CHECKING:
    from ..models import AgentSpec, Project, ToolSpec

from .base import BaseAdapter

# agent-config.yaml `model_params` key -> crewai.LLM field. See module
# docstring, "model_params" -- verified across GeminiCompletion,
# AnthropicCompletion, OpenAICompletion, and the base litellm-fallback LLM
# class. `top_k` deliberately excluded (Gemini-only, not universal).
_MODEL_PARAM_MAP = {
    "temperature": "temperature",
    "max_tokens": "max_tokens",
    "top_p": "top_p",
    "stop": "stop",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
    "seed": "seed",
}


class CrewAIAdapter(BaseAdapter):
    target = "crewai"

    def build(self, project: "Project", agent_name: str) -> Crew:
        self._check_env(project, agent_name)

        reachable = self._reachable_agents(project, agent_name)  # includes agent_name
        has_outgoing = {
            name: any(edge.from_ == name for edge in project.graph.edges)
            for name in reachable
        }
        member_names = [name for name in reachable if name != agent_name]
        is_manager = bool(member_names)  # root has >=1 other reachable agent

        root_agent = self._build_agent(
            project, agent_name, allow_delegation=has_outgoing[agent_name], as_manager=is_manager
        )

        if not is_manager:
            # No one for the build root to manage/delegate to -- fall back
            # to a solo-member sequential crew (see module docstring,
            # "Manager-or-solo-member decision").
            return Crew(agents=[root_agent], tasks=[], process=Process.sequential)

        members = [
            self._build_agent(
                project, name, allow_delegation=has_outgoing[name], as_manager=False
            )
            for name in member_names
        ]
        return Crew(
            agents=members,
            tasks=[],
            process=Process.hierarchical,
            manager_agent=root_agent,
        )

    # -- agent construction ---------------------------------------------

    def _build_agent(
        self, project: "Project", name: str, allow_delegation: bool, as_manager: bool
    ) -> Agent:
        spec = project.agents[name]

        if as_manager and spec.tools:
            warnings.warn(
                f"{name}: CrewAI's hierarchical manager role cannot hold "
                f"tools (kickoff() raises if it does) -- {name}'s "
                f"{len(spec.tools)} declared tool(s) are dropped for this "
                f"build since it has outgoing edges and is being built as "
                f"the crew's manager. See crewai_adapter.py's module "
                f"docstring, 'Manager tools'.",
                stacklevel=2,
            )
        tools = [] if as_manager else [self._crewai_tool_for(t) for t in spec.tools]

        return Agent(
            role=spec.name,
            goal=spec.config.description,
            backstory=spec.instructions,
            tools=tools,
            llm=self._llm_for(project, spec),
            allow_delegation=allow_delegation,
        )

    # -- tool wiring (SDK's own function-tool decorator) -----------------

    def _crewai_tool_for(self, tool_spec: "ToolSpec") -> Any:
        return crewai_tool(tool_spec.func)

    # -- model routing ------------------------------------------------------

    def _llm_for(self, project: "Project", spec: "AgentSpec") -> LLM:
        override = spec.config.targets.get("crewai", {})
        if "model" in override:
            # Per-target override: already SDK-native form, passed through as-is.
            model = override["model"]
        else:
            model = project.resolve_model(spec.name)  # LiteLLM-format string

        return LLM(model=model, **self._model_param_kwargs(spec))

    def _model_param_kwargs(self, spec: "AgentSpec") -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        for key, value in spec.config.model_params.items():
            mapped = _MODEL_PARAM_MAP.get(key)
            if mapped is None:
                warnings.warn(
                    f"{spec.name}: model_params key '{key}' is not supported "
                    f"by the CrewAI adapter and will be ignored",
                    stacklevel=2,
                )
                continue
            kwargs[mapped] = value
        return kwargs
