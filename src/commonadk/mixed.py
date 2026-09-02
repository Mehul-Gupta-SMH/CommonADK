"""Mixed-target spawning: several agents, each on its own SDK, one process.

See `docs/mixed-target-design.md` for the full design -- this module is its
implementation. Three layers (design doc, "The three-layer model"):

1. **Per-agent build** -- unchanged, the existing six adapters in
   `adapters/*.py`.
2. **Per-runtime unit ("island")** -- a maximal set of agents that share an
   effective runtime and are connected to each other by `interactions.yaml`
   edges between agents of that same runtime. Each island is built by its
   own adapter as a normal sub-graph (`_compute_islands`, `build_mixed`).
3. **Coordinator** -- handles the edges layer 2 filtered out: edges whose
   two endpoints resolve to different runtimes. Each becomes a plain
   Python callable "transfer to `<destination>`" tool on the edge's source
   agent (`_make_transfer_func`, `_ATTACH`), wired to call the destination
   island's own native run mechanism (`_INVOKERS`, mirroring `cli.py`'s
   `_run_*` functions).

This module imports no agent SDK at module scope -- exactly like
`adapters/__init__.py` and `adapters/base.py`, every SDK import here is
local to the function that actually needs it, so `commonadk.load()` and
`Project.build_mixed` for a project that sets no agent's `runtime:` (or
only sets `runtime:` values whose SDKs happen to be installed) never pay
for an SDK this particular build doesn't touch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .models import InteractionGraph, Project


@dataclass
class RuntimeUnit:
    """One island: a native sub-graph built by one target's adapter."""

    runtime: str
    root: str
    members: list[str]
    native: Any


@dataclass
class MixedSystem:
    """The result of `Project.build_mixed(...)`: every island plus the
    cross-runtime bridges wired between them.

    `units` and `unit_of` together answer "what runtime is agent X on, and
    what's the native object for its island" -- `unit_for(name)` is the
    convenience form of that lookup. `entry_native` is the one object a
    caller most often wants: the native root object of the island
    containing the overall build's entry agent.
    """

    project: Project
    entry: str
    default_target: str
    agent_runtime: dict[str, str]
    units: dict[str, RuntimeUnit]  # keyed by island root agent name
    unit_of: dict[str, str]  # agent name -> its island's root agent name
    cross_edges: list[tuple[str, str]] = field(default_factory=list)

    def unit_for(self, agent_name: str) -> RuntimeUnit:
        """The `RuntimeUnit` (island) containing `agent_name`."""
        return self.units[self.unit_of[agent_name]]

    @property
    def entry_native(self) -> Any:
        """The native object for the island containing the entry agent."""
        return self.unit_for(self.entry).native


# ---------------------------------------------------------------------------
# build_mixed
# ---------------------------------------------------------------------------


def build_mixed(project: Project, agent_name: str, default_target: str) -> MixedSystem:
    """Build every runtime unit reachable from `agent_name` and wire the
    cross-runtime edges between them. See module docstring and
    docs/mixed-target-design.md.
    """
    from .adapters import get_adapter

    project._require_agent(agent_name)  # same KeyError style as Project.build

    agent_runtime = {
        name: project.effective_runtime(name, default_target) for name in project.agents
    }

    reachable = _reachable_all_edges(project, agent_name)

    _check_env_all(project, reachable, agent_name)

    islands = _compute_islands(project, reachable, agent_runtime, agent_name)

    units: dict[str, RuntimeUnit] = {}
    unit_of: dict[str, str] = {}
    for root, members in islands:
        runtime = agent_runtime[root]
        sub_project = _project_view(project, members)
        native = get_adapter(runtime).build(sub_project, root)
        units[root] = RuntimeUnit(
            runtime=runtime, root=root, members=sorted(members), native=native
        )
        for member in members:
            unit_of[member] = root

    cross_edges = [
        (edge.from_, edge.to)
        for edge in project.graph.edges
        if edge.from_ in reachable
        and edge.to in reachable
        and agent_runtime[edge.from_] != agent_runtime[edge.to]
    ]

    for src, dst in cross_edges:
        src_root = unit_of[src]
        if src != src_root:
            raise ValueError(
                f"commonadk: cross-runtime edge '{src}' -> '{dst}' "
                f"originates at '{src}', which is not the root of its "
                f"runtime unit ('{src_root}', runtime "
                f"{agent_runtime[src]!r}). v1 only supports a cross-runtime "
                f"edge sourced at a unit's root agent -- see "
                f"docs/mixed-target-design.md, 'Cross-runtime edges'."
            )

    for src, dst in cross_edges:
        src_unit = units[unit_of[src]]
        dst_unit = units[unit_of[dst]]
        _attach_bridge(project, src_unit, dst_unit, dst)

    return MixedSystem(
        project=project,
        entry=agent_name,
        default_target=default_target,
        agent_runtime=agent_runtime,
        units=units,
        unit_of=unit_of,
        cross_edges=cross_edges,
    )


# ---------------------------------------------------------------------------
# reachability / env preflight
# ---------------------------------------------------------------------------


def _reachable_all_edges(project: Project, start: str) -> set[str]:
    """Every agent reachable from `start` over every edge (intra- and
    cross-runtime alike) -- the cross-runtime analogue of
    `BaseAdapter._reachable_agents`, which only ever sees one target's
    filtered edge set.
    """
    seen = {start}
    order = [start]
    i = 0
    while i < len(order):
        current = order[i]
        i += 1
        for edge in project.graph.edges:
            if edge.from_ == current and edge.to not in seen:
                seen.add(edge.to)
                order.append(edge.to)
    return seen


def _check_env_all(project: Project, reachable: set[str], agent_name: str) -> None:
    """Fail loudly, up front, for every reachable agent across every
    runtime -- before any island is built. Reuses `Project.check_env`, the
    same primitive `BaseAdapter._check_env` (adapters/base.py) calls per
    agent for a single-target build.
    """
    missing_lines: list[str] = []
    for name in sorted(reachable):
        agent = project.agents[name]
        missing_names = set(project.check_env(name))
        if not missing_names:
            continue
        for req in agent.config.requires.env:
            if req.name in missing_names:
                detail = f"{req.name} ({req.description})" if req.description else req.name
                missing_lines.append(
                    f"  - {name} [{agent.config.runtime or '<default>'}]: {detail}"
                )

    if missing_lines:
        raise OSError(
            "commonadk: missing required environment variable(s) across the "
            f"mixed-target build (building {agent_name!r}):\n"
            + "\n".join(missing_lines)
        )


# ---------------------------------------------------------------------------
# island computation
# ---------------------------------------------------------------------------


def _compute_islands(
    project: Project,
    reachable: set[str],
    agent_runtime: dict[str, str],
    entry: str,
) -> list[tuple[str, set[str]]]:
    """Group `reachable` agents into islands (design doc, "Island
    computation") and pick each island's root. Returns `(root, members)`
    pairs.
    """
    intra_edges = [
        edge
        for edge in project.graph.edges
        if edge.from_ in reachable
        and edge.to in reachable
        and agent_runtime[edge.from_] == agent_runtime[edge.to]
    ]

    parent = {name: name for name in reachable}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for edge in intra_edges:
        union(edge.from_, edge.to)

    groups: dict[str, set[str]] = {}
    for name in reachable:
        groups.setdefault(find(name), set()).add(name)

    islands: list[tuple[str, set[str]]] = []
    for members in groups.values():
        preferred = entry if entry in members else None
        root = _pick_root(members, intra_edges, preferred)
        islands.append((root, members))
    return islands


def _pick_root(
    members: set[str], intra_edges: list, preferred: Optional[str]
) -> str:
    """The one agent in `members` that reaches every other member over
    directed intra-island edges -- the root `mixed.py` hands to the
    island's adapter (adapters only walk forward from one root). Prefers
    `preferred` (the overall build entry) when it qualifies; otherwise the
    lexicographically-first qualifying agent, for a deterministic pick.
    """
    adjacency: dict[str, list[str]] = {member: [] for member in members}
    for edge in intra_edges:
        if edge.from_ in members:
            adjacency[edge.from_].append(edge.to)

    def reaches_everyone(start: str) -> bool:
        seen = {start}
        stack = [start]
        while stack:
            current = stack.pop()
            for nxt in adjacency[current]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen == members

    candidates = sorted(member for member in members if reaches_everyone(member))
    if not candidates:
        raise ValueError(
            f"commonadk: runtime unit {sorted(members)} has no single agent "
            f"that reaches every other member of the unit via its own "
            f"interactions.yaml edges -- v1 requires each runtime unit to "
            f"be buildable from one root (see docs/mixed-target-design.md, "
            f"'Island computation'). Restructure the edges so one agent in "
            f"this group reaches the rest, or give one of these agents its "
            f"own `runtime:` so it becomes its own unit."
        )
    if preferred in candidates:
        return preferred
    return candidates[0]


def _project_view(project: Project, members: set[str]) -> Project:
    """A shallow copy of `project` whose `graph.edges` are restricted to
    edges wholly inside `members` -- what an island's own adapter is
    handed, so it builds a normal sub-graph exactly like a single-target
    `build()` (design doc, "Per-runtime unit").
    """
    filtered_edges = [
        edge for edge in project.graph.edges if edge.from_ in members and edge.to in members
    ]
    return project.model_copy(
        update={"graph": InteractionGraph(entry=project.graph.entry, edges=filtered_edges)}
    )


# ---------------------------------------------------------------------------
# cross-runtime bridge: attach (source side) + invoke (destination side)
# ---------------------------------------------------------------------------


def _make_transfer_func(dst_name: str, invoke: Callable[[str], str]) -> Callable[[str], str]:
    """A plain, typed, documented Python function -- the same shape every
    `tools.py` function already has to be -- that becomes the cross-runtime
    bridge tool. Every source-capable adapter's own tool-wrapping mechanism
    turns this into a native tool exactly like it would any other function.
    """

    def transfer(message: str) -> str:
        """Transfer the conversation to another agent running on a
        different runtime, and return its response.

        Args:
            message: The full context/request the other agent needs.

        Returns:
            The other agent's response as text.
        """
        return invoke(message)

    transfer.__name__ = f"transfer_to_{dst_name}"
    transfer.__qualname__ = transfer.__name__
    transfer.__doc__ = (
        f"Transfer the conversation to the '{dst_name}' agent, which runs "
        f"on a different runtime. Send it the full context/request it "
        f"needs; its response is returned as text.\n\n"
        f"Args:\n    message: What to tell '{dst_name}'.\n\n"
        f"Returns:\n    '{dst_name}''s response as text."
    )
    return transfer


def _attach_bridge(
    project: Project, src_unit: RuntimeUnit, dst_unit: RuntimeUnit, dst_name: str
) -> None:
    attach = _ATTACH.get(src_unit.runtime)
    if attach is None:
        raise ValueError(_unsupported_source_error(src_unit, dst_name))
    invoke = _make_invoker(dst_unit, project)
    attach(src_unit, dst_name, invoke)


def _make_invoker(dst_unit: RuntimeUnit, project: Project) -> Callable[[str], str]:
    invoker = _INVOKERS[dst_unit.runtime]
    native = dst_unit.native
    app_name = project.config.name

    def invoke(message: str) -> str:
        return invoker(native, message, app_name)

    return invoke


# -- source side: attach a "transfer to <destination>" tool to an island root

_UNSUPPORTED_SOURCE_REASON = {
    "autogen": (
        "AssistantAgent only accepts tools at construction time "
        "(autogen_agentchat.agents.AssistantAgent.__init__); its wrapped "
        "tool list is stored on a private, undocumented `_tools` "
        "attribute with no supported public way to add a tool after "
        "construction (verified against the installed autogen-agentchat "
        "package -- see docs/mixed-target-design.md, 'Supported/"
        "unsupported cross-runtime source targets'). commonadk will not "
        "build a documented feature on an SDK's private internals."
    ),
    "langgraph": (
        "LangGraphAdapter.build() compiles the whole StateGraph in one "
        "call (builder.compile()) and returns the resulting "
        "CompiledStateGraph; there is no supported hook to add a tool to "
        "an already-compiled node afterward, and reusing the adapter "
        "unchanged means there is no earlier point to inject one before "
        "compile (verified against the installed langgraph package -- see "
        "docs/mixed-target-design.md, 'Supported/unsupported cross-"
        "runtime source targets')."
    ),
}


def _unsupported_source_error(src_unit: RuntimeUnit, dst_name: str) -> str:
    reason = _UNSUPPORTED_SOURCE_REASON.get(
        src_unit.runtime,
        "this target has no supported way to attach a tool to an "
        "already-built agent in v1.",
    )
    return (
        f"commonadk: agent '{src_unit.root}' (runtime "
        f"{src_unit.runtime!r}) has a cross-runtime edge to '{dst_name}', "
        f"but runtime {src_unit.runtime!r} is not one of the cross-runtime "
        f"source-capable targets in v1 ({sorted(_ATTACH)}). {reason}"
    )


def _attach_google_adk(src_unit: RuntimeUnit, dst_name: str, invoke: Callable[[str], str]) -> None:
    func = _make_transfer_func(dst_name, invoke)
    src_unit.native.tools.append(func)


def _attach_openai(src_unit: RuntimeUnit, dst_name: str, invoke: Callable[[str], str]) -> None:
    from agents import function_tool

    func = _make_transfer_func(dst_name, invoke)
    src_unit.native.tools.append(function_tool(func))


def _attach_crewai(src_unit: RuntimeUnit, dst_name: str, invoke: Callable[[str], str]) -> None:
    from crewai import Process
    from crewai.tools import tool as crewai_tool

    crew = src_unit.native
    is_manager = crew.process == Process.hierarchical and crew.manager_agent is not None
    if is_manager:
        raise ValueError(
            f"commonadk: agent '{src_unit.root}' (runtime 'crewai') has a "
            f"cross-runtime edge to '{dst_name}', but '{src_unit.root}' is "
            f"built as this runtime unit's CrewAI hierarchical manager (it "
            f"has intra-unit delegate(s): {sorted(m for m in src_unit.members if m != src_unit.root)}), "
            f"and CrewAI managers cannot hold tools (crewai_adapter.py, "
            f"'Manager tools'). Restructure the graph so the cross-runtime "
            f"edge originates from a non-manager crewai agent, or give "
            f"'{src_unit.root}' no intra-runtime delegates."
        )
    root_agent = crew.agents[0]
    func = _make_transfer_func(dst_name, invoke)
    root_agent.tools.append(crewai_tool(func))


def _attach_claude(src_unit: RuntimeUnit, dst_name: str, invoke: Callable[[str], str]) -> None:
    from claude_agent_sdk import create_sdk_mcp_server
    from claude_agent_sdk import tool as sdk_tool

    options = src_unit.native
    server_key = f"_bridge_{src_unit.root}_to_{dst_name}"
    tool_name = f"transfer_to_{dst_name}"
    description = (
        f"Transfer the conversation to the '{dst_name}' agent, which runs "
        f"on a different runtime. Send it the full context/request it "
        f"needs; its response is returned as text."
    )
    input_schema = {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
    }

    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        result = invoke(args.get("message", ""))
        return {"content": [{"type": "text", "text": result}]}

    wrapped = sdk_tool(tool_name, description, input_schema)(handler)
    options.mcp_servers[server_key] = create_sdk_mcp_server(
        name=server_key, tools=[wrapped]
    )
    options.allowed_tools.append(f"mcp__{server_key}__{tool_name}")


_ATTACH: dict[str, Callable[[RuntimeUnit, str, Callable[[str], str]], None]] = {
    "google-adk": _attach_google_adk,
    "openai": _attach_openai,
    "crewai": _attach_crewai,
    "claude": _attach_claude,
}


# -- destination side: invoke an already-built island with a text prompt

def _invoke_google_adk(native: Any, prompt: str, app_name: str) -> str:
    import asyncio

    from google.adk.runners import InMemoryRunner
    from google.genai import types as genai_types

    runner = InMemoryRunner(agent=native, app_name=app_name)
    user_id = "commonadk-mixed"

    async def _run() -> str:
        session = await runner.session_service.create_session(
            app_name=runner.app_name, user_id=user_id
        )
        message = genai_types.Content(role="user", parts=[genai_types.Part(text=prompt)])
        chunks: list[str] = []
        async for event in runner.run_async(
            user_id=user_id, session_id=session.id, new_message=message
        ):
            if event.is_final_response() and event.content and event.content.parts:
                chunks.extend(
                    part.text for part in event.content.parts if getattr(part, "text", None)
                )
        return "\n".join(chunks)

    return asyncio.run(_run())


def _invoke_openai(native: Any, prompt: str, app_name: str) -> str:
    from agents import Runner

    result = Runner.run_sync(native, prompt)
    return str(result.final_output)


def _invoke_claude(native: Any, prompt: str, app_name: str) -> str:
    import asyncio

    from claude_agent_sdk import ResultMessage, query

    async def _run() -> str:
        chunks: list[str] = []
        async for message in query(prompt=prompt, options=native):
            if isinstance(message, ResultMessage) and message.result:
                chunks.append(message.result)
        return "\n".join(chunks)

    return asyncio.run(_run())


def _invoke_crewai(native: Any, prompt: str, app_name: str) -> str:
    from crewai import Process, Task

    agent = None if native.process == Process.hierarchical else native.agents[0]
    native.tasks = [
        Task(
            description=prompt,
            expected_output="A complete response to the request above.",
            agent=agent,
        )
    ]
    result = native.kickoff()
    return str(result.raw)


def _invoke_autogen(native: Any, prompt: str, app_name: str) -> str:
    import asyncio

    result = asyncio.run(native.run(task=prompt))
    return str(result.messages[-1].content)


def _invoke_langgraph(native: Any, prompt: str, app_name: str) -> str:
    result = native.invoke({"messages": [{"role": "user", "content": prompt}]})
    return str(result["messages"][-1].content)


_INVOKERS: dict[str, Callable[[Any, str, str], str]] = {
    "google-adk": _invoke_google_adk,
    "openai": _invoke_openai,
    "claude": _invoke_claude,
    "crewai": _invoke_crewai,
    "autogen": _invoke_autogen,
    "langgraph": _invoke_langgraph,
}
