"""commonadk's command-line interface.

Three subcommands, mirroring plan.md ("M4 -- CLI & docs"):

- `commonadk validate <common-dir>` -- load + validate a `common/` project and
  print a human-readable summary (or the error list, on failure).
- `commonadk render <common-dir>` -- regenerate `interaction-layer.md` from
  `interactions.yaml`.
- `commonadk run <common-dir> --target
  {google-adk,openai,claude,crewai,autogen,langgraph} PROMPT` -- build an
  agent for a target SDK and execute one turn.

`validate` and `render` never need an agent SDK installed -- they only touch
`loader.py`/`mermaid.py`, which have no SDK imports at module scope. `run`
does need one, but only for the target actually requested, so every
SDK-touching import in this module lives inside the function that uses it
(see `_run_google_adk` / `_run_openai` / `_run_claude` / `_run_crewai` /
`_run_autogen` / `_run_langgraph`), never at module scope. `_run_claude` additionally
preflights `ANTHROPIC_API_KEY` itself --
the Claude Agent SDK's bundled CLI needs it to authenticate, but (unlike
`requires.env` in `agent-config.yaml`) nothing in the SDK declares or checks
for it up front.

Every command funnels its expected failure modes -- `ValidationError` (bad
project), `OSError` (missing required env var, from the adapters' env
preflight), `ValueError` (unknown build target, unbuildable graph), and
`ImportError` (target SDK not installed) -- through `main`'s top-level
`try/except`, so the CLI always prints one clean message and a non-zero exit
code instead of a Python traceback.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import TYPE_CHECKING, Optional, Sequence

from .loader import load
from .mermaid import write_interaction_layer
from .validation import ValidationError

if TYPE_CHECKING:
    from .models import Project


def _version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("commonadk")
    except PackageNotFoundError:
        return "0.0.0+unknown"


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="commonadk",
        description=(
            "Define an agent system once, in a framework-neutral common/ "
            "folder, and build it on any supported agent SDK."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"commonadk {_version()}"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_p = subparsers.add_parser(
        "validate", help="Load and validate a common/ project folder"
    )
    validate_p.add_argument("common_dir", help="Path to the project's common/ folder")

    render_p = subparsers.add_parser(
        "render", help="Regenerate interaction-layer.md from interactions.yaml"
    )
    render_p.add_argument("common_dir", help="Path to the project's common/ folder")

    run_p = subparsers.add_parser(
        "run", help="Build an agent for a target SDK and run one turn"
    )
    run_p.add_argument("common_dir", help="Path to the project's common/ folder")
    run_p.add_argument(
        "--target",
        required=True,
        metavar="{google-adk,openai,claude,crewai,autogen,langgraph}",
        help="Agent SDK to build against: google-adk, openai, claude, crewai, autogen, or langgraph",
    )
    run_p.add_argument(
        "--agent",
        default=None,
        help="Agent to run (default: the project's entry agent)",
    )
    run_p.add_argument("prompt", help="The user message to send")

    return parser


# ---------------------------------------------------------------------------
# shared loading helper
# ---------------------------------------------------------------------------


def _load_project(common_dir: str) -> tuple["Project", list[warnings.WarningMessage]]:
    """`loader.load`, with every warning it raises captured instead of just
    printed to stderr by Python's default warning machinery -- so callers can
    fold them into the CLI's own output ("show them, don't swallow them")."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        project = load(common_dir)
    return project, list(caught)


def _print_warnings(caught: list[warnings.WarningMessage]) -> None:
    if not caught:
        return
    print("\nWarnings:")
    for w in caught:
        print(f"  - {w.message}")


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def _cmd_validate(common_dir: str) -> int:
    project, caught_warnings = _load_project(common_dir)

    lines: list[str] = []
    entry = project.config.entry or project.graph.entry
    lines.append(f"Project: {project.config.name}  (entry agent: {entry})")
    lines.append("")
    lines.append("Agents:")

    for name in sorted(project.agents):
        spec = project.agents[name]
        raw_model = spec.config.model or f"{project.config.default_model} (project default)"
        resolved_model = project.resolve_model(name)
        lines.append(f"  {name}")
        lines.append(f"    model: {raw_model} -> {resolved_model}")

        tool_names = ", ".join(sorted(t.name for t in spec.tools))
        lines.append(f"    tools: {tool_names or '(none)'}")

        env_reqs = spec.config.requires.env
        if not env_reqs:
            lines.append("    env: (none required)")
        else:
            lines.append("    env:")
            for req in env_reqs:
                is_set = bool(os.environ.get(req.name))
                status = "set" if is_set else "not set"
                required = "required" if req.required else "optional"
                desc = f" -- {req.description}" if req.description else ""
                lines.append(f"      {req.name}: {status} ({required}){desc}")

    print("\n".join(lines))
    _print_warnings(caught_warnings)
    return 0


def _cmd_render(common_dir: str) -> int:
    project, caught_warnings = _load_project(common_dir)
    out_path = write_interaction_layer(common_dir, project.graph)
    print(f"Wrote {out_path}")
    _print_warnings(caught_warnings)
    return 0


def _cmd_run(common_dir: str, target: str, agent: Optional[str], prompt: str) -> int:
    project, caught_warnings = _load_project(common_dir)
    _print_warnings(caught_warnings)

    agent_name = agent or project.config.entry or project.graph.entry
    if agent_name is None:
        raise ValueError(
            "commonadk: no --agent given and the project has no entry agent"
        )
    if agent_name not in project.agents:
        raise ValueError(
            f"commonadk: unknown agent {agent_name!r}. Known agents: "
            f"{sorted(project.agents)}"
        )

    runner = _RUN_TARGETS.get(target)
    if runner is None:
        # Not one of this CLI's known targets -- delegate to the adapter
        # registry purely for its "unknown target, known targets are ..."
        # error message, so the CLI never hand-maintains a second list of
        # valid targets that can drift from adapters/__init__.py's.
        from .adapters import get_adapter

        get_adapter(target)
        raise AssertionError(f"unreachable: get_adapter({target!r}) did not raise")

    output = runner(project, agent_name, prompt)
    print(output)
    return 0


# -- per-target execution (SDK imports are lazy, inside these functions) ----


def _run_google_adk(project: "Project", agent_name: str, prompt: str) -> str:
    import asyncio

    from google.adk.runners import InMemoryRunner
    from google.genai import types as genai_types

    agent = project.build(agent_name, target="google-adk")
    runner = InMemoryRunner(agent=agent, app_name=project.config.name)
    user_id = "commonadk-cli"

    async def _invoke() -> str:
        session = await runner.session_service.create_session(
            app_name=runner.app_name, user_id=user_id
        )
        message = genai_types.Content(
            role="user", parts=[genai_types.Part(text=prompt)]
        )
        chunks: list[str] = []
        async for event in runner.run_async(
            user_id=user_id, session_id=session.id, new_message=message
        ):
            if event.is_final_response() and event.content and event.content.parts:
                chunks.extend(
                    part.text
                    for part in event.content.parts
                    if getattr(part, "text", None)
                )
        return "\n".join(chunks)

    return asyncio.run(_invoke())


def _run_openai(project: "Project", agent_name: str, prompt: str) -> str:
    from agents import Runner

    agent = project.build(agent_name, target="openai")
    result = Runner.run_sync(agent, prompt)
    return str(result.final_output)


def _run_claude(project: "Project", agent_name: str, prompt: str) -> str:
    import asyncio

    from claude_agent_sdk import ResultMessage, query

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise OSError(
            "commonadk: missing required environment variable for target "
            "'claude': ANTHROPIC_API_KEY (the Claude Agent SDK's bundled "
            "CLI needs it to authenticate with the Anthropic API)"
        )

    options = project.build(agent_name, target="claude")

    async def _invoke() -> str:
        chunks: list[str] = []
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, ResultMessage) and message.result:
                chunks.append(message.result)
        return "\n".join(chunks)

    return asyncio.run(_invoke())


def _run_crewai(project: "Project", agent_name: str, prompt: str) -> str:
    from crewai import Process, Task

    crew = project.build(agent_name, target="crewai")

    # Hierarchical crews (build root has outgoing edges): leave the task
    # unassigned -- the manager agent (the build root) picks which crew
    # member actually executes it. Sequential crews here are always the
    # solo-member fallback (see crewai_adapter.py's module docstring,
    # "Manager-or-solo-member decision"), so the one member must be
    # assigned explicitly.
    agent = None if crew.process == Process.hierarchical else crew.agents[0]
    crew.tasks = [
        Task(
            description=prompt,
            expected_output="A complete response to the request above.",
            agent=agent,
        )
    ]
    result = crew.kickoff()
    return str(result.raw)


def _run_autogen(project: "Project", agent_name: str, prompt: str) -> str:
    import asyncio

    built = project.build(agent_name, target="autogen")

    # `built` is either a bare `AssistantAgent` (build root has no outgoing
    # edges -- see autogen_adapter.py's module docstring, "WHAT build()
    # RETURNS") or a ready-to-run `Swarm` team (build root has at least one
    # outgoing edge). Both expose the same `async .run(task=...) ->
    # TaskResult` shape, so no branching on the return type is needed here.
    result = asyncio.run(built.run(task=prompt))
    return str(result.messages[-1].content)


def _run_langgraph(project: "Project", agent_name: str, prompt: str) -> str:
    graph = project.build(agent_name, target="langgraph")

    # `graph` is either a lone react agent's own `CompiledStateGraph` (build
    # root has no outgoing edges) or the compiled multi-agent `StateGraph`
    # (build root has at least one) -- see langgraph_adapter.py's module
    # docstring, "WHAT build() RETURNS". Both expose the same `.invoke(...)`
    # shape over a `MessagesState`-style input, so no branching on the
    # return type is needed here.
    result = graph.invoke({"messages": [{"role": "user", "content": prompt}]})
    return str(result["messages"][-1].content)


_RUN_TARGETS = {
    "google-adk": _run_google_adk,
    "openai": _run_openai,
    "claude": _run_claude,
    "crewai": _run_crewai,
    "autogen": _run_autogen,
    "langgraph": _run_langgraph,
}


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "validate":
            return _cmd_validate(args.common_dir)
        if args.command == "render":
            return _cmd_render(args.common_dir)
        if args.command == "run":
            return _cmd_run(args.common_dir, args.target, args.agent, args.prompt)
        parser.print_help()
        return 1
    except ValidationError as e:
        print(str(e), file=sys.stderr)
        return 1
    except (OSError, ValueError, ImportError) as e:
        print(f"commonadk: {e}" if not str(e).startswith("commonadk") else str(e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
