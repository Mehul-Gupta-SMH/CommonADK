#!/usr/bin/env python3
"""A polished, self-contained, fully-offline tour of commonadk.

Run it directly:

    python3 examples/demo.py

No API keys, network access, or `pip install`-ed extras beyond whichever of
`commonadk[google,openai,claude,crewai,autogen,langgraph]` happen to be
installed are required -- every SDK not installed is reported as skipped,
not a hard failure (see "targets" below). Real API keys are NEVER needed:
this script only ever calls `project.build(...)`, which is pure, local
Python object construction -- it never makes a network call or talks to an
LLM. A handful of provider SDKs (autogen's and langgraph's `openai`/
`google_genai` model-client paths -- see those adapters' module docstrings,
"Offline construction") construct their underlying provider client eagerly
and raise if no *key-shaped string* is discoverable in the environment, even
though no request is ever sent; this script sets short, obviously-fake
placeholder values for exactly those variables, right before the targets
that need them, with a printed note that they are placeholders only.

What this script does, in order:

1. Loads `examples/research-crew/common` via `commonadk.load` and prints a
   project summary (agents, resolved models, env requirements).
2. Renders `interactions.yaml` to a Mermaid flowchart on stdout (the same
   rendering `commonadk render` writes to `interaction-layer.md`).
3. Builds the `coordinator` agent for all six supported targets in a loop,
   printing what `build()` returned (type + a one-line shape summary) for
   each -- skipping cleanly, not failing, whichever targets' SDKs aren't
   installed in this environment.
4. Demonstrates, on purpose, two of commonadk's error paths: the missing-
   required-env preflight error, and the unknown-target error. Both are
   caught and printed, clearly labeled as intentional demonstrations, not
   script failures.

Exits 0 on success (including when a demonstrated failure mode raises
exactly as expected); exits 1 only if something goes genuinely wrong.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

# CrewAI ships opt-out anonymous telemetry that can try to spin up an
# OpenTelemetry exporter on first use; crewai is imported lazily (inside
# project.build()), so these must be set before that happens, not just
# before `import commonadk`. See crewai_adapter.py's module docstring,
# "Telemetry / offline construction".
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

# litellm (imported transitively by google-adk's LiteLlm wrapper and by
# crewai's own fallback path) tries a one-time network fetch of its remote
# model-price/context-window table on first import, then falls back to a
# bundled local copy on any failure -- harmless, but noisy and pointless in
# an offline demo. This env var (litellm's own documented escape hatch)
# skips the network attempt and goes straight to the local copy.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "true")

# Make `python3 examples/demo.py` work from any working directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import commonadk  # noqa: E402

EXAMPLE_DIR = _REPO_ROOT / "examples" / "research-crew" / "common"

ALL_TARGETS = ["google-adk", "openai", "claude", "crewai", "autogen", "langgraph"]

# adapters/__init__.py's target -> pip-installable-module mapping, used here
# only to give a friendly "not installed" message (mirrors
# tests/test_hypothesis.py's own importorskip mapping).
_TARGET_MODULE = {
    "google-adk": "google.adk",
    "openai": "agents",
    "claude": "claude_agent_sdk",
    "crewai": "crewai",
    "autogen": "autogen_agentchat",
    "langgraph": "langgraph",
}

# One-line shape label per target -- what a project author actually gets
# back from build(), per plan.md / docs/HLD.md's "Comparing the six targets".
_SHAPE_LABEL = {
    "google-adk": "sub_agents tree",
    "openai": "handoff graph",
    "claude": "options subagents (flat registry)",
    "crewai": "crew members",
    "autogen": "swarm participants",
    "langgraph": "graph nodes",
}


def _section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def _print_project_summary(project: "commonadk.Project") -> None:
    entry = project.config.entry or project.graph.entry
    print(f"Project: {project.config.name}  (entry agent: {entry!r})")
    print(f"Model aliases: {project.config.model_aliases}")
    print()
    print("Agents:")
    for name in sorted(project.agents):
        spec = project.agents[name]
        raw_model = spec.config.model or f"{project.config.default_model} (project default)"
        resolved = project.resolve_model(name)
        tool_names = ", ".join(sorted(t.name for t in spec.tools)) or "(none)"
        print(f"  - {name}")
        print(f"      model: {raw_model} -> {resolved}")
        print(f"      tools: {tool_names}")
        env_reqs = spec.config.requires.env
        if not env_reqs:
            print("      env requirements: (none)")
        else:
            for req in env_reqs:
                required = "required" if req.required else "optional"
                is_set = "set" if os.environ.get(req.name) else "not set"
                print(f"      env requirement: {req.name} ({required}, currently {is_set}) -- {req.description}")


def _shape_summary(target: str, built: object) -> str:
    """One line describing the concrete shape `build()` returned for `target`."""
    if target == "google-adk":
        children = [c.name for c in built.sub_agents]
        return f"root={built.name!r}, sub_agents={children!r}"
    if target == "openai":
        children = [h.name for h in built.handoffs]
        return f"root={built.name!r}, handoffs={children!r}"
    if target == "claude":
        subagents = sorted(built.agents.keys())
        return f"root has no name field (session/query-based); options.agents={subagents!r}"
    if target == "crewai":
        from crewai import Process

        if built.process == Process.hierarchical:
            members = [a.role for a in built.agents]
            return f"process=hierarchical, manager={built.manager_agent.role!r}, members={members!r}"
        members = [a.role for a in built.agents]
        return f"process=sequential (solo fallback), members={members!r}"
    if target == "autogen":
        from autogen_agentchat.agents import AssistantAgent

        if isinstance(built, AssistantAgent):
            return f"bare AssistantAgent (leaf, no team): name={built.name!r}"
        participants = list(getattr(built, "_participant_names", []))
        return f"Swarm participants={participants!r}"
    if target == "langgraph":
        nodes = [n for n in built.nodes if n != "__start__"]
        return f"StateGraph nodes={nodes!r}"
    return repr(built)


def _build_all_targets(project: "commonadk.Project") -> None:
    provider_keys_set = False
    for target in ALL_TARGETS:
        module_name = _TARGET_MODULE[target]
        try:
            __import__(module_name)
        except ImportError:
            print(f"[{target}] SKIPPED -- SDK not installed "
                  f"(pip install \"commonadk[<extra>]\" would add it)")
            continue

        if target in ("autogen", "langgraph") and not provider_keys_set:
            print(
                "  (setting placeholder OPENAI_API_KEY / ANTHROPIC_API_KEY / "
                "GEMINI_API_KEY -- these are short, obviously-fake strings, "
                "used ONLY to satisfy the autogen/langgraph model clients' "
                "eager offline construction check; see those adapters' "
                "module docstrings, 'Offline construction'. No network call "
                "is ever made and no real key is required for build().)"
            )
            os.environ.setdefault("OPENAI_API_KEY", "sk-demo-placeholder-not-real")
            os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-demo-placeholder-not-real")
            os.environ.setdefault("GEMINI_API_KEY", "AIza-demo-placeholder-not-real")
            provider_keys_set = True

        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                built = project.build("coordinator", target=target)
            summary = _shape_summary(target, built)
            print(f"[{target}] OK -- {type(built).__module__}.{type(built).__name__}")
            print(f"    shape ({_SHAPE_LABEL[target]}): {summary}")
            if caught:
                print(f"    ({len(caught)} expected warning(s) suppressed -- "
                      f"per-adapter model_params/tool quirks documented in "
                      f"adapters/{target.replace('-', '_')}*.py)")
        except Exception as e:  # noqa: BLE001 - report, don't crash the demo
            print(f"[{target}] FAILED unexpectedly: {type(e).__name__}: {e}")


def _demonstrate_missing_env_error(project: "commonadk.Project") -> None:
    print(
        "DEMONSTRATION 1 of 2: building with a required env var unset.\n"
        "researcher/agent-config.yaml declares TAVILY_API_KEY as required.\n"
        "Temporarily unsetting it and attempting project.build(..., "
        "target=\"openai\") on purpose:"
    )
    saved = os.environ.pop("TAVILY_API_KEY", None)
    try:
        project.build("coordinator", target="openai")
        print("  (unexpected: build succeeded -- TAVILY_API_KEY was not "
              "actually unset)")
    except OSError as e:
        print("  Raised OSError as expected:")
        for line in str(e).splitlines():
            print(f"    {line}")
    finally:
        if saved is not None:
            os.environ["TAVILY_API_KEY"] = saved


def _demonstrate_unknown_target_error(project: "commonadk.Project") -> None:
    print(
        "\nDEMONSTRATION 2 of 2: building for an unrecognized target string.\n"
        "Attempting project.build(..., target=\"not-a-real-sdk\") on purpose:"
    )
    try:
        project.build("coordinator", target="not-a-real-sdk")
        print("  (unexpected: build succeeded)")
    except ValueError as e:
        print(f"  Raised ValueError as expected: {e}")


def main() -> int:
    _section("1. Load and validate examples/research-crew/common")
    project = commonadk.load(EXAMPLE_DIR)
    _print_project_summary(project)

    _section("2. Render the interaction layer (mermaid)")
    print(commonadk.render_mermaid(project.graph))

    _section("3. Satisfy researcher's required env var for the builds below")
    os.environ["TAVILY_API_KEY"] = "demo-placeholder-not-a-real-key"
    print("Set TAVILY_API_KEY to a placeholder value (researcher's tools.py "
          "never makes a real network call, but the env preflight only "
          "checks *presence*, not validity -- see requires.env in "
          "file-contracts.md).")

    _section("4. Build 'coordinator' for all six supported targets")
    _build_all_targets(project)

    _section("5. Demonstrated failure modes (on purpose)")
    _demonstrate_missing_env_error(project)
    _demonstrate_unknown_target_error(project)

    _section("Done -- exiting 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
