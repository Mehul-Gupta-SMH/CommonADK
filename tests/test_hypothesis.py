"""The cross-target hypothesis test (plan.md "Hypothesis" / "v1 success criterion").

plan.md's bet is that `common/` can be the single source of truth: the same
project builds, unmodified, on every supported SDK target. This test is
that claim made executable -- it grows by one target per adapter milestone
(M2: google-adk, M3: openai, M5: claude, M6: crewai, M7: autogen, M8:
langgraph) rather than living duplicated inside each adapter's own test
file, so the claim stays visible as one parametrized test instead of
scattered assertions.

Each target is gated by its own `pytest.importorskip` *inside* the test
body (not at module scope) so this file collects and runs regardless of
which SDKs happen to be installed -- an uninstalled target's parametrized
case is skipped individually rather than failing the whole file, and the
other targets still run.
"""

from __future__ import annotations

import pytest

import commonadk


@pytest.fixture()
def tavily_env(monkeypatch):
    """Satisfy researcher's one *required* env var; POSTGRES_DSN stays unset
    on purpose -- it's declared optional and must never block a build."""
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.delenv("POSTGRES_DSN", raising=False)


@pytest.fixture()
def provider_keys_env(monkeypatch):
    """Fake provider API keys, only actually needed by the autogen and
    langgraph targets -- see autogen_adapter.py's and langgraph_adapter.py's
    module docstrings, "Offline construction": their model clients construct
    the underlying SDK client eagerly and raise if no key is discoverable.
    Harmless to set for every other target's build in this same
    parametrized test, since none of them read these vars."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test")
    monkeypatch.setenv("GOOGLE_API_KEY", "AIza-test")


@pytest.mark.parametrize(
    "target", ["google-adk", "openai", "claude", "crewai", "autogen", "langgraph"]
)
def test_same_project_builds_on_every_installed_target(
    example_common_dir, tavily_env, provider_keys_env, target
):
    # This is the v1 success criterion from plan.md ("Hypothesis"): the same
    # `common/` folder must build unmodified on every supported SDK target.
    # Load the project ONCE and build the same entry agent under whichever
    # targets are actually installed in this environment.
    importorskip_module = {
        "google-adk": "google.adk",
        "openai": "agents",
        "claude": "claude_agent_sdk",
        "crewai": "crewai",
        "autogen": "autogen_agentchat",
        "langgraph": "langgraph",
    }[target]
    pytest.importorskip(importorskip_module)

    project = commonadk.load(example_common_dir)

    agent = project.build("coordinator", target=target)

    assert agent is not None
    # Every adapter's returned object carries the agent's name/instructions
    # somewhere, but under a different attribute name per SDK (Google ADK
    # and OpenAI Agents return a live agent object with `.name`; the Claude
    # Agent SDK returns a `ClaudeAgentOptions` with no `.name` field at all
    # -- see claude_agent.py's module docstring, "WHAT build() RETURNS";
    # CrewAI returns a `Crew` whose build-root agent is `.manager_agent`,
    # not a member of `.agents` -- see crewai_adapter.py's module
    # docstring, "Manager-or-solo-member decision"; AutoGen returns a
    # `Swarm` team here -- coordinator has an outgoing edge -- whose first
    # participant is the build root, see autogen_adapter.py's module
    # docstring, "WHAT build() RETURNS"; LangGraph returns a compiled
    # multi-agent `StateGraph` here -- coordinator has an outgoing edge --
    # whose nodes are keyed by agent name, see langgraph_adapter.py's module
    # docstring, "WHAT build() RETURNS").
    if target == "claude":
        assert agent.system_prompt.strip() != ""
    elif target == "crewai":
        assert agent.manager_agent.role == "coordinator"
    elif target == "autogen":
        assert agent._participant_names[0] == "coordinator"
    elif target == "langgraph":
        assert "coordinator" in agent.nodes
    else:
        assert agent.name == "coordinator"
