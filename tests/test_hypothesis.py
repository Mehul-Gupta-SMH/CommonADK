"""The cross-target hypothesis test (plan.md "Hypothesis" / "v1 success criterion").

plan.md's bet is that `common/` can be the single source of truth: the same
project builds, unmodified, on every supported SDK target. This test is
that claim made executable -- it grows by one target per adapter milestone
(M2: google-adk, M3: openai, M5: claude) rather than living duplicated
inside each adapter's own test file, so the claim stays visible as one
parametrized test instead of scattered assertions.

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


@pytest.mark.parametrize("target", ["google-adk", "openai", "claude"])
def test_same_project_builds_on_every_installed_target(
    example_common_dir, tavily_env, target
):
    # This is the v1 success criterion from plan.md ("Hypothesis"): the same
    # `common/` folder must build unmodified on every supported SDK target.
    # Load the project ONCE and build the same entry agent under whichever
    # targets are actually installed in this environment.
    importorskip_module = {
        "google-adk": "google.adk",
        "openai": "agents",
        "claude": "claude_agent_sdk",
    }[target]
    pytest.importorskip(importorskip_module)

    project = commonadk.load(example_common_dir)

    agent = project.build("coordinator", target=target)

    assert agent is not None
    # Every adapter's returned object carries the agent's name/instructions
    # somewhere, but under a different attribute name per SDK (Google ADK
    # and OpenAI Agents return a live agent object with `.name`; the Claude
    # Agent SDK returns a `ClaudeAgentOptions` with no `.name` field at all
    # -- see claude_agent.py's module docstring, "WHAT build() RETURNS").
    if target == "claude":
        assert agent.system_prompt.strip() != ""
    else:
        assert agent.name == "coordinator"
