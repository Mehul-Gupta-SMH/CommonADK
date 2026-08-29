"""Runs `examples/demo.py` end-to-end as a real subprocess.

This is deliberately a subprocess test, not an in-process one: the whole
point of `examples/demo.py` is that it works as a standalone script a human
runs with `python3 examples/demo.py`, with no special test-harness state
(fixtures, monkeypatched env, etc.) behind it -- it must self-provision
every placeholder env var it needs. Running it out-of-process is what
actually proves that.

Gated by `pytest.importorskip` for all six adapter SDKs at module scope
(mirroring `test_adapter_*.py`'s own gating style) -- if any one of the six
extras isn't installed, this test skips cleanly rather than failing, since
`examples/demo.py` is documented to build all six targets.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("google.adk")
pytest.importorskip("agents")
pytest.importorskip("claude_agent_sdk")
pytest.importorskip("crewai")
pytest.importorskip("autogen_agentchat")
pytest.importorskip("langgraph")
pytest.importorskip("langchain")

import commonadk  # noqa: E402 - import after importorskip gating, deliberate

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_SCRIPT = REPO_ROOT / "examples" / "demo.py"


def test_demo_script_exists():
    assert DEMO_SCRIPT.is_file()


def test_demo_script_runs_end_to_end_and_exits_zero(monkeypatch):
    # Strip every env var the script is documented to self-provision, so a
    # green result here actually proves self-provisioning works -- not that
    # the ambient test environment happened to already have them set.
    for name in (
        "TAVILY_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "POSTGRES_DSN",
    ):
        monkeypatch.delenv(name, raising=False)

    result = subprocess.run(
        [sys.executable, str(DEMO_SCRIPT)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, (
        f"examples/demo.py exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )

    out = result.stdout

    # Section 1: project summary.
    assert "Project: research-crew" in out
    assert "coordinator" in out and "researcher" in out and "writer" in out

    # Section 2: rendered mermaid matches the same rendering commonadk render
    # would produce for the shipped example.
    project = commonadk.load(REPO_ROOT / "examples" / "research-crew" / "common")
    assert commonadk.render_mermaid(project.graph) in out

    # Section 4: every one of the six targets actually built successfully --
    # the whole point of the demo -- not silently skipped.
    for target in (
        "google-adk",
        "openai",
        "claude",
        "crewai",
        "autogen",
        "langgraph",
    ):
        assert f"[{target}] OK" in out, f"target {target!r} did not report OK in demo output"
        assert f"[{target}] SKIPPED" not in out

    # Section 5: both demonstrated failure modes actually fired.
    assert "DEMONSTRATION 1 of 2" in out
    assert "Raised OSError as expected" in out
    assert "TAVILY_API_KEY" in out
    assert "DEMONSTRATION 2 of 2" in out
    assert "Raised ValueError as expected" in out
    assert "Unknown build target 'not-a-real-sdk'" in out

    assert "Done -- exiting 0" in out
