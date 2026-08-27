"""Tests for the `commonadk` CLI (M4).

Every test here invokes `cli.main(argv)` in-process (no subprocess, no
network) and asserts on its integer return value plus stdout/stderr captured
via `capsys` -- the same in-process style `console_scripts` wrappers use
under the hood (`sys.exit(main())`).
"""

from __future__ import annotations

import yaml
import pytest

from commonadk import cli
from commonadk.mermaid import render_mermaid
from commonadk.loader import load


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def test_validate_success_mentions_agents_and_missing_env(
    example_common_dir, monkeypatch, capsys
):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("POSTGRES_DSN", raising=False)

    rc = cli.main(["validate", str(example_common_dir)])
    out = capsys.readouterr().out

    assert rc == 0
    assert "coordinator" in out
    assert "researcher" in out
    assert "writer" in out
    # TAVILY_API_KEY is a required env var for researcher and is unset here --
    # the summary must flag it, not just list its name.
    assert "TAVILY_API_KEY" in out
    assert "not set" in out
    tavily_line = next(line for line in out.splitlines() if "TAVILY_API_KEY" in line)
    assert "not set" in tavily_line


def test_validate_success_flags_set_env_as_set(example_common_dir, monkeypatch, capsys):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.delenv("POSTGRES_DSN", raising=False)

    rc = cli.main(["validate", str(example_common_dir)])
    out = capsys.readouterr().out

    assert rc == 0
    tavily_line = next(line for line in out.splitlines() if "TAVILY_API_KEY" in line)
    assert "not set" not in tavily_line
    assert "set" in tavily_line


def test_validate_broken_project_exits_1_and_prints_error(tmp_project, capsys):
    # tmp_project is a fresh, mutable copy of the example (backed by
    # pytest's tmp_path) -- introduce a typo'd tool name so validation fails.
    researcher_cfg = tmp_project / "researcher" / "agent-config.yaml"
    data = yaml.safe_load(researcher_cfg.read_text())
    data["tools"] = ["search_wbe"]  # typo of search_web
    researcher_cfg.write_text(yaml.safe_dump(data))

    rc = cli.main(["validate", str(tmp_project)])
    captured = capsys.readouterr()

    assert rc == 1
    assert "search_wbe" in captured.err
    assert "not defined as a function" in captured.err
    # the CLI must not have printed a success summary anywhere
    assert "Project: research-crew" not in captured.out


def test_validate_missing_project_folder_exits_1(tmp_path, capsys):
    rc = cli.main(["validate", str(tmp_path / "does-not-exist")])
    captured = capsys.readouterr()

    assert rc == 1
    assert "not found" in captured.err


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------


def test_render_writes_interaction_layer_matching_mermaid(tmp_project, capsys):
    out_path = tmp_project / "interaction-layer.md"
    out_path.write_text("stale content that must be replaced\n")

    rc = cli.main(["render", str(tmp_project)])
    captured = capsys.readouterr()

    assert rc == 0
    assert str(out_path) in captured.out

    project = load(tmp_project)
    expected_mermaid = render_mermaid(project.graph)
    content = out_path.read_text()
    assert expected_mermaid in content
    assert "stale content" not in content


def test_render_broken_project_exits_1(tmp_project, capsys):
    interactions_path = tmp_project / "interactions.yaml"
    data = yaml.safe_load(interactions_path.read_text())
    data["edges"].append({"from": "coordinator", "to": "nonexistent", "type": "delegate"})
    interactions_path.write_text(yaml.safe_dump(data))

    rc = cli.main(["render", str(tmp_project)])
    captured = capsys.readouterr()

    assert rc == 1
    assert "nonexistent" in captured.err


# ---------------------------------------------------------------------------
# run -- error paths, testable without network (env preflight and the
# adapter registry both fire before any SDK call happens)
# ---------------------------------------------------------------------------


def test_run_missing_required_env_var_exits_nonzero(example_common_dir, monkeypatch, capsys):
    pytest.importorskip("google.adk")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("POSTGRES_DSN", raising=False)

    rc = cli.main(
        ["run", str(example_common_dir), "--target", "google-adk", "research this"]
    )
    captured = capsys.readouterr()

    assert rc != 0
    assert "TAVILY_API_KEY" in captured.err
    assert "missing required environment variable" in captured.err


def test_run_claude_missing_anthropic_key_exits_nonzero(
    example_common_dir, monkeypatch, capsys
):
    pytest.importorskip("claude_agent_sdk")
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.delenv("POSTGRES_DSN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    rc = cli.main(
        ["run", str(example_common_dir), "--target", "claude", "research this"]
    )
    captured = capsys.readouterr()

    assert rc != 0
    assert "ANTHROPIC_API_KEY" in captured.err


def test_run_unknown_target_exits_nonzero_naming_known_targets(
    example_common_dir, monkeypatch, capsys
):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")

    rc = cli.main(
        ["run", str(example_common_dir), "--target", "not-a-real-sdk", "hi"]
    )
    captured = capsys.readouterr()

    assert rc != 0
    assert "not-a-real-sdk" in captured.err
    assert "google-adk" in captured.err
    assert "openai" in captured.err


def test_run_unknown_agent_exits_nonzero(example_common_dir, monkeypatch, capsys):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")

    rc = cli.main(
        [
            "run",
            str(example_common_dir),
            "--target",
            "google-adk",
            "--agent",
            "nonexistent",
            "hi",
        ]
    )
    captured = capsys.readouterr()

    assert rc != 0
    assert "nonexistent" in captured.err
    assert "coordinator" in captured.err


def test_run_broken_project_exits_1_before_touching_any_sdk(tmp_project, capsys):
    researcher_cfg = tmp_project / "researcher" / "agent-config.yaml"
    data = yaml.safe_load(researcher_cfg.read_text())
    data["tools"] = ["search_wbe"]  # typo of search_web
    researcher_cfg.write_text(yaml.safe_dump(data))

    rc = cli.main(["run", str(tmp_project), "--target", "google-adk", "hi"])
    captured = capsys.readouterr()

    assert rc == 1
    assert "search_wbe" in captured.err


# ---------------------------------------------------------------------------
# --version
# ---------------------------------------------------------------------------


def test_version_prints_and_exits_0(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--version"])

    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "commonadk" in out
