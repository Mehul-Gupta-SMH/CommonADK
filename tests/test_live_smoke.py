"""Tests for `scripts/live_smoke.py` (issue #8 -- verified live runs).

Everything here is offline: no test in this file makes a real network call
or needs a real `ANTHROPIC_API_KEY`. Three different techniques keep it
that way, matched to what each test actually needs to exercise:

- `--list` and the missing-API-key path never call `project.build(...)` at
  all, so they're exercised as real subprocesses with no SDK gating needed.
- `--dry-run` DOES call `project.build(...)` for every requested target
  (that's the whole point -- prove the plumbing builds), so its end-to-end
  test is gated by `pytest.importorskip` for all six adapter SDKs at module
  scope, mirroring `tests/test_demo.py`'s own gating style for the same
  reason.
- The report-shape / usage-marker / failure-handling tests monkeypatch
  `commonadk.cli._RUN_TARGETS` and `commonadk.runners.get_runner` directly,
  so `scripts/live_smoke.py`'s own logic (not any adapter's) is what's
  under test -- no SDK needed, no `--dry-run`, and no real model call,
  while still exercising the exact code path a real (non-dry-run) run
  takes.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LIVE_SMOKE_SCRIPT = REPO_ROOT / "scripts" / "live_smoke.py"
LIVE_SMOKE_COMMON = REPO_ROOT / "examples" / "live-smoke" / "common"

ALL_TARGETS = ["google-adk", "openai", "claude", "crewai", "autogen", "langgraph"]
ALL_TARGETS_MODULES = [
    "google.adk",
    "agents",
    "claude_agent_sdk",
    "crewai",
    "autogen_agentchat",
    "langgraph",
]
def _load_live_smoke_module():
    """Import scripts/live_smoke.py as a module without needing it on
    sys.path permanently -- it's a standalone script, not part of the
    installed `commonadk` package."""
    spec = importlib.util.spec_from_file_location("live_smoke", LIVE_SMOKE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec -- the script's `TargetResult`
    # dataclass uses `from __future__ import annotations` (string
    # annotations), and dataclasses resolves those via
    # sys.modules[cls.__module__], which fails if the module was never
    # registered under that name.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def live_smoke():
    return _load_live_smoke_module()


def _run_subprocess(args: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    import os

    full_env = dict(os.environ)
    if env is not None:
        full_env.update(env)
    return subprocess.run(
        [sys.executable, str(LIVE_SMOKE_SCRIPT), *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
        env=full_env,
    )


# ---------------------------------------------------------------------------
# examples/live-smoke/common itself: must pass commonadk validate, route
# every target through the same Anthropic default_model with no per-target
# override.
# ---------------------------------------------------------------------------


def test_live_smoke_script_and_project_exist():
    assert LIVE_SMOKE_SCRIPT.is_file()
    assert LIVE_SMOKE_COMMON.is_dir()


def test_live_smoke_project_validates_and_has_no_per_target_model_overrides():
    import commonadk

    project = commonadk.load(str(LIVE_SMOKE_COMMON))
    assert project.config.default_model == "anthropic/claude-haiku-4-5"
    entry = project.config.entry or project.graph.entry
    assert entry in project.agents
    for name, spec in project.agents.items():
        assert spec.config.targets == {}, (
            f"{name} carries a targets.<sdk>.model override -- the whole point "
            "of this project is that one Anthropic default_model drives every "
            "target unmodified"
        )
        assert project.resolve_model(name) == "anthropic/claude-haiku-4-5"


# ---------------------------------------------------------------------------
# --list: no API key, no project SDK needed at all.
# ---------------------------------------------------------------------------


def test_list_exits_zero_and_names_every_target_without_an_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = _run_subprocess(["--list"], env={"ANTHROPIC_API_KEY": ""})
    assert result.returncode == 0, result.stderr
    for target in ALL_TARGETS:
        assert target in result.stdout
    # This once asserted that some target reported "unavailable", which held
    # while only google-adk and openai had runners. Issue #22 ported all six,
    # so nothing is unavailable any more and asserting otherwise would be
    # asserting a regression. What the column is actually for is honest
    # reporting of runner availability, so assert that directly against the
    # registry -- which keeps this test meaningful if a future adapter
    # (issue #11) lands before its runner and reintroduces an unported target.
    from commonadk.runners import known_unported_targets

    for target in known_unported_targets():
        assert target in result.stdout
    assert result.stdout.count("yes") >= len(ALL_TARGETS) - len(known_unported_targets())


# ---------------------------------------------------------------------------
# Missing ANTHROPIC_API_KEY: fails loudly and immediately, no report written.
# ---------------------------------------------------------------------------


def test_missing_api_key_fails_loudly_and_writes_no_report(tmp_path):
    out_dir = tmp_path / "out"
    result = _run_subprocess(
        ["--out-dir", str(out_dir)],
        env={"ANTHROPIC_API_KEY": ""},
    )
    assert result.returncode != 0
    assert "ANTHROPIC_API_KEY" in result.stderr
    # No report, no directory -- the failure happens before any file I/O.
    assert not out_dir.exists()


def test_missing_api_key_check_is_skipped_for_dry_run_and_list(tmp_path):
    pytest.importorskip("google.adk")
    out_dir = tmp_path / "out"
    result = _run_subprocess(
        ["--dry-run", "--targets", "google-adk", "--out-dir", str(out_dir)],
        env={"ANTHROPIC_API_KEY": ""},
    )
    # Whether this particular target's SDK is installed varies, but it must
    # not fail with the "missing API key" message -- --dry-run never needs one.
    assert "ANTHROPIC_API_KEY is not set" not in result.stderr


def test_unknown_target_is_rejected_before_any_api_key_check():
    result = _run_subprocess(["--targets", "not-a-real-target", "--dry-run"])
    assert result.returncode != 0
    assert "not-a-real-target" in result.stdout + result.stderr


# ---------------------------------------------------------------------------
# --dry-run: real project.build() for every target, no model call. Gated
# since it genuinely needs every adapter SDK installed.
# ---------------------------------------------------------------------------


def test_dry_run_builds_every_target_offline_and_writes_a_full_report(tmp_path):
    for mod in ALL_TARGETS_MODULES:
        pytest.importorskip(mod)

    out_dir = tmp_path / "out"
    result = _run_subprocess(
        ["--dry-run", "--out-dir", str(out_dir)],
        env={"ANTHROPIC_API_KEY": ""},
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    report_path = out_dir / "live_smoke_report.json"
    assert report_path.is_file()
    data = json.loads(report_path.read_text())

    assert data["dry_run"] is True
    assert data["summary"]["total"] == 6
    assert data["summary"]["succeeded"] == 6
    assert data["summary"]["failed"] == 0
    assert [r["target"] for r in data["results"]] == ALL_TARGETS
    for r in data["results"]:
        assert r["status"] == "dry_run"
        assert r["usage"] == "not_called (dry run)"
        assert r["error"] is None
        # No model call happened -- no trace file should exist for anyone,
        # including google-adk/openai which DO have a runner for real runs.
        assert r["trace_file"] is None


# ---------------------------------------------------------------------------
# Report shape, per-target usage marker, and failure handling -- driven
# in-process with commonadk.cli._RUN_TARGETS / commonadk.runners.get_runner
# monkeypatched, so no adapter SDK and no real model call is needed at all.
# ---------------------------------------------------------------------------


def _fake_trace_with_usage():
    from commonadk.runners import LLMCall, RunFinished, RunStarted, Trace

    trace = Trace()
    trace.append(RunStarted(run_id="r1", target="google-adk", agent_name="assistant", prompt="hi"))
    trace.append(
        LLMCall(
            run_id="r1",
            agent_name="assistant",
            model="claude-haiku-4-5",
            prompt_tokens=12,
            completion_tokens=8,
            total_tokens=20,
            cost_usd=0.00006,
        )
    )
    trace.append(
        RunFinished(
            run_id="r1",
            final_text="That text has 9 words.",
            total_prompt_tokens=12,
            total_completion_tokens=8,
            total_tokens=20,
            total_cost_usd=0.00006,
            usage_complete=True,
        )
    )
    return trace


def _install_fake_backends(monkeypatch, *, failing_target: str | None = None):
    """Replace every target's real execution path with a fast, offline
    double: commonadk.runners.get_runner for google-adk/openai (returns a
    fixed, real Trace built from real event dataclasses -- no SDK
    involved), commonadk.cli._RUN_TARGETS for the other four (returns a
    fixed string). Optionally make one target raise, to exercise the
    error/non-zero-exit path.
    """
    import commonadk.cli as cli_module
    import commonadk.runners as runners_module

    class _FakeRunner:
        def __init__(self, target):
            self.target = target

        def run_sync(self, project, agent_name, prompt, **kwargs):
            if self.target == failing_target:
                raise RuntimeError(f"simulated failure for {self.target}")
            return _fake_trace_with_usage()

    def fake_get_runner(target):
        return _FakeRunner(target)

    monkeypatch.setattr(runners_module, "get_runner", fake_get_runner)

    def make_fake_cli_runner(target):
        def _fake_run(project, agent_name, prompt):
            if target == failing_target:
                raise RuntimeError(f"simulated failure for {target}")
            return "That text has 9 words."

        return _fake_run

    for target in ("claude", "crewai", "autogen", "langgraph"):
        monkeypatch.setitem(cli_module._RUN_TARGETS, target, make_fake_cli_runner(target))


def test_report_shape_and_usage_marker_for_all_targets(live_smoke, monkeypatch, tmp_path):
    _install_fake_backends(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")

    out_dir = tmp_path / "out"
    rc = live_smoke.main(["--out-dir", str(out_dir)])
    assert rc == 0

    report_path = out_dir / "live_smoke_report.json"
    data = json.loads(report_path.read_text())

    # Top-level report shape.
    for key in (
        "schema_version",
        "timestamp",
        "model",
        "prompt",
        "common_dir",
        "dry_run",
        "targets_requested",
        "results",
        "summary",
    ):
        assert key in data
    assert data["model"] == "claude-haiku-4-5"
    assert data["dry_run"] is False
    assert data["summary"] == {"total": 6, "succeeded": 6, "failed": 0}

    results_by_target = {r["target"]: r for r in data["results"]}
    assert set(results_by_target) == set(ALL_TARGETS)

    # Which targets get real usage data is decided by the runner registry,
    # not by a hardcoded list. This test used to name google-adk/openai as
    # the runner-backed pair and the other four as permanently
    # "unavailable"; issue #22 ported all six, so that split is now a
    # description of a limitation that no longer exists. Keying off the
    # registry keeps both branches meaningful if an adapter ever lands
    # before its runner (issue #11).
    from commonadk.runners import known_targets as runner_known_targets

    traced = set(runner_known_targets())
    assert traced, "expected at least one runner-backed target"

    for target in ALL_TARGETS:
        r = results_by_target[target]
        assert r["status"] == "success"
        assert r["final_text"] == "That text has 9 words."

        if target in traced:
            assert isinstance(r["usage"], dict)
            assert r["usage"]["total_tokens"] == 20
            assert r["usage"]["cost_usd"] == pytest.approx(0.00006)
            assert r["usage"]["usage_complete"] is True
            assert r["trace_file"] == f"{target}-trace.json"
            assert (out_dir / r["trace_file"]).is_file()
        else:
            # No runner -- usage must be the explicit string "unavailable",
            # never 0, never null (either could be misread as "reported zero
            # usage", i.e. a free call).
            assert r["usage"] == "unavailable"
            assert r["usage"] != 0
            assert r["usage"] is not None
            assert r["trace_file"] is None


def test_a_failing_target_is_recorded_and_exits_nonzero_others_unaffected(
    live_smoke, monkeypatch, tmp_path
):
    _install_fake_backends(monkeypatch, failing_target="claude")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")

    out_dir = tmp_path / "out"
    rc = live_smoke.main(["--out-dir", str(out_dir)])
    assert rc == 1

    data = json.loads((out_dir / "live_smoke_report.json").read_text())
    assert data["summary"]["failed"] == 1
    assert data["summary"]["succeeded"] == 5

    results_by_target = {r["target"]: r for r in data["results"]}
    failed = results_by_target["claude"]
    assert failed["status"] == "error"
    assert "simulated failure for claude" in failed["error"]

    # Every other target still ran and succeeded -- one failure doesn't
    # abort the rest of the sweep.
    for target in ("google-adk", "openai", "crewai", "autogen", "langgraph"):
        assert results_by_target[target]["status"] == "success"


def test_a_failing_runner_target_records_error_with_no_trace_file(live_smoke, monkeypatch, tmp_path):
    _install_fake_backends(monkeypatch, failing_target="google-adk")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")

    out_dir = tmp_path / "out"
    rc = live_smoke.main(["--targets", "google-adk", "--out-dir", str(out_dir)])
    assert rc == 1

    data = json.loads((out_dir / "live_smoke_report.json").read_text())
    result = data["results"][0]
    assert result["status"] == "error"
    assert "simulated failure for google-adk" in result["error"]
    # No trace was ever produced for a run that failed before returning one.
    assert result["trace_file"] is None
    assert result["usage"] is None


def test_custom_prompt_and_model_flow_through_to_the_report(live_smoke, monkeypatch, tmp_path):
    _install_fake_backends(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")

    out_dir = tmp_path / "out"
    rc = live_smoke.main(
        [
            "--targets",
            "claude",
            "--model",
            "claude-sonnet-5",
            "--prompt",
            "Count these five words please.",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    data = json.loads((out_dir / "live_smoke_report.json").read_text())
    assert data["model"] == "claude-sonnet-5"
    assert data["prompt"] == "Count these five words please."


def test_invalid_model_is_rejected_by_argparse(live_smoke):
    with pytest.raises(SystemExit):
        live_smoke.main(["--dry-run", "--model", "claude-3-opus-20240229"])
