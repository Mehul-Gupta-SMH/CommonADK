#!/usr/bin/env python3
"""Verified live runs (issue #8) -- one real turn per target, against the
`examples/live-smoke/common` project, using a single `ANTHROPIC_API_KEY`.

This is the one script in this repo that is allowed to spend real money: it
sends an actual prompt to a real Anthropic model, once per requested target.
Every other check in this codebase (`commonadk validate`, `examples/demo.py`,
the whole `pytest` suite) is deliberately offline; this script is the
opposite by design, and exists specifically so a maintainer can prove
`commonadk run` actually works end to end, not just that it builds.

Why one Anthropic key can drive all six targets: every adapter has an
Anthropic path (verified against each adapter's `_model_for`/`_client_for`/
`_llm_for` -- see docs/demo-runs.md, "Live runs", and each adapter's own
module docstring for the source read):

    claude      native -- the SDK speaks Anthropic only
    google-adk  google.adk.models.lite_llm.LiteLlm(model="anthropic/...")
    openai      agents.extensions.models.litellm_model.LitellmModel(model="anthropic/...")
    crewai      crewai.LLM(model="anthropic/...") -- routes to a native client itself
    autogen     autogen_ext.models.anthropic.AnthropicChatCompletionClient
    langgraph   langchain.chat_models.init_chat_model("anthropic:...")

`examples/live-smoke/common`'s `default_model` is `anthropic/claude-haiku-4-5`
(the cheapest current Anthropic model) with NO per-agent
`targets.<sdk>.model` override anywhere -- unlike `examples/research-crew`
(gemini-default, needs a `targets.claude.model` override to build for
"claude" at all), this project routes every target through the exact same
model string. `claude-haiku-4-5`, `claude-sonnet-5`, and `claude-opus-5` are
the only valid current Anthropic model ids used anywhere in this repo --
never a date-suffixed id.

Usage:

    python3 scripts/live_smoke.py --list
    python3 scripts/live_smoke.py --dry-run
    ANTHROPIC_API_KEY=sk-... python3 scripts/live_smoke.py
    ANTHROPIC_API_KEY=sk-... python3 scripts/live_smoke.py --targets claude,openai
    ANTHROPIC_API_KEY=sk-... python3 scripts/live_smoke.py --model claude-sonnet-5

Per target, this script builds the entry agent, runs exactly one real turn,
and records: outcome (success/error), wall time, a truncated final answer,
and -- for `google-adk`/`openai`, the two targets `commonadk.runners`
supports (issue #22) -- the full normalized trace with token counts and
cost, honest about any gap (`usage_complete`/`cost_complete`, never a
silently-summed partial total). The other four targets have no runner yet,
so this script records `"usage": "unavailable"` for them explicitly --
never `0`, which would misleadingly claim the call was free.

A missing `ANTHROPIC_API_KEY` fails loudly and immediately, before touching
any target and before writing any report file -- never a hang, never an
empty/misleading report. `--dry-run` runs everything up to (never
including) the actual model call, so the whole pipeline -- project load,
per-target `build()`, report writing -- can be exercised with no API key and
no cost; `--list` needs neither a key nor a loaded project.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMMON_DIR = REPO_ROOT / "examples" / "live-smoke" / "common"

SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path and (SRC / "commonadk").is_dir():
    sys.path.insert(0, str(SRC))

# Canonical target order -- matches adapters/__init__.py's registry and
# every other target-listing surface in this repo (README's "Supported
# targets" table, cli.py's --target metavar, ...).
TARGET_ORDER = ["google-adk", "openai", "claude", "crewai", "autogen", "langgraph"]

# The two targets commonadk.runners has a runner for today (issue #22) --
# only these get a full normalized trace with token/cost data. The other
# four are run through the same build-and-print path `commonadk run` itself
# uses for them (cli._RUN_TARGETS), which has no usage data to report at
# all -- see the module docstring's "usage: unavailable" note.
RUNNER_TARGETS = {"google-adk", "openai"}

# The only three current, valid Anthropic model ids this project uses
# anywhere -- never a date-suffixed id (see pricing.py, which prices exactly
# these three). haiku is the default: cheapest viable model for a plumbing
# smoke test; sonnet/opus are opt-in for a heavier check.
VALID_MODELS = ["claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5"]
DEFAULT_MODEL = "claude-haiku-4-5"

# Short, deterministic, no network, no PII -- reliably elicits exactly one
# count_words call and a short answer from a small model (see
# examples/live-smoke/common/assistant/skill.md).
DEFAULT_PROMPT = "The quick brown fox jumps over the lazy dog."

FINAL_TEXT_TRUNCATE_CHARS = 500


@dataclass
class TargetResult:
    target: str
    status: str  # "success" | "error" | "dry_run"
    wall_time_s: float
    final_text: Optional[str] = None
    final_text_truncated: bool = False
    usage: Any = None  # dict (runner targets) | "unavailable" | "not_called (dry run)" | None
    trace_file: Optional[str] = None
    error: Optional[str] = None


def _truncate(text: Optional[str], limit: int = FINAL_TEXT_TRUNCATE_CHARS) -> tuple[Optional[str], bool]:
    if text is None:
        return None, False
    if len(text) <= limit:
        return text, False
    return text[:limit] + "...", True


def _sdk_available(target: str) -> bool:
    """Whether `target`'s adapter SDK is importable, without raising."""
    from commonadk.adapters import get_adapter

    try:
        get_adapter(target)
        return True
    except ImportError:
        return False


def cmd_list(targets: list[str]) -> int:
    from commonadk.runners import known_targets as runner_known_targets

    traced = set(runner_known_targets())
    print(f"{'target':<12} {'sdk installed':<15} {'has runner (tokens/cost)':<28}")
    for target in targets:
        installed = "yes" if _sdk_available(target) else "no"
        has_runner = "yes" if target in traced else "no (usage: unavailable)"
        print(f"{target:<12} {installed:<15} {has_runner:<28}")
    print(f"\nDefault model: {DEFAULT_MODEL} (also valid: {', '.join(VALID_MODELS[1:])})")
    print(f"Default project: {DEFAULT_COMMON_DIR}")
    return 0


def run_target(
    project: Any,
    agent_name: str,
    target: str,
    prompt: str,
    *,
    dry_run: bool,
    out_dir: Path,
) -> TargetResult:
    t0 = time.monotonic()

    if dry_run:
        try:
            project.build(agent_name, target=target)
        except Exception as exc:  # noqa: BLE001 -- report every failure, not just expected ones
            return TargetResult(
                target=target,
                status="error",
                wall_time_s=time.monotonic() - t0,
                usage="not_called (dry run)",
                error=f"{type(exc).__name__}: {exc}",
            )
        return TargetResult(
            target=target,
            status="dry_run",
            wall_time_s=time.monotonic() - t0,
            usage="not_called (dry run)",
        )

    if target in RUNNER_TARGETS:
        return _run_via_runner(project, agent_name, target, prompt, out_dir, t0)
    return _run_via_cli_path(project, agent_name, target, prompt, t0)


def _run_via_runner(
    project: Any, agent_name: str, target: str, prompt: str, out_dir: Path, t0: float
) -> TargetResult:
    from commonadk.runners import RunError, RunFinished, get_runner

    runner = get_runner(target)
    trace = None
    try:
        trace = runner.run_sync(project, agent_name, prompt)
    except Exception as exc:  # noqa: BLE001
        # The runner's own contract (docs/runner-design.md, "Fatal vs.
        # inline RunError") still leaves a hook-observed Trace unavailable
        # here since we didn't register one -- run_sync re-raises, so
        # build our own minimal record from the exception. The runner
        # layer's CLI integration (cli.py's --trace) shows the pattern for
        # capturing every event up to the fatal RunError via a hook, which
        # this script does not need: it only cares about the terminal
        # outcome, not the intermediate step trace, when a run fails.
        # A runner target that failed before producing any Trace has
        # genuinely no usage data -- None here, distinct from the four
        # unported targets' "unavailable" (which means "this target has no
        # runner", not "this specific call reported nothing").
        return TargetResult(
            target=target,
            status="error",
            wall_time_s=time.monotonic() - t0,
            usage=None,
            error=f"{type(exc).__name__}: {exc}",
        )

    trace_path = out_dir / f"{target}-trace.json"
    trace.write(trace_path)

    final_event = trace.final_event()
    wall_time_s = time.monotonic() - t0

    if isinstance(final_event, RunFinished):
        final_text, truncated = _truncate(final_event.final_text)
        usage = trace.rollup()["llm_calls"]
        return TargetResult(
            target=target,
            status="success",
            wall_time_s=wall_time_s,
            final_text=final_text,
            final_text_truncated=truncated,
            usage=usage,
            trace_file=trace_path.name,
        )

    # Ended in RunError (inline errors don't end a run -- see
    # docs/runner-design.md, "Fatal vs. inline RunError" -- so reaching
    # here without an exception having been raised should not happen, but
    # is handled honestly rather than assumed away).
    message = final_event.message if isinstance(final_event, RunError) else "run did not finish"
    return TargetResult(
        target=target,
        status="error",
        wall_time_s=wall_time_s,
        usage=trace.rollup()["llm_calls"],
        trace_file=trace_path.name,
        error=message,
    )


def _run_via_cli_path(project: Any, agent_name: str, target: str, prompt: str, t0: float) -> TargetResult:
    from commonadk.cli import _RUN_TARGETS

    runner_fn = _RUN_TARGETS[target]
    try:
        final_text = runner_fn(project, agent_name, prompt)
    except Exception as exc:  # noqa: BLE001
        return TargetResult(
            target=target,
            status="error",
            wall_time_s=time.monotonic() - t0,
            usage="unavailable",
            error=f"{type(exc).__name__}: {exc}",
        )

    truncated_text, was_truncated = _truncate(str(final_text))
    return TargetResult(
        target=target,
        status="success",
        wall_time_s=time.monotonic() - t0,
        final_text=truncated_text,
        final_text_truncated=was_truncated,
        # Explicitly "unavailable", never 0 or omitted -- none of the four
        # unported targets (claude, crewai, autogen, langgraph) go through
        # commonadk.runners, so no token/cost data was ever observed for
        # this call. This is a fact about the runner layer's current
        # coverage, not a claim that the call was free.
        usage="unavailable",
    )


def _print_summary_table(results: list[TargetResult]) -> None:
    header = f"{'target':<12} {'status':<10} {'wall_s':>8} {'tokens':>8} {'cost_usd':>10}  final_text"
    print(header)
    print("-" * len(header))
    for r in results:
        tokens = "-"
        cost = "-"
        if isinstance(r.usage, dict):
            tokens = str(r.usage.get("total_tokens") if r.usage.get("total_tokens") is not None else "?")
            cost = f"{r.usage['cost_usd']:.6f}" if r.usage.get("cost_usd") is not None else "?"
        text = (r.final_text or r.error or "").replace("\n", " ")
        if len(text) > 60:
            text = text[:60] + "..."
        print(f"{r.target:<12} {r.status:<10} {r.wall_time_s:>8.2f} {tokens:>8} {cost:>10}  {text}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="live_smoke.py",
        description="Run one real turn per target against examples/live-smoke/common.",
    )
    parser.add_argument(
        "--targets",
        default=",".join(TARGET_ORDER),
        help=f"Comma-separated subset of targets to run (default: all six -- {', '.join(TARGET_ORDER)})",
    )
    parser.add_argument("--list", action="store_true", help="Print the target table and exit -- no API key needed")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build each target's entry agent but never call the model -- no API key needed",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=VALID_MODELS,
        help=f"Anthropic model id to use for every target (default: {DEFAULT_MODEL}, the cheapest current "
        "model; claude-sonnet-5/claude-opus-5 are opt-in for a heavier check)",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt to send on the one real turn")
    parser.add_argument(
        "--common-dir",
        default=str(DEFAULT_COMMON_DIR),
        help=f"Path to the live-smoke common/ project (default: {DEFAULT_COMMON_DIR})",
    )
    parser.add_argument(
        "--out-dir",
        default=".",
        help="Directory to write the JSON report and per-target trace files into (default: current directory)",
    )
    parser.add_argument(
        "--report-name",
        default="live_smoke_report.json",
        help="Filename for the JSON report, written under --out-dir",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    requested_targets = [t.strip() for t in args.targets.split(",") if t.strip()]

    unknown = [t for t in requested_targets if t not in TARGET_ORDER]
    if unknown:
        print(f"live_smoke: unknown target(s) {unknown}. Known targets: {TARGET_ORDER}", file=sys.stderr)
        return 2

    if args.list:
        return cmd_list(requested_targets)

    # Fail loudly and immediately -- before loading the project, before
    # touching any target, before writing any file -- rather than hanging
    # on the first real SDK call or producing a report that looks like a
    # real (all-empty) run. --dry-run never calls a model, so it is exempt.
    if not args.dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "live_smoke: ANTHROPIC_API_KEY is not set in the environment. "
            "Every target in this script routes through an Anthropic model "
            "(see the module docstring) and this is not a hang-and-wait "
            "situation -- set ANTHROPIC_API_KEY (e.g. from the CLAUDE_API_KEY "
            "secret, mapped in .github/workflows/live-runs.yml) or pass "
            "--dry-run/--list to exercise this script without one.",
            file=sys.stderr,
        )
        return 2

    common_dir = Path(args.common_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import commonadk

    try:
        project = commonadk.load(str(common_dir))
    except Exception as exc:  # noqa: BLE001
        print(f"live_smoke: failed to load project at {common_dir}: {exc}", file=sys.stderr)
        return 2

    # Apply --model in-memory (never touching the committed config.yaml):
    # every agent in this project resolves through `default_model` with no
    # alias and no per-target override, so overwriting it here is
    # sufficient and mirrors the pattern already used in this codebase to
    # exercise post-load mutation (tests/test_models.py,
    # test_resolve_model_unknown_alias_raises).
    project.config.default_model = f"anthropic/{args.model}"

    agent_name = project.config.entry or project.graph.entry
    if agent_name is None:
        print(f"live_smoke: {common_dir} has no entry agent", file=sys.stderr)
        return 2

    results: list[TargetResult] = []
    for target in requested_targets:
        print(f"--- {target} ---", flush=True)
        result = run_target(
            project, agent_name, target, args.prompt, dry_run=args.dry_run, out_dir=out_dir
        )
        results.append(result)
        if result.error:
            print(f"    FAILED: {result.error}")
        else:
            print(f"    ok ({result.wall_time_s:.2f}s)")

    print()
    _print_summary_table(results)

    succeeded = sum(1 for r in results if r.status in ("success", "dry_run"))
    failed = [r for r in results if r.status == "error"]

    report = {
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "prompt": args.prompt,
        "common_dir": str(common_dir),
        "dry_run": args.dry_run,
        "targets_requested": requested_targets,
        "results": [asdict(r) for r in results],
        "summary": {
            "total": len(results),
            "succeeded": succeeded,
            "failed": len(failed),
        },
    }
    report_path = out_dir / args.report_name
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote report: {report_path}")

    if failed:
        print(f"\n{len(failed)} of {len(results)} target(s) FAILED:", file=sys.stderr)
        for r in failed:
            print(f"  - {r.target}: {r.error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:  # pragma: no cover -- interactive convenience only
        print("\nlive_smoke: interrupted", file=sys.stderr)
        sys.exit(130)
