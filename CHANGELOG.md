# Changelog

All notable changes to CommonADK are recorded here. Versions follow
[semantic versioning](https://semver.org/); while the project is pre-1.0,
breaking changes can land in a patch release and are called out explicitly.

## 0.0.2 — 2026-09-12

### Fixed

- **`pip install "commonadk[crewai]"` was broken for gemini and anthropic
  models.** The extra declared only `crewai[litellm]`, but CrewAI routes
  `gemini/...` and `anthropic/...` to *native* provider clients that litellm
  never sees, and those import `google-genai` and `anthropic` directly.
  Building any agent on such a model raised `ImportError` from
  `crewai/llms/providers/gemini/completion.py` before a single call was made
  — which covers most users, since the shipped example uses a gemini model.
  The extra now declares `crewai[litellm,google-genai,anthropic]`.

  This affected v0.0.1 on PyPI. It stayed hidden in development because the
  `google` extra pulls `google-genai` in transitively via `google-adk`, so
  any environment with more than one extra installed already had it.
- README images (banner, demo GIF) used repo-relative paths, so both
  rendered as broken images on the PyPI project page, where the description
  is served standalone. They now use absolute URLs.

### Added

- **Execution and telemetry layer for all six targets** (#22). Every target
  — `google-adk`, `openai`, `claude`, `crewai`, `autogen`, `langgraph` — now
  has a runner with normalized per-step events (run/agent/LLM call/tool
  call/transfer/error), JSON trace export, observe-only hooks, and token and
  cost metering. `commonadk run --stream` and `--trace out.json` work on all
  of them.

  Usage reporting is honest or absent: where an SDK does not report token
  counts, the trace carries `null` with `usage_complete: false`, never a `0`
  that would read as a free call. The Claude runner takes cost verbatim from
  the SDK's own `total_cost_usd` rather than a static price table; the CrewAI
  runner reports genuine per-call usage from CrewAI's event bus.

### Changed

- **`delegate` and `handoff` edges now build differently where an SDK can
  express the distinction** (#10, first of five items). LangGraph, Google
  ADK, OpenAI Agents and AutoGen each map `delegate` to a sub-call that
  returns and `handoff` to a transfer that does not. Claude Agent SDK and
  CrewAI keep the collapsed mapping, because neither SDK has any
  transfer-and-never-return primitive.

  **Behavior change:** on AutoGen and LangGraph, a build root whose only
  outgoing edges are delegates now returns a bare, directly runnable agent
  instead of a `Swarm` or multi-node graph — the team wrapper only ever
  encoded *handoff*. The same `interactions.yaml` yields a different object
  from `build()`. Nothing in `common/` needs changing, but code that reached
  into the returned object may need to.

## 0.0.1 — 2026-09-02

First release. Framework-neutral core (file contracts, all-errors-at-once
validation, LiteLLM-format model strings and aliases, env preflight by name),
six SDK adapters, mermaid interaction diagrams, and the
`validate | render | run | new | import` CLI.
