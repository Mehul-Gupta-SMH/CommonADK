# Changelog

All notable changes to CommonADK are recorded here. Versions follow
[semantic versioning](https://semver.org/); while the project is pre-1.0,
breaking changes can land in a patch release and are called out explicitly.

## 0.0.3 — 2026-10-12

### Added

- **`LLMCall.cache_read_tokens` and `LLMCall.cache_write_tokens`**, summed
  into `Trace.rollup()["llm_calls"]` alongside the existing counts. Kept
  separate from `prompt_tokens` rather than folded in, because a cache read
  bills at a fraction of the input rate and a cache write at a premium: one
  "input" number cannot carry three rates, and folding them in would fix the
  magnitude and break the pricing.
- The rollup's `note` now states that `prompt_tokens` is the **uncached**
  input whenever cache tokens are present, appended rather than replacing
  the usage/cost completeness notes — cached input is true independently of
  those.

### Fixed

- **The Claude Agent SDK runner dropped prompt-cache token counts, so
  `prompt_tokens` was the uncached input alone.** `ModelUsage` reports four
  input-side counts — `inputTokens`, `cacheReadInputTokens`,
  `cacheCreationInputTokens`, `outputTokens` — and only the first and last
  were summed. For a multi-turn agent run, where every turn after the first
  reads the prefix from cache, that leaves `prompt_tokens` a small and
  nearly constant fraction of the input while `total_cost_usd` bills all of
  it. A consumer measured ten runs across four task statements and four
  tool sets: four distinct `prompt_tokens` values, three of them identical,
  and an SDK cost roughly twice what those tokens price to.

  No completeness flag caught it and none could. `usage_complete` is
  `count == 0 or len(reported) == count` — "every call I *saw* reported
  usage", not "I saw every call" — and every call here *had* reported
  usage. It had just not reported all of the input.

  `LLMCall` now carries `cache_read_tokens` and `cache_write_tokens`, the
  rollup sums them, and the rollup's `note` states that `prompt_tokens` is
  the uncached portion whenever cache tokens are present. They are kept
  separate from `prompt_tokens` rather than folded in because the three are
  billed at different rates, and one number cannot carry three rates.

  A runner that cannot report cache usage leaves both `None` — the
  None-vs-0 rule. The regression test that should have caught this had both
  cache fields pinned at `0`, where a dropped zero is indistinguishable
  from a carried one; it now uses non-zero counts.

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

  Verified against real APIs, not only stubs: [live run
  #6](https://github.com/Mehul-Gupta-SMH/CommonADK/actions/runs/34691962910)
  ran all six targets on `claude-haiku-4-5`, five reporting genuine tokens
  and cost, and `openai` — the one SDK reporting no usage for that call —
  reporting `?` rather than `0`. See [docs/demo-runs.md](docs/demo-runs.md).

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
