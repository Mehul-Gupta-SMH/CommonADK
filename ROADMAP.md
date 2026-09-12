# Roadmap

What has shipped, what is next, and what is planned further out. Each open item
links to a GitHub issue — **discussion happens on the issues**, this file is the
map. [`plan.md`](plan.md) holds the original design plan and milestone record;
[`tasks.md`](tasks.md) logs how everything was built.

## Shipped

| Milestone | Feature |
|---|---|
| M1 | Framework-neutral core: pydantic file contracts, all-errors-at-once validation, strict YAML schemas, typed-tool enforcement, LiteLLM model strings + aliases, env preflight by name |
| M1 | Generated interaction diagrams — mermaid rendered from `interactions.yaml`, never hand-drawn |
| M2–M3, M5–M8 | Six adapters: Google ADK, OpenAI Agents, Claude Agent SDK, CrewAI, AutoGen, LangGraph — each documenting its verified constraints and edge-mapping fidelity ([docs/HLD.md](docs/HLD.md)) |
| M4 | CLI: `commonadk validate | render | run | --version` |
| — | Cross-target hypothesis test (one `Project`, six builds), full docs, offline demo |
| #6 | CI: core job, a matrix leg per SDK extra, and a non-blocking all-extras job |
| #7 | **Released on PyPI** — `pip install commonadk`, published by the tag-triggered workflow. v0.0.2 fixes a real packaging bug: the `crewai` extra under-declared CrewAI's native provider packages, so `commonadk[crewai]` raised ImportError on gemini and anthropic models in v0.0.1. See [CHANGELOG.md](CHANGELOG.md) |
| #9 | Mixed-target spawning foundation: `runtime:` honored in-process, native per-runtime islands, cross-runtime edges bridged by plain callables ([design](docs/mixed-target-design.md)) |
| #12 | Broader `model_params` per adapter (per-provider maps where SDKs need them), the `commonadk new <agent>` scaffolding command, and the standing dependency-pin watch — all eight neutral params re-verified against installed SDK source |
| [#22](https://github.com/Mehul-Gupta-SMH/CommonADK/issues/22) | **Execution and telemetry layer** — a real runner for **all six targets**, with normalized per-step events, token/cost meters and observe-only hooks. Usage is reported honestly or not at all: a gap is `null` with `usage_complete: false`, never a `0` that would read as a free call. Claude takes cost from the SDK's own figure rather than a static table; CrewAI reports genuine per-call usage off its event bus ([design](docs/runner-design.md)) |
| [#10](https://github.com/Mehul-Gupta-SMH/CommonADK/issues/10) | **delegate vs handoff honored** on 4 of 6 targets — LangGraph, Google ADK, OpenAI Agents and AutoGen each express sub-call-that-returns separately from transfer-that-does-not. Claude and CrewAI keep the collapsed mapping because neither SDK has any transfer-and-never-return primitive (the remaining #10 items — pipelines, fan-out, loops, shared state — are still open) |
| [#8](https://github.com/Mehul-Gupta-SMH/CommonADK/issues/8) | Verified live runs — **all 6 targets** execute one real turn from the same unmodified agent definition against `examples/live-smoke/common` on `claude-haiku-4-5`, each invoking the tool and returning its answer ([run #6](https://github.com/Mehul-Gupta-SMH/CommonADK/actions/runs/34691962910), captured in [docs/demo-runs.md](docs/demo-runs.md#live-runs)). Run #6 also confirms the telemetry layer against real APIs: five of six targets report genuine tokens and cost, and `openai` — the one SDK that reports no usage for this call — says `?` rather than a confident `0`. Earlier runs exposed two real defects, both fixed: an `autogen-ext`/`anthropic` 1.x incompatibility, and the openai runner reporting unmeasured usage as zero |

## Next up

| Feature | Issue |
|---|---|
| Mixed-target spawning, part two: cross-runtime edges over the wire (A2A), and sourcing an edge from a non-root island member | [#9](https://github.com/Mehul-Gupta-SMH/CommonADK/issues/9) |

## Planned

| Feature | Issue |
|---|---|
| **Richer edge semantics, part two** — sequential pipelines, parallel fan-out/fan-in, loops with exit conditions, shared state (delegate vs handoff already shipped, see above) | [#10](https://github.com/Mehul-Gupta-SMH/CommonADK/issues/10) |
| **Additional adapters** — Semantic Kernel, PydanticAI, Strands, smolagents (help wanted) | [#11](https://github.com/Mehul-Gupta-SMH/CommonADK/issues/11) |
| **Quality backlog** — build-time observability logging around `load()`/`build()` is the one checkbox left (good first issue; the run-time half shipped with #22) | [#12](https://github.com/Mehul-Gupta-SMH/CommonADK/issues/12) |

## Principles that carry forward

- Where a target cannot express a semantic, adapters raise a **clear, specific
  error** — never silent degradation.
- Adapters are written against the **installed** SDK, with verified constraints
  documented in their module docstrings.
- The core stays SDK-free: importing and validating a project never requires an
  agent SDK to be installed.
