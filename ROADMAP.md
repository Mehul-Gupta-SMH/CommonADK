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
| — | Cross-target hypothesis test (one `Project`, six builds), 126 offline tests, full docs, offline demo |

## Next up

| Feature | Issue |
|---|---|
| CI: run the suite on push/PR — minimal core-only CI shipped; per-extra matrix and all-extras still open | [#6](https://github.com/Mehul-Gupta-SMH/CommonADK/issues/6) |
| Publish to PyPI (tag-triggered, trusted publishing) — workflow shipped, awaiting first tag | [#7](https://github.com/Mehul-Gupta-SMH/CommonADK/issues/7) |
| Verified live runs: one real LLM turn per target, secrets-gated | [#8](https://github.com/Mehul-Gupta-SMH/CommonADK/issues/8) |

## Planned

| Feature | Issue |
|---|---|
| **Mixed-target spawning** — pin agents to different SDKs in one project (`runtime:` key, A2A for cross-runtime edges) | [#9](https://github.com/Mehul-Gupta-SMH/CommonADK/issues/9) |
| **Richer edge semantics** — honor delegate vs handoff where expressible; pipelines, parallel fan-out, loops, shared state | [#10](https://github.com/Mehul-Gupta-SMH/CommonADK/issues/10) |
| **Additional adapters** — Semantic Kernel, PydanticAI, Strands, smolagents (help wanted) | [#11](https://github.com/Mehul-Gupta-SMH/CommonADK/issues/11) |
| **Quality backlog** — broader `model_params`, `commonadk new` scaffolding, observability hooks, dependency-pin watch (good first issues) | [#12](https://github.com/Mehul-Gupta-SMH/CommonADK/issues/12) |

## Principles that carry forward

- Where a target cannot express a semantic, adapters raise a **clear, specific
  error** — never silent degradation.
- Adapters are written against the **installed** SDK, with verified constraints
  documented in their module docstrings.
- The core stays SDK-free: importing and validating a project never requires an
  agent SDK to be installed.
