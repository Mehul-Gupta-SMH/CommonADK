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
| #7 | **Released on PyPI** — `pip install commonadk` (v0.0.1), published by the tag-triggered workflow |
| #9 | Mixed-target spawning foundation: `runtime:` honored in-process, native per-runtime islands, cross-runtime edges bridged by plain callables ([design](docs/mixed-target-design.md)) |
| #12 | Broader `model_params` per adapter (per-provider maps where SDKs need them) and the `commonadk new <agent>` scaffolding command |

## Next up

| Feature | Issue |
|---|---|
| Verified live runs: one real LLM turn per target, secrets-gated | [#8](https://github.com/Mehul-Gupta-SMH/CommonADK/issues/8) |
| Mixed-target spawning, part two: cross-runtime edges over the wire (A2A), and sourcing an edge from a non-root island member | [#9](https://github.com/Mehul-Gupta-SMH/CommonADK/issues/9) |

## Planned

| Feature | Issue |
|---|---|
| **Richer edge semantics** — honor delegate vs handoff where expressible; pipelines, parallel fan-out, loops, shared state | [#10](https://github.com/Mehul-Gupta-SMH/CommonADK/issues/10) |
| **Additional adapters** — Semantic Kernel, PydanticAI, Strands, smolagents (help wanted) | [#11](https://github.com/Mehul-Gupta-SMH/CommonADK/issues/11) |
| **Quality backlog** — observability hooks and the dependency-pin watch remain (good first issues) | [#12](https://github.com/Mehul-Gupta-SMH/CommonADK/issues/12) |

## Principles that carry forward

- Where a target cannot express a semantic, adapters raise a **clear, specific
  error** — never silent degradation.
- Adapters are written against the **installed** SDK, with verified constraints
  documented in their module docstrings.
- The core stays SDK-free: importing and validating a project never requires an
  agent SDK to be installed.
