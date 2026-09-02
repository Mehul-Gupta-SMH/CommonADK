# CommonADK Docs

CommonADK defines an agent system once, in a framework-neutral `common/`
folder, and materializes it into any supported agent SDK via a thin
per-SDK adapter — the same bet LiteLLM makes for model providers. This
folder documents the design and file formats that make that work; it
complements, rather than replaces, the docs at the repo root.

| Doc | Covers |
|---|---|
| [`HLD.md`](HLD.md) | High-level design: the hypothesis, system architecture, component responsibilities, key design decisions with rationale, and a comparison of all six supported targets — the spectrum of edge-semantics fidelity from LangGraph's precise per-edge handoff tools down to CrewAI's crew-wide delegation — the flagship design insight |
| [`LLD.md`](LLD.md) | Low-level design: module-by-module contracts for every file in `src/commonadk/` — models, loader pipeline, validation checks, mermaid rendering, all six adapters, the CLI, the full error taxonomy, and what each test file covers |
| [`file-contracts.md`](file-contracts.md) | The authoritative reference for authoring a `common/` folder — every file's field table (name, type, required, default, meaning), validation rules, per-target `model` override forms, and a fully annotated example drawn from the shipped project |
| [`demo-runs.md`](demo-runs.md) | Demo runs: real, captured (not fabricated) output from `commonadk --version`, `validate`, `render`, [`examples/demo.py`](../examples/demo.py) (all six targets built offline), and the clean missing-env error — plus which env vars a live `commonadk run` needs per target |
| [`mixed-target-design.md`](mixed-target-design.md) | Mixed-target spawning: pinning individual agents to different SDKs (`runtime:`) within one in-process build — the three-layer model (per-agent build → per-runtime "island" → coordinator), the island-computation algorithm, the verified supported/unsupported cross-runtime source targets, failure modes, and exactly where a future networked/A2A path would attach |

Outside this folder:

- [`../README.md`](../README.md) — quickstart: install, load the shipped
  example, `commonadk validate|render|run` against any of the six
  supported targets.
- [`../plan.md`](../plan.md) — planning and roadmap: the settled design
  decisions, milestones (M1–M8, covering the six adapters), and what's
  deliberately deferred (mixed-target spawning, richer edge semantics).
- [`../tasks.md`](../tasks.md) — the action log: what was built, reviewed,
  and shipped, milestone by milestone.
