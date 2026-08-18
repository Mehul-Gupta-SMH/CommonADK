# CommonADK Docs

CommonADK defines an agent system once, in a framework-neutral `common/`
folder, and materializes it into any supported agent SDK via a thin
per-SDK adapter — the same bet LiteLLM makes for model providers. This
folder documents the design and file formats that make that work; it
complements, rather than replaces, the docs at the repo root.

| Doc | Covers |
|---|---|
| [`HLD.md`](HLD.md) | High-level design: the hypothesis, system architecture, component responsibilities, key design decisions with rationale, and the structural asymmetry between Google ADK's `sub_agents` tree and the OpenAI Agents SDK's `handoffs` graph — the flagship design insight |
| [`LLD.md`](LLD.md) | Low-level design: module-by-module contracts for every file in `src/commonadk/` — models, loader pipeline, validation checks, mermaid rendering, both adapters, the CLI, the full error taxonomy, and what each test file covers |
| [`file-contracts.md`](file-contracts.md) | The authoritative reference for authoring a `common/` folder — every file's field table (name, type, required, default, meaning), validation rules, and a fully annotated example drawn from the shipped project |

Outside this folder:

- [`../README.md`](../README.md) — quickstart: install, load the shipped
  example, `commonadk validate|render|run`.
- [`../plan.md`](../plan.md) — planning and roadmap: the settled design
  decisions, milestones (M1–M4), and what's deliberately deferred (mixed-
  target spawning, richer edge semantics, more adapters).
- [`../tasks.md`](../tasks.md) — the action log: what was built, reviewed,
  and shipped, milestone by milestone.
