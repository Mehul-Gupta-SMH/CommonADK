# CommonADK — Plan

## Hypothesis

An agent system can be defined once, in a framework-neutral format, and materialized
automatically into any agent SDK — the way LiteLLM lets you call any LLM through one
interface. The `common/` folder is the single source of truth; per-SDK support is
provided by thin runtime adapters, never by hand-written per-SDK agent code.

**v1 success criterion:** the same `common/` folder builds and runs unmodified on both
Google ADK and OpenAI Agents SDK.

## Decisions (settled)

| Decision | Choice |
|---|---|
| Translation mechanism | **Runtime factory** — no generated agent code; adapters instantiate live SDK objects from `common/` |
| Interaction source of truth | **Structured YAML** (`common/interactions.yaml`); mermaid in `interaction-layer.md` is auto-generated documentation |
| v1 SDK targets | **Google ADK + OpenAI Agents SDK** |
| Model identifiers | **LiteLLM-format strings** everywhere; adapters use the SDK-native path when the model is native, else the SDK's LiteLLM wrapper (`LiteLlm` in Google ADK, `LitellmModel` in OpenAI Agents) |
| Secrets | `agent-config.yaml` declares required env var **names only**, never values; loader validates presence and fails with a precise list |
| Per-SDK folders | Created only when their adapter lands — no empty stubs |
| Edge semantics v1 | Intersection both SDKs express: `delegate` and `handoff`. Pipelines/loops/shared state deferred |
| Process | Planning lives in `plan.md`; every action is logged in `tasks.md`; implementation is delegated to spawned Sonnet 5 subagents, orchestrated and reviewed before commit |

## Architecture

One core package, `commonadk`, with three responsibilities:

1. **Load & validate** — parse `common/` into framework-neutral Pydantic models
   (`AgentSpec`, `ToolSpec`, `InteractionGraph`, `ProjectConfig`). Fail loudly on:
   bad YAML, tools referenced but not defined in `tools.py`, tools missing type hints
   or docstrings, edges naming unknown agents, invalid edge types, missing required
   env vars (at build time), unknown model aliases.
2. **Adapt** — one adapter per SDK behind a common interface.
   `GoogleADKAdapter`: `AgentSpec` → `google.adk.Agent`; `delegate`/`handoff` edges → `sub_agents`.
   `OpenAIAdapter`: `AgentSpec` → `agents.Agent`; edges → `handoffs`.
   Tools: plain typed functions from `tools.py` wrapped into `FunctionTool` / `@function_tool`.
3. **Render** — regenerate the mermaid block in `common/interaction-layer.md` from
   `interactions.yaml` via CLI, so the diagram never drifts from the spec.

### User-facing API

```python
import commonadk

project = commonadk.load("common/")                        # parse + validate
agent = project.build("coordinator", target="google-adk")  # live Google ADK agent
agent = project.build("coordinator", target="openai")      # live OpenAI Agents agent
```

## File contracts

```
common/
├── config.yaml              # project-wide config (below)
├── interactions.yaml        # typed interaction edges (source of truth)
├── interaction-layer.md     # GENERATED mermaid rendering of interactions.yaml
└── <agent-name>/
    ├── skill.md             # instructions/persona → system instructions
    ├── tools.py             # plain typed Python functions with docstrings
    └── agent-config.yaml    # per-agent config (below)
```

### `common/config.yaml`

```yaml
name: my-project
entry: coordinator            # default root agent
targets: [google-adk, openai]
default_model: fast           # alias or LiteLLM string
model_aliases:
  fast: gemini/gemini-2.5-flash
  smart: anthropic/claude-sonnet-5
```

### `common/<agent>/agent-config.yaml`

```yaml
name: researcher
description: Finds and summarizes sources for a topic.

model: gemini/gemini-2.5-pro   # LiteLLM-format string, or an alias from config.yaml
model_params:
  temperature: 0.2
  max_tokens: 4096

tools:                          # function names defined in this agent's tools.py
  - search_web
  - fetch_page

requires:                       # user-supplied runtime prerequisites — NAMES only
  env:
    - name: TAVILY_API_KEY
      description: Search API key used by search_web
      required: true
    - name: POSTGRES_DSN
      description: Connection string for the citations database
      required: false

targets:                        # optional per-SDK overrides (escape hatch)
  google-adk:
    model: gemini-2.5-flash
  openai: {}
```

### `common/<agent>/skill.md`

Markdown instructions passed as the agent's system instructions. Optional YAML
frontmatter reserved for future metadata.

### `common/<agent>/tools.py`

Plain functions. Type hints and docstrings are **required** (enforced at validate
time) — they are what every SDK converts into tool schemas.

### `common/interactions.yaml`

```yaml
entry: coordinator
edges:
  - from: coordinator
    to: researcher
    type: delegate     # v1: delegate | handoff
  - from: researcher
    to: writer
    type: handoff
```

## Repo layout

```
src/commonadk/           # models, loader, validation, mermaid renderer, CLI
src/commonadk/adapters/  # google_adk.py, openai.py (thin, committed)
examples/research-crew/  # demo common/ folder: coordinator, researcher, writer
tests/
plan.md                  # this file — all planning
tasks.md                 # action log
```

## Milestones

- **M1 — Core (no SDKs):** Pydantic models, loader, validation, mermaid renderer,
  example project, tests. Fully testable offline. Deps: pydantic + pyyaml only.
- **M2 — Google ADK adapter:** example runs end-to-end on Google ADK.
- **M3 — OpenAI Agents adapter:** *same example* runs unmodified — the hypothesis test.
- **M4 — CLI & docs:** `commonadk validate | render | run`, README, usage docs.
