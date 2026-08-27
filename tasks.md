# CommonADK — Task Log

All actions taken on this project are logged here. Planning lives in `plan.md`.

## Milestone status

- [x] **M1 — Core**: models, loader, validation, mermaid renderer, example, tests (31 tests passing)
- [x] **M2 — Google ADK adapter** (43 tests passing; entry agent builds full tree)
- [x] **M3 — OpenAI Agents adapter** (hypothesis test PASSED: same `common/` builds on both SDKs; 53 tests)
- [x] **M4 — CLI & docs** (`commonadk validate|render|run` + README; 64 tests passing)

- [x] **M5 — Claude Agent SDK adapter** (79 tests; hypothesis test now spans google-adk/openai/claude)
- [x] **M6 — CrewAI adapter** (92 tests; hypothesis test spans 4 targets)
- [x] **M7 — AutoGen adapter** (107 tests; hypothesis test spans 5 targets)
- [x] **M8 — LangGraph adapter** (124 tests; hypothesis test spans all 6 targets)

## Action log

| Date | Actor | Action |
|---|---|---|
| 2026-08-18 | Orchestrator | Planning session: hypothesis, mechanism (runtime factory), interaction source of truth (YAML → generated mermaid), v1 targets (Google ADK + OpenAI Agents), LiteLLM model layer, `requires.env` contract settled |
| 2026-08-18 | Orchestrator | Wrote `plan.md` and `tasks.md`; committed to `claude/project-planning-hypothesis-qopc3j` |
| 2026-08-18 | Orchestrator | Spawned Sonnet 5 subagent to implement M1 |
| 2026-08-18 | Orchestrator | Roadmap added to `plan.md`: mixed-target spawning (per-agent runtime pinning, future); v1 stays single-target per build; reserved `runtime:` key in agent-config schema |
| 2026-08-18 | Sonnet 5 subagent | Implemented M1: `commonadk` package (models, loader, validation, mermaid renderer), `examples/research-crew`, 28 tests passing |
| 2026-08-18 | Orchestrator | Reviewed M1: approved core; flagged missing reserved `runtime:` key and silent unknown-YAML-key acceptance |
| 2026-08-18 | Sonnet 5 subagent | Follow-up: added `runtime:` reservation warning and `extra="forbid"` on YAML-facing models; 31 tests passing |
| 2026-08-18 | Orchestrator | Independently reran tests + smoke-tested load/alias-resolution/check_env; committed and pushed M1 |
| 2026-08-18 | Sonnet 5 subagent | Implemented M2: adapter registry (lazy imports), GoogleADKAdapter (native gemini vs LiteLlm routing, env preflight, multi-parent/cycle detection), `Project.build()`, 12 new tests; added `litellm` to google extra (ADK's LiteLlm wrapper hard-requires it) |
| 2026-08-18 | Orchestrator | Reviewed M2: adapter approved; flagged example graph (writer had two parents — unbuildable on Google ADK, would break M3 hypothesis test) |
| 2026-08-18 | Sonnet 5 subagent | Follow-up: example reshaped to clean tree (coordinator→researcher→writer), interaction-layer.md regenerated, multi-parent/cycle coverage moved to fixtures |
| 2026-08-18 | Orchestrator | Independently reran suite (43 passing) + smoke-built coordinator tree on google-adk; committed and pushed M2 |
| 2026-08-18 | Sonnet 5 subagent | Implemented M3: OpenAIAgentsAdapter (native openai vs LitellmModel routing, memoized shared instances for multi-parent graphs, cycle-safe two-pass wiring), env preflight hoisted to BaseAdapter, 10 new tests incl. `test_same_project_builds_on_both_targets` |
| 2026-08-18 | Orchestrator | Reviewed M3: approved; independently reran suite (53 passing) and live-verified the hypothesis — one Project builds coordinator on google-adk (sub_agents tree) and openai (handoff graph) from the same `common/`; committed and pushed |
| 2026-08-18 | Sonnet 5 subagent | Implemented M4: argparse CLI (`validate`/`render`/`run`/`--version`, lazy SDK imports, clean error paths, warnings surfaced), console script, 11 CLI tests, README rewritten from stub into full docs |
| 2026-08-18 | Orchestrator | Reviewed M4: approved; independently reran suite (64 passing) and exercised the CLI (validate summary, unknown-target error, version); committed and pushed — v1 plan (M1–M4) complete |
| 2026-08-18 | Sonnet 5 subagent | Wrote docs/: README (index), HLD.md, LLD.md, file-contracts.md — all cross-checked against source (field tables from pydantic models, error strings matched verbatim, examples copied from shipped project) |
| 2026-08-18 | Orchestrator | Reviewed docs; fixed stale "once the CLI lands" note in mermaid.py's generated header (flagged by docs review) and regenerated example interaction-layer.md via `commonadk render`; committed and pushed |
| 2026-08-18 | Sonnet 5 subagent | Implemented M5: ClaudeAgentSDKAdapter (ClaudeAgentOptions with flat subagent registry + Agent-tool edges, per-agent in-process MCP tool servers, anthropic-only model routing with clear error), CLI claude branch, example claude overrides, hypothesis test parametrized over 3 targets |
| 2026-08-18 | Orchestrator | Reviewed M5: approved; verified subagent tool isolation, Agent-tool granted only to agents with outgoing edges, non-anthropic error path; 79 tests passing; committed and pushed |
| 2026-08-19 | Sonnet 5 subagent | Implemented M6: CrewAIAdapter (always returns a Crew — hierarchical with root as manager_agent when reachable agents exist, solo sequential for leaves; LiteLLM-native model pass-through with temperature/max_tokens; crew-wide delegation coarsening documented; manager tools dropped with warning per CrewAI's own constraint), CLI crewai branch, 12 new tests |
| 2026-08-19 | Orchestrator | Reviewed M6: approved; verified crew shapes, delegation flags, model routing live; noted pip dependency tension (crewai pins openai<3, openai-agents prefers >=3 — resolved to openai 2.54.0, all 92 tests still pass); committed and pushed |
| 2026-08-19 | Sonnet 5 subagent | Implemented M7: AutoGenAdapter (Swarm of reachable agents with root as initial speaker, leaf builds return bare AssistantAgent; openai/anthropic/gemini model clients with explicit model_info to dodge autogen-ext's broken fuzzy model tables; string-name handoffs make multi-parent/cycles trivial), CLI autogen branch, 14 new tests |
| 2026-08-19 | Orchestrator | Reviewed M7: approved; verified Swarm/leaf shapes and tool wiring live; example builds unmodified on autogen (no overrides needed); noted protobuf downgrade by autogen-core (suite unaffected); committed and pushed |
| 2026-08-19 | Sonnet 5 subagent | Implemented M8: LangGraphAdapter (compiled StateGraph of react-agent nodes with per-edge transfer_to_<agent> handoff tools via Command(goto=...) — the only target honoring edge targets precisely; init_chat_model routing for gemini/openai/anthropic; used create_agent, not the deprecated create_react_agent), CLI langgraph branch, 16 new tests |
| 2026-08-19 | Orchestrator | Reviewed M8: approved; verified graph nodes and leaf shape live; example builds unmodified; no new pip pins needed; committed and pushed — v2 adapter expansion (M5–M8) complete, 6 targets total |
| 2026-08-19 | Sonnet 5 subagent | Refreshed docs/ for all six targets: HLD comparison table + edge-fidelity spectrum as core finding, LLD subsections for the four new adapters + updated registry/CLI/error-taxonomy/testing sections, file-contracts per-target override forms, index updated; also fixed stale captions and milestone refs |
| 2026-08-19 | Orchestrator | Reviewed docs refresh: approved; corrected one inaccuracy (LLD quoted the pre-M4-fix mermaid header text as current); committed and pushed; updated PR #2 description to six-target scope |
