# CommonADK — Task Log

All actions taken on this project are logged here. Planning lives in `plan.md`.

## Milestone status

- [x] **M1 — Core**: models, loader, validation, mermaid renderer, example, tests (31 tests passing)
- [x] **M2 — Google ADK adapter** (43 tests passing; entry agent builds full tree)
- [x] **M3 — OpenAI Agents adapter** (hypothesis test PASSED: same `common/` builds on both SDKs; 53 tests)
- [ ] **M4 — CLI & docs**

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
