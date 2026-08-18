# CommonADK — Task Log

All actions taken on this project are logged here. Planning lives in `plan.md`.

## Milestone status

- [x] **M1 — Core**: models, loader, validation, mermaid renderer, example, tests (31 tests passing)
- [ ] **M2 — Google ADK adapter**
- [ ] **M3 — OpenAI Agents adapter** (hypothesis test: same `common/` runs on both)
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
