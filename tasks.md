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
| 2026-08-27 | Orchestrator | Branch restarted from merged main (PRs #2 and #3 both merged); pushed restart point |
| 2026-08-27 | Sonnet 5 subagent | Docs audit (fixed stale two-SDKs wording; added multi-extra install note with verified pip tensions) + demo runs: examples/demo.py (offline, builds all six targets, labeled error demonstrations), docs/demo-runs.md (real captured output only), tests/test_demo.py; 126 tests passing |
| 2026-08-27 | Orchestrator | Reviewed: reran suite (126) and demo (exit 0, all six targets OK); committed, pushed, opened PR |
| 2026-08-27 | Sonnet 5 subagent | Added repo visuals: assets/banner.png (1280x640, GitHub social-preview size) + assets/demo.gif (14-frame terminal replay of real captured validate/demo output), both with reproducible Pillow generator scripts; README banner + Demo section |
| 2026-08-27 | Orchestrator | Reviewed visuals (inspected banner and final GIF frame; verified transcript matches real demo output); committed, pushed, opened PR |
| 2026-08-30 | Orchestrator | Opened roadmap issues #6-#12 (CI, PyPI release, live runs, mixed-target spawning, richer edge semantics, new adapters, quality backlog); added ROADMAP.md linking them; README roadmap section now points at ROADMAP.md; opened PR |
| 2026-08-30 | Sonnet 5 subagent | v0.0.1 release prep: version bump, real build (sdist+wheel, twine check PASSED, fresh-venv core-only install verified), publish.yml (tag-triggered PyPI Trusted Publishing with tag/version guard), minimal ci.yml (core-only tests on push/PR), README/ROADMAP/demo-runs updates |
| 2026-08-30 | Orchestrator | Reviewed release setup: workflows verified, suite reran green (126); committed, pushed, opened PR; tag v0.0.1 to be pushed after merge + PyPI pending-publisher registration |
| 2026-08-30 | Orchestrator | Tag push blocked by branch-scoped git credentials (403); added workflow_dispatch path to publish.yml (publishes pyproject version, then tags from inside Actions); merged and triggered the release run |
| 2026-09-02 | Sonnet 5 subagent | Full CI matrix (core + per-extra matrix + non-blocking all-extras, concurrency group); broader model_params per adapter verified against installed SDKs (incl. per-provider maps for autogen/langgraph where clients silently drop or reroute unmapped keys); `commonadk new <agent>` scaffolding with optional edge wiring; docs updated; 165 tests |
| 2026-09-02 | Orchestrator | Reviewed: verified CI yaml structure, reran suite (165), smoke-tested `commonadk new` end-to-end (scaffold -> validate exit 0, edge appended, overwrite refused); committed and pushed |
| 2026-09-02 | Sonnet 5 subagent | M9 foundation: mixed-target spawning in-process — `runtime:` now honored (real validation errors, no longer a warning), island computation (union-find over same-runtime edges, root must reach all members), each island built natively by its own adapter, cross-runtime edges bridged by plain callables; design doc, mixed-crew example, 17 tests |
| 2026-09-02 | Orchestrator | Reviewed M9 in its worktree (143 passing there; earlier failures were only a stale editable install pointing at the main tree), patched into the branch alongside the CI/model_params work, resolved cleanly, full suite 182 passing |
