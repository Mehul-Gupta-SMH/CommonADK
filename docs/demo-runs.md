# Demo runs

Every command output on this page was actually run in this environment and
pasted verbatim (trimmed for readability where noted — elisions are marked
`# ...`) — nothing here is a fabricated or hand-written transcript. This
complements [`file-contracts.md`](file-contracts.md) and
[`HLD.md`](HLD.md): those explain the shapes; this page shows the real thing
running. See also [`examples/demo.py`](../examples/demo.py), the script
these captures come from (sections 4 and 5) or that reproduces the same
`commonadk` CLI commands (sections 1–3).

No API keys exist in this environment, and none of the captures below used
one — every `commonadk validate`/`render` and every `project.build(...)`
call is pure, local, offline construction (see "Running for real" below for
what a live `commonadk run` actually needs, per target, and ["Live
runs"](#live-runs) at the very bottom for the one workflow in this repo that
actually spends money on a real model call, and its honesty rule: no output
is pasted below until that workflow has actually been run).

## `commonadk --version`

```
$ commonadk --version
commonadk 0.0.1
```

## `commonadk validate examples/research-crew/common`

Full, unedited output:

```
$ commonadk validate examples/research-crew/common
Project: research-crew  (entry agent: coordinator)

Agents:
  coordinator
    model: fast -> gemini/gemini-2.5-flash
    tools: format_handoff_note, split_into_subtopics
    env: (none required)
  researcher
    model: gemini/gemini-2.5-pro -> gemini/gemini-2.5-pro
    tools: fetch_page, search_web
    env:
      TAVILY_API_KEY: not set (required) -- Search API key used by search_web
      POSTGRES_DSN: not set (optional) -- Connection string for the citations database
  writer
    model: fast -> gemini/gemini-2.5-flash
    tools: count_words, format_as_markdown
    env: (none required)
```

## `commonadk render examples/research-crew/common`

```
$ commonadk render examples/research-crew/common
Wrote examples/research-crew/common/interaction-layer.md
```

The file it (re)writes — `examples/research-crew/common/interaction-layer.md`
— already matched this exact output before the run (that's what
`test_example_interaction_layer_matches_current_graph` guards), so this
command was a no-op rewrite here:

```mermaid
flowchart TD
    coordinator(["coordinator (entry)"])
    researcher["researcher"]
    writer["writer"]
    coordinator -- delegate --> researcher
    researcher -. handoff .-> writer
```

## `python3 examples/demo.py`

The interesting sections, captured from a real run in this environment with
**no** env vars pre-set (the script self-provisions every placeholder it
needs — see `tests/test_demo.py`, which asserts exactly this by stripping
those vars before running it as a subprocess). Full output is ~75 lines;
elided parts are marked.

**Section 1 — project summary** (agents, resolved models, env requirements):

```
==============================================================================
1. Load and validate examples/research-crew/common
==============================================================================
Project: research-crew  (entry agent: 'coordinator')
Model aliases: {'fast': 'gemini/gemini-2.5-flash', 'smart': 'anthropic/claude-sonnet-5'}

Agents:
  - coordinator
      model: fast -> gemini/gemini-2.5-flash
      tools: format_handoff_note, split_into_subtopics
      env requirements: (none)
  - researcher
      model: gemini/gemini-2.5-pro -> gemini/gemini-2.5-pro
      tools: fetch_page, search_web
      env requirement: TAVILY_API_KEY (required, currently not set) -- Search API key used by search_web
      env requirement: POSTGRES_DSN (optional, currently not set) -- Connection string for the citations database
  - writer
      model: fast -> gemini/gemini-2.5-flash
      tools: count_words, format_as_markdown
      env requirements: (none)
```

**Section 4 — building `coordinator` for all six targets** (this is the
core proof: the same `common/` folder, unmodified, building on every
supported SDK):

```
==============================================================================
4. Build 'coordinator' for all six supported targets
==============================================================================
[google-adk] OK -- google.adk.agents.llm_agent.LlmAgent
    shape (sub_agents tree): root='coordinator', sub_agents=['researcher']
    (2 expected warning(s) suppressed -- per-adapter model_params/tool quirks documented in adapters/google_adk*.py)
[openai] OK -- agents.agent.Agent
    shape (handoff graph): root='coordinator', handoffs=['researcher']
[claude] OK -- claude_agent_sdk.types.ClaudeAgentOptions
    shape (options subagents (flat registry)): root has no name field (session/query-based); options.agents=['researcher', 'writer']
    (4 expected warning(s) suppressed -- per-adapter model_params/tool quirks documented in adapters/claude*.py)
[crewai] OK -- crewai.crew.Crew
    shape (crew members): process=hierarchical, manager='coordinator', members=['researcher', 'writer']
    (11 expected warning(s) suppressed -- per-adapter model_params/tool quirks documented in adapters/crewai*.py)
  (setting placeholder OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY -- these are short, obviously-fake strings, used ONLY to satisfy the autogen/langgraph model clients' eager offline construction check; see those adapters' module docstrings, 'Offline construction'. No network call is ever made and no real key is required for build().)
[autogen] OK -- autogen_agentchat.teams._group_chat._swarm_group_chat.Swarm
    shape (swarm participants): Swarm participants=['coordinator', 'researcher', 'writer']
[langgraph] OK -- langgraph.graph.state.CompiledStateGraph
    shape (graph nodes): StateGraph nodes=['coordinator', 'researcher', 'writer']
```

Note the shape of each `build()` return value lines up exactly with
[`HLD.md`'s "Comparing the six targets"](HLD.md#comparing-the-six-targets):
Google ADK's tree (`sub_agents=['researcher']` — `writer` is one level
deeper, under `researcher`), OpenAI Agents' handoff reference list, Claude's
flat `options.agents` registry, CrewAI's hierarchical crew (manager +
members), AutoGen's `Swarm` participant list, and LangGraph's node-keyed
`StateGraph`.

**Section 5 — the two demonstrated failure modes**:

```
==============================================================================
5. Demonstrated failure modes (on purpose)
==============================================================================
DEMONSTRATION 1 of 2: building with a required env var unset.
researcher/agent-config.yaml declares TAVILY_API_KEY as required.
Temporarily unsetting it and attempting project.build(..., target="openai") on purpose:
  Raised OSError as expected:
    commonadk: missing required environment variable(s) for target 'openai' (building 'coordinator'):
      - researcher: TAVILY_API_KEY (Search API key used by search_web)

DEMONSTRATION 2 of 2: building for an unrecognized target string.
Attempting project.build(..., target="not-a-real-sdk") on purpose:
  Raised ValueError as expected: Unknown build target 'not-a-real-sdk'. Known targets: ['autogen', 'claude', 'crewai', 'google-adk', 'langgraph', 'openai']

==============================================================================
Done -- exiting 0
==============================================================================
```

`python3 examples/demo.py` exits `0` — captured directly (`echo $?` after
the run above printed `0`).

## `commonadk run` — the clean missing-env error

This is the same failure mode as demo.py's Demonstration 1, but through the
actual CLI entry point rather than a direct `project.build()` call — real,
captured output, `TAVILY_API_KEY` genuinely unset in the shell:

```
$ unset TAVILY_API_KEY
$ commonadk run examples/research-crew/common --target openai --agent researcher "test prompt"
commonadk: missing required environment variable(s) for target 'openai' (building 'researcher'):
  - researcher: TAVILY_API_KEY (Search API key used by search_web)
$ echo $?
1
```

And the unknown-target error, also real and captured:

```
$ commonadk run examples/research-crew/common --target nope "test prompt"
commonadk: Unknown build target 'nope'. Known targets: ['autogen', 'claude', 'crewai', 'google-adk', 'langgraph', 'openai']
$ echo $?
1
```

Both preflights run **before** any SDK object is touched — `run` never gets
as far as importing `openai-agents` or evaluating the `--target` string
against a live adapter in either case (see
[`LLD.md`'s error taxonomy](LLD.md#error-taxonomy)).

## Running for real

Everything above is offline construction — no LLM was ever called. Actually
running `commonadk run <common-dir> --target <target> "<prompt>"` against a
live model additionally needs, **on top of** whatever `requires.env`
declares per-agent (`TAVILY_API_KEY` for `researcher`, in the shipped
example, on every target — that's a tool credential, not a model-provider
one, and every target needs it identically):

**None of the commands below were run** — this environment has no API keys,
and the task rules for this pass forbid fabricating LLM output. The env
vars and commands are derived directly from each adapter's own model-routing
code (`src/commonadk/adapters/*.py`) and the underlying SDK/`litellm`
conventions each adapter routes through, not guessed:

| Target | Env var(s) needed for the shipped example's models | Why (source) | Example command |
|---|---|---|---|
| `google-adk` | `GOOGLE_API_KEY` or `GEMINI_API_KEY` | The example's models resolve to `gemini/...`, which `google_adk.py`'s `_model_for` passes through as a **bare native id** — ADK's own Gemini model client reads the key from the environment | `export TAVILY_API_KEY=... GEMINI_API_KEY=...`<br>`commonadk run examples/research-crew/common --target google-adk "Research electric vehicle adoption"` |
| `openai` | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) — because the example's models are `gemini/...`, not `openai/...` | Non-`openai/...` models are wrapped in `agents.extensions.models.litellm_model.LitellmModel` (`openai_agents.py`'s `_model_for`), which routes through `litellm` — `litellm` reads the provider-appropriate key itself (`GEMINI_API_KEY`/`GOOGLE_API_KEY` for a `gemini/...` model string) | `export TAVILY_API_KEY=... GEMINI_API_KEY=...`<br>`commonadk run examples/research-crew/common --target openai "Research electric vehicle adoption"` |
| `claude` | `ANTHROPIC_API_KEY` | The Claude Agent SDK's bundled CLI needs it to authenticate — `cli.py`'s `_run_claude` preflights this itself, since (per `claude_agent.py`'s module docstring) nothing in the SDK declares or checks for it the way `requires.env` does; also required for the SDK's own model calls once running (the shipped example's per-agent `targets.claude.model: claude-sonnet-5` overrides are already in place, so no `agent-config.yaml` changes are needed) | `export TAVILY_API_KEY=... ANTHROPIC_API_KEY=...`<br>`commonadk run examples/research-crew/common --target claude "Research electric vehicle adoption"` |
| `crewai` | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | `crewai.LLM(model="gemini/...")` (`crewai_adapter.py`'s `_llm_for`) parses the LiteLLM-format string itself and routes `gemini` to a native client that reads the same key | `export TAVILY_API_KEY=... GEMINI_API_KEY=...`<br>`commonadk run examples/research-crew/common --target crewai "Research electric vehicle adoption"` |
| `autogen` | `GEMINI_API_KEY` | `autogen_adapter.py`'s `_client_for` routes `gemini/...` through `OpenAIChatCompletionClient`, whose own `__init__` special-cases a `"gemini-"`-prefixed model name and reads `GEMINI_API_KEY` from the environment when no `api_key` kwarg is given (this is also the var `build()` itself needs just to *construct* the client — see "Offline construction" in that adapter's module docstring, and the placeholder value `examples/demo.py` sets for exactly this reason) | `export TAVILY_API_KEY=... GEMINI_API_KEY=...`<br>`commonadk run examples/research-crew/common --target autogen "Research electric vehicle adoption"` |
| `langgraph` | `GOOGLE_API_KEY` or `GEMINI_API_KEY` | `langgraph_adapter.py`'s `_model_for` routes `gemini/...` through `init_chat_model("google_genai:...")`, whose `ChatGoogleGenerativeAI` construction is eager and raises immediately without one of these two (same "Offline construction" note as `autogen`, and the same reason `examples/demo.py` sets a placeholder) | `export TAVILY_API_KEY=... GOOGLE_API_KEY=...`<br>`commonadk run examples/research-crew/common --target langgraph "Research electric vehicle adoption"` |

If you swap any agent's `model:` to an `openai/...` string instead (or add
a `targets.<target>.model` override), the required key changes accordingly
— `OPENAI_API_KEY` for a native or LiteLLM-routed `openai/...` model on any
target. See [`file-contracts.md`'s per-target override table](file-contracts.md#targets--per-target-overrides)
for exactly what form each target's override expects.

## Live runs

Issue [#8](https://github.com/Mehul-Gupta-SMH/CommonADK/issues/8), "Verified
live runs" — everything above this section is offline construction; this is
the one piece of the project that actually calls a real model.
[`examples/live-smoke/common`](../examples/live-smoke/common) is a
minimal project built for exactly this: one agent (`assistant`), one
trivial, deterministic, no-network tool (`count_words`), and a
`default_model` of `anthropic/claude-haiku-4-5` — the cheapest current
Anthropic model — with **no** per-target `targets.<sdk>.model` override
anywhere, because every one of the six adapters has a verified Anthropic
path (see each adapter's own `_model_for`/`_client_for`/`_llm_for`, and the
table in [`scripts/live_smoke.py`](../scripts/live_smoke.py)'s own module
docstring): native for `claude`; `LiteLlm(model="anthropic/...")` for
`google-adk`; `LitellmModel(model="anthropic/...")` for `openai`;
`crewai.LLM(model="anthropic/...")`'s own provider routing for `crewai`;
`AnthropicChatCompletionClient` for `autogen`; `init_chat_model("anthropic:...")`
for `langgraph`. One `ANTHROPIC_API_KEY` genuinely drives all six.

[`scripts/live_smoke.py`](../scripts/live_smoke.py) runs one real turn per
target, records the outcome (success/error), wall time, and a truncated
final answer, and — for `google-adk`/`openai`, the two targets
`commonadk.runners` (issue #22) has a runner for — the full normalized
trace, with token counts and cost, honest about any gap
(`usage_complete`/`cost_complete` rather than a silently-summed partial
total). The other four targets (`claude`, `crewai`, `autogen`, `langgraph`)
have no runner yet, so their report entries carry `"usage": "unavailable"`
explicitly — never `0`, which would misleadingly claim the call was free.
It supports `--list` (print the target table and exit, no key needed),
`--dry-run` (build every target's entry agent but never call the model, no
key needed — this is what `tests/test_live_smoke.py` exercises offline),
`--targets` (a comma-separated subset), and `--model` (`claude-haiku-4-5`
by default; `claude-sonnet-5`/`claude-opus-5` are valid, more expensive
opt-ins for a heavier check — never a date-suffixed model id). It fails
loudly and immediately — before touching any target, before writing any
file — if `ANTHROPIC_API_KEY` isn't set, rather than hanging on the first
real SDK call or writing a misleadingly-empty report.

**Triggering a run.** [`.github/workflows/live-runs.yml`](../.github/workflows/live-runs.yml)
is `workflow_dispatch`-only — it never fires on `push` or `pull_request`,
because unlike every other workflow in this repo (`ci.yml`, `publish.yml`'s
own test/build jobs), this one spends real money. From the repo's Actions
tab, run "Live runs" manually, optionally overriding `targets` (default: all
six) and `model` (default: `claude-haiku-4-5`), or `dry_run: true` to
exercise the workflow itself for free. It maps the repository secret
`CLAUDE_API_KEY` onto the `ANTHROPIC_API_KEY` environment variable every SDK
path actually reads (the secret is deliberately named `CLAUDE_API_KEY`, not
`ANTHROPIC_API_KEY` — the workflow does the mapping so no SDK has to be
told about that naming choice), detects a missing/empty secret in its own
dedicated step rather than a job-level `if:` against `secrets.*` (the same
detect-in-a-step pattern `publish.yml` uses for `PYPI_API_TOKEN`, adopted
here for the same reason: a job-level `env:` condition on a secret silently
evaluated false there), installs all six SDK extras, runs the script,
writes the summary table to the job's `$GITHUB_STEP_SUMMARY`, and uploads
the JSON report plus every per-target trace file as a build artifact.

**Live run #5 — 2026-09-10: all six targets green.** [GitHub Actions run
#5](https://github.com/Mehul-Gupta-SMH/CommonADK/actions/runs/34541706890),
`workflow_dispatch` on `main` at commit `b16990e`, model
`claude-haiku-4-5`, all six targets, `dry_run: false`. This is the run
taken after the two fixes below landed, and it is the current reference
result:

```
target       status       wall_s   tokens   cost_usd  final_text
----------------------------------------------------------------
google-adk   success        9.74     1716   0.002012  That text has 9 words.
openai       success        2.42        ?          ?  That text has 9 words.
claude       success        3.21        -          -  That text has 9 words.
crewai       success        6.76        -          -  That text has 9 words.
autogen      success        1.26        -          -  9
langgraph    success        1.54        -          -  That text has 9 words.
```

**Six of six SDKs executed a real turn from one unmodified agent
definition** — each called `claude-haiku-4-5`, invoked the `count_words`
tool, and returned the tool's answer. The project's hypothesis, verified
at runtime on every supported target.

Three things in that table are worth reading carefully:

- **`autogen` now succeeds** (1.26s) where run #4 failed before reaching
  the API. The `anthropic<1` pin in the `autogen` extra took effect on a
  fresh CI install, confirming the fix in the environment that broke.
- **`openai` reports `?`, not `0`/`$0.000000`.** That `?` means "the SDK
  did not report usage" — the honesty fix working in production. Run #4
  printed a confident zero for the same call; the number was never real.
  See "the 0-not-None wrinkle" in [`runner-design.md`](runner-design.md).
- **`autogen` answered `9` rather than `"That text has 9 words."`** Same
  tool, same correct result, different presentation — identical
  instructions produce different response shapes across frameworks. Not a
  defect; a property of the frameworks worth knowing about.

Note on timings: `crewai` took 6.76s here against 25.52s in run #4 — the
same work, a ~4x swing between two runs minutes apart. Treat every wall
time on this page as one sample on shared CI hardware, never as a
benchmark.

**Live run #4 — 2026-09-10 (superseded by #5, kept as the record of what the first real run exposed).** [GitHub Actions run
#4](https://github.com/Mehul-Gupta-SMH/CommonADK/actions/runs/34539312259),
triggered by `workflow_dispatch` on `main` at commit `772cab3`, model
`claude-haiku-4-5`, all six targets, `dry_run: false`. Verbatim summary
table from the job:

```
target       status       wall_s   tokens   cost_usd  final_text
----------------------------------------------------------------
google-adk   success       10.02     1716   0.002012  That text has 9 words.
openai       success        2.56        0   0.000000  That text has 9 words.
claude       success        3.54        -          -  That text has 9 words.
crewai       success       25.52        -          -  That text has 9 words.
autogen      error          0.22        -          -  TypeError: AsyncMessages.create() got an unexpected keyword ...
langgraph    success        1.83        -          -  That text has 9 words.

1 of 6 target(s) FAILED:
  - autogen: TypeError: AsyncMessages.create() got an unexpected keyword argument 'temperature'
```

**Five of six SDKs executed a real turn from the same unmodified agent
definition** — each called `claude-haiku-4-5`, invoked the `count_words`
tool on the default 9-word prompt, and returned the identical answer,
`"That text has 9 words."`. This is this project's central hypothesis
demonstrated at runtime, not just at `build()` time.

**`autogen` failed on a verified upstream incompatibility, not a CommonADK
bug.** `autogen-ext` 0.7.5 hard-codes `"temperature":
create_args.get("temperature", 1.0)` into every Anthropic request
(`autogen_ext/models/anthropic/_anthropic_client.py`), and `anthropic` 1.x
removed `temperature` from `messages.create()`. `examples/live-smoke/common`
sets no `model_params` at all, so nothing CommonADK passed caused this
failure — it is a known issue under investigation, being fixed by a
separate change, not yet resolved as of this run.

**`openai`'s `0` tokens / `$0.000000` is not a real measurement — read it as
"not measured," never as "this call was free."** OpenAI Agents' `Usage`
fields default to `0` rather than `None`, and usage isn't populated when the
model runs through the LiteLLM bridge (`LitellmModel`, which is how this
target reaches an `anthropic/...` model). A separate change is in progress
to fix this reporting gap; until then, `openai`'s token/cost columns should
not be trusted.

`google-adk`'s 1,716 tokens / $0.002012 is genuine, captured end-to-end by
the runner/telemetry layer (`commonadk.runners`).

Worth noting as an observation, not a benchmark: wall time ranged from
1.83s (`langgraph`) to 25.52s (`crewai`) for identical work — roughly a 14x
spread. That's one sample, one model, on CI hardware; it says nothing
general about relative SDK performance.

Earlier attempts, briefly, for context: run #1 was a dry run (no model
calls); run #2 failed because the secret was set on the Codespaces tab
rather than the Actions tab; run #3 failed on account credit balance. Run
#4, above, is the first that actually executed.
