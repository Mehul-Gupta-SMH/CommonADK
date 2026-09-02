# CommonADK — Low-Level Design

Audience: a contributor about to modify `src/commonadk/`. For the "why", see
[`HLD.md`](HLD.md). For the `common/` file formats, see
[`file-contracts.md`](file-contracts.md). Line numbers are omitted
deliberately — read the linked source for current line numbers; the
contracts described here (function names, field names, error strings) are
what's stable.

## Package layout

```
src/commonadk/
├── __init__.py        # public re-exports
├── models.py           # Pydantic models: config, spec, and resolution logic
├── loader.py            # common/ -> Project pipeline
├── validation.py         # cross-cutting checks over a loaded-but-unwrapped project
├── mermaid.py             # interactions.yaml -> mermaid rendering
├── cli.py                  # argparse CLI: validate | render | run | new | --version
└── adapters/
    ├── __init__.py         # target -> adapter registry, lazy SDK imports
    ├── base.py               # BaseAdapter ABC + shared env-preflight/BFS
    ├── google_adk.py           # AgentSpec -> google.adk.agents.Agent
    ├── openai_agents.py         # AgentSpec -> agents.Agent
    ├── claude_agent.py           # AgentSpec -> claude_agent_sdk.ClaudeAgentOptions
    ├── crewai_adapter.py          # AgentSpec -> crewai.Crew
    ├── autogen_adapter.py          # AgentSpec -> AssistantAgent / Swarm
    └── langgraph_adapter.py         # AgentSpec -> compiled langgraph StateGraph
```

`__init__.py` re-exports everything a caller needs without reaching into
submodules: `load`, `render_mermaid`, `write_interaction_layer`,
`BaseAdapter`, `get_adapter`, `ValidationError`, and every model class
(`AgentConfig`, `AgentSpec`, `EnvRequirement`, `InteractionEdge`,
`InteractionGraph`, `Project`, `ProjectConfig`, `Requires`, `ToolParameter`,
`ToolSpec`).

## `models.py`

Framework-neutral Pydantic models. Every model populated directly from a
`common/` YAML file sets `model_config = ConfigDict(extra="forbid")` — an
unrecognized key is a load-time error, not a silent no-op (see
[`file-contracts.md`](file-contracts.md) for the full rationale and
examples). `ToolSpec`, `AgentSpec`, and `Project` hold live Python objects
(`Callable`, nested models) rather than raw YAML, so they use
`arbitrary_types_allowed=True` instead.

### `EnvRequirement`

| field | type | required | default | meaning |
|---|---|---|---|---|
| `name` | `str` | yes | — | env var name (never a value) |
| `description` | `str` | no | `""` | human-readable purpose |
| `required` | `bool` | no | `True` | blocks build if missing when `True` |

### `Requires`

| field | type | required | default |
|---|---|---|---|
| `env` | `list[EnvRequirement]` | no | `[]` |

### `ProjectConfig` (parsed `common/config.yaml`)

| field | type | required | default |
|---|---|---|---|
| `name` | `str` | yes | — |
| `entry` | `Optional[str]` | no | `None` |
| `targets` | `list[str]` | no | `[]` |
| `default_model` | `str` | yes | — |
| `model_aliases` | `dict[str, str]` | no | `{}` |

### `AgentConfig` (parsed `common/<agent>/agent-config.yaml`)

| field | type | required | default |
|---|---|---|---|
| `name` | `str` | yes | — |
| `description` | `str` | no | `""` |
| `model` | `Optional[str]` | no | `None` |
| `model_params` | `dict[str, Any]` | no | `{}` |
| `tools` | `list[str]` | no | `[]` |
| `requires` | `Requires` | no | `Requires()` |
| `targets` | `dict[str, dict[str, Any]]` | no | `{}` |
| `runtime` | `Optional[str]` | no | `None` |

`runtime` is reserved for future mixed-target spawning (`plan.md`,
"Deferred / roadmap"). Unset by every shipped agent; if set,
`validation._check_runtime` emits a warning, not an error — v1 still
builds every agent under the single target passed to `project.build()`.

### `ToolParameter` / `ToolSpec`

`ToolParameter`: `name: str`, `type: str`, `required: bool`, `default:
Optional[str] = None` (a `repr()` of the Python default, when one exists).

`ToolSpec`: `name: str`, `description: str = ""`, `func: Callable[...,
Any]` (the live function), `parameters: list[ToolParameter] = []`,
`return_type: Optional[str] = None`, `has_docstring: bool = False`,
`fully_typed: bool = False`.

**`ToolSpec.from_function(func)`** — the introspection contract every tool
goes through:

1. `name = func.__name__`; `description = inspect.getdoc(func) or ""`.
2. Resolves type hints via `inspect.get_annotations(func, eval_str=True)`
   (falls back to `func.__annotations__` if that raises — e.g. a forward
   reference that can't be evaluated).
3. Walks `inspect.signature(func).parameters`, skipping `*args`/`**kwargs`
   (`VAR_POSITIONAL`/`VAR_KEYWORD`). For each remaining parameter without an
   annotation, `type` is set to the literal string `"Any"` and
   `fully_typed` is set `False`; for one with an annotation, `type` is the
   annotation's `__name__` (or its `str()` if it has no `__name__`, e.g. a
   generic like `list[str]`). `required` is `True` unless the parameter has
   a default; `default` is `repr(param.default)` when a default exists,
   else `None`.
4. `return_type` is the return annotation's `__name__` (or its `str()` if
   it has no `__name__`, e.g. `list[str]`), or `None` if unannotated.
5. `has_docstring = bool(description.strip())`.

`fully_typed` and `has_docstring` are exactly what
`validation._check_tools` inspects to decide pass/fail; `return_type is
None` is what triggers its return-hint warning.

### `InteractionEdge` / `InteractionGraph` (parsed `common/interactions.yaml`)

`InteractionEdge`: `from_: str` (YAML alias `"from"`, `populate_by_name=True`
so both `from_=` and `from=`/`**{"from": ...}` construction work), `to: str`,
`type: Literal["delegate", "handoff"]`. An edge type outside that literal
set is a Pydantic parse error surfaced as a `ValidationError` at
`interactions.yaml` load time — `validation._check_edges` never sees an
invalid type, only unknown agent names.

`InteractionGraph`: `entry: Optional[str] = None`, `edges:
list[InteractionEdge] = []`.

### `AgentSpec` / `Project` (resolved, in-memory only — never parsed from YAML directly)

`AgentSpec`: `config: AgentConfig`, `instructions: str = ""` (from
`skill.md`, frontmatter stripped), `tools: list[ToolSpec] = []`; `.name`
is a property that reads `self.config.name`.

`Project`: `config: ProjectConfig`, `agents: dict[str, AgentSpec] = {}`,
`graph: InteractionGraph = InteractionGraph()`.

**`Project.resolve_model(agent_name) -> str`** — resolves an agent's
LiteLLM-format model string:

1. Look up the agent via `_require_agent` (raises `KeyError` — see below —
   if `agent_name` isn't in `self.agents`).
2. `raw = agent.config.model or self.config.default_model` — the agent's
   own `model:` if set, else the project's `default_model`.
3. Delegate to `resolve_model_string(raw)`.

**`Project.resolve_model_string(raw) -> str`** — the `/`-heuristic:

- If `raw` contains `"/"`, it's treated as a literal LiteLLM string
  (`provider/model-id`, e.g. `gemini/gemini-2.5-pro`) and returned
  unchanged.
- Otherwise it's looked up in `self.config.model_aliases`.
- If it's in neither form, raises `ValueError(f"Unknown model alias
  {raw!r}. Defined aliases: {sorted(self.config.model_aliases)}")`.

Note this is a *runtime* method, distinct from
`validation._check_models`, which runs the same resolvability logic
(`"/" in raw or raw in model_aliases`) at load time so an unresolvable
model is a load-time `ValidationError`, not a later `ValueError` at build
time — the `ValueError` path only fires if a `Project`'s in-memory config
is mutated after a successful load (exercised directly by
`test_resolve_model_unknown_alias_raises`, which mutates
`project.agents["coordinator"].config.model` post-load).

**`Project.build(agent_name, target) -> Any`** — thin delegate to
`adapters.get_adapter(target).build(self, agent_name)`. The `from
.adapters import get_adapter` import is local to this method (not module
scope) specifically so that constructing/using a `Project` never requires
any agent SDK to be installed — only calling `build()` for a given target
needs that target's SDK.

**`Project.check_env(agent_name) -> list[str]`** — returns the `name`s of
every `EnvRequirement` in the agent's `requires.env` where `required` is
`True` and `os.environ.get(name)` is falsy. Presence-only: it never reads
or validates the value, only whether the variable is set.

**`Project._require_agent(agent_name) -> AgentSpec`** — raises
`KeyError(f"Unknown agent {agent_name!r}. Known agents:
{sorted(self.agents)}")` if not found; used by both `resolve_model` and
`check_env`.

## `loader.py`

`load(path) -> Project` — the `common/` → `Project` pipeline. Best-effort
collection, then one accumulated raise:

1. `root = Path(path)`. If `root.is_dir()` is `False`, raises
   `ValidationError([f"project folder not found: {root}"])` **immediately**
   — this one check is not deferred to the accumulation pass, since nothing
   else can proceed without a folder.
2. `_load_project_config(root, errors)` — reads `config.yaml`. Missing
   file, invalid YAML, or a Pydantic validation failure each append one
   message to `errors` and return `None` (not raised yet).
3. `_load_interactions(root, errors)` — same pattern for
   `interactions.yaml`.
4. `_discover_agent_folders(root)` — every subdirectory of `root` not
   starting with `.`, sorted by name (deterministic iteration order).
5. For each folder, in that sorted order:
   - `_load_agent_config(folder, errors)` — reads `agent-config.yaml`
     (missing/invalid YAML/Pydantic errors appended, folder skipped on
     failure via `continue`).
   - Duplicate-name check: if `cfg.name` was already claimed by an earlier
     folder, appends a `"duplicate agent name '...'"` error and skips this
     folder — the *first* folder claiming a name wins, later ones are
     dropped from the project.
   - `_load_skill(folder, cfg.name, errors)` — reads `skill.md`; missing
     file appends an error and returns `""`. Frontmatter is stripped with
     `_FRONTMATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*\n?", re.DOTALL)`
     applied once (`count=1`) at the start of the file, then the remaining
     text is `.strip()`-ed. A body containing a `---`-delimited block *not*
     at the very start of the file is left untouched (the regex is
     anchored with `\A`).
   - `_load_tools(folder, cfg.name, errors)` — reads `tools.py` via
     `importlib.util.spec_from_file_location(f"commonadk._loaded_tools.{folder.name}",
     path)`, then `importlib.util.module_from_spec` +
     `spec.loader.exec_module(module)`. Any exception raised while
     importing the module (syntax error, import error, top-level
     exception) is caught and appended as `f"{agent_name}/tools.py: error
     while importing: {e!r}"`. On success, every attribute in
     `vars(module)` that is a function (`inspect.isfunction`), whose
     `__module__` equals the module's own synthetic name (so re-exported
     imports from elsewhere aren't picked up as tools), and whose name
     doesn't start with `_`, is turned into a `ToolSpec` via
     `ToolSpec.from_function` and collected into a `{name: ToolSpec}` dict.
6. `validate(...)` (see next section) runs unconditionally after
   collection — even if `project_config` or `graph` came back `None` —
   passing everything collected so far. Its `(errors, warnings)` are
   returned; `val_errors` are appended to the loader's own `errors` list.
7. If `errors` is non-empty at this point (from steps 2–6 combined),
   `raise ValidationError(errors)` — one exception, every problem found.
8. Otherwise, `val_warnings` are emitted one by one via
   `warnings.warn(message, stacklevel=2)` — non-fatal, doesn't stop the
   load.
9. Builds each `AgentSpec`: for `cfg.tools` names that exist in that
   agent's collected tool dict, looks up the matching `ToolSpec` (a name
   missing here would already have failed validation in step 6, so this
   filter is a formality, not a second error path).
10. Returns `Project(config=project_config, agents=agents, graph=graph or
    InteractionGraph())`.

`_format_pydantic_error(exc)` turns a Pydantic `ValidationError` into one
semicolon-joined line: for each error, `"{loc}: {msg}"` (dotted location
path, or `"<root>"` if empty) plus `" (got: {input!r})"` when the offending
input is present and not empty/`None`.

## `validation.py`

`validate(*, project_config, graph, agent_configs, agent_tools,
agent_folder_names) -> (errors, warnings)`. Runs every check and returns
both lists — nothing is raised here; `loader.load` decides what to do with
the results. Checks, in call order:

| Check | Function | Severity |
|---|---|---|
| Tool listed in `agent-config.yaml` but not defined in `tools.py` | `_check_tools` | error |
| Tool missing a docstring | `_check_tools` | error |
| Tool has a parameter without a type hint | `_check_tools` | error |
| Tool has no return type hint | `_check_tools` | **warning** |
| Agent folder name ≠ `name:` field in its `agent-config.yaml` | `_check_folder_names` | error |
| Edge's `from`/`to` names an agent that doesn't exist | `_check_edges` | error |
| Edge `type` outside `delegate`/`handoff` | *(Pydantic `Literal`, at parse time — never reaches `_check_edges`)* | error |
| No entry agent resolvable (`config.yaml` and `interactions.yaml` both unset) | `_check_entry` | error |
| `config.yaml entry` names an unknown agent | `_check_entry` | error |
| `interactions.yaml entry` names an unknown agent | `_check_entry` | error |
| `config.yaml entry` and `interactions.yaml entry` disagree | `_check_entry` | error |
| `default_model` unresolvable (no `/`, not in `model_aliases`) | `_check_models` | error |
| An agent's `model` (or fallback to `default_model`) unresolvable | `_check_models` | error |
| `runtime:` set on any agent | `_check_runtime` | **warning** |

`_check_tools` and `_check_folder_names` always run (guarded only by the
dicts they're passed being non-empty in practice); `_check_edges` only runs
if `graph is not None`; `_check_models` only runs if `project_config is not
None`; `_check_entry` and `_check_runtime` always run and internally handle
`None` project config / graph.

`resolvable(raw)` (local to `_check_models`) is exactly
`Project.resolve_model_string`'s success condition — kept in sync by hand,
not shared code, since one runs at load time over raw config dicts and the
other runs at build time over a live `Project`.

**`ValidationError`**: `.errors: list[str]` holds every individual message.
`str(exc)` (used by the CLI) is `f"commonadk validation failed with
{len(errors)} error(s):\n" + "\n".join(f"  - {e}" for e in errors)` — one
header line, one indented bullet per error.

## `mermaid.py`

Pure rendering over an `InteractionGraph`; no I/O beyond
`write_interaction_layer`'s single file write.

**`_node_id(name)`** sanitizes an agent name into a valid Mermaid node
identifier: `re.sub(r"[^0-9A-Za-z_]", "_", name)` — any character outside
`[0-9A-Za-z_]` becomes `_`. Agent names in the shipped example
(`coordinator`, `researcher`, `writer`) need no sanitization; this exists
for names with spaces/hyphens/etc.

**`render_mermaid(graph)`**:

1. Collects node names: `graph.entry` first (if set), then every edge's
   `from_`/`to`, deduplicated via `dict.setdefault` (order of first
   appearance is irrelevant since the node list is re-sorted next).
2. Emits `flowchart TD`, then one line per node **sorted alphabetically by
   name** (not by graph order). The entry node renders as a Mermaid
   *stadium* shape with an explicit label: `{node_id}(["{name}
   (entry)"])`. Every other node renders as a rectangle: `{node_id}["{name}"]`.
3. Emits edges in `graph.edges` order (not sorted): `delegate` edges as a
   solid arrow with a label, `{src} -- delegate --> {dst}`; `handoff`
   edges as a dashed arrow, `{src} -. handoff .-> {dst}` — visually
   distinct arrow styles are asserted directly by
   `test_render_mermaid_distinguishes_delegate_and_handoff`.

**`write_interaction_layer(common_dir, graph) -> Path`** writes
`{common_dir}/interaction-layer.md`: the generated-file header, a `#
Interaction Layer` heading, then the mermaid block inside a ` ```mermaid `
fence. The header text is:

```
<!-- GENERATED FILE -- do not edit by hand.
     Regenerate with `commonadk.mermaid.write_interaction_layer`
     (or `commonadk render`) from interactions.yaml. -->
```

This is verbatim what ships in `mermaid.py` and in the committed
`examples/research-crew/common/interaction-layer.md`.

## `adapters/`

### `base.py` — `BaseAdapter`

Abstract base every target adapter subclasses. Contract:

- `target: str` — the string a caller passes as `target=` (e.g.
  `"google-adk"`, `"openai"`), also used in error messages.
- `build(self, project: Project, agent_name: str) -> Any` — abstract;
  each adapter's own docstring documents what concrete type it returns.

Two methods are implemented once here and shared by every adapter, because
"which agents does this build touch" and "fail loudly up front if any of
them is missing a required env var" are target-independent:

**`_reachable_agents(project, start) -> list[str]`** — plain breadth-first
search over `project.graph.edges`, starting from `start` and following
`edge.from_ == current` outward. Returns `start` plus every agent
transitively reachable via edges, `start` included even with no outgoing
edges. Order is BFS discovery order (not sorted).

**`_check_env(project, agent_name)`** — for every agent in
`_reachable_agents(project, agent_name)` (not just `agent_name` itself, and
not just its direct sub_agents/handoffs — the whole build can transitively
need any reachable agent's env vars), calls `project.check_env(name)` and
collects `"  - {agent}: {var_name} ({description})"` lines for every
missing required var. If any were collected, raises:

```
OSError(
    "commonadk: missing required environment variable(s) for "
    f"target {self.target!r} (building '{agent_name}'):\n"
    + "\n".join(missing_lines)
)
```

This is the *build-time* env preflight — distinct from
`commonadk validate`'s informational set/not-set report, which never
raises.

### `adapters/__init__.py` — registry

`_REGISTRY: dict[str, tuple[str, str, str]]` maps target name to `(module
path, class name, pip extra)` — six entries:

| target | module | class | extra |
|---|---|---|---|
| `"google-adk"` | `commonadk.adapters.google_adk` | `GoogleADKAdapter` | `google` |
| `"openai"` | `commonadk.adapters.openai_agents` | `OpenAIAgentsAdapter` | `openai` |
| `"claude"` | `commonadk.adapters.claude_agent` | `ClaudeAgentSDKAdapter` | `claude` |
| `"crewai"` | `commonadk.adapters.crewai_adapter` | `CrewAIAdapter` | `crewai` |
| `"autogen"` | `commonadk.adapters.autogen_adapter` | `AutoGenAdapter` | `autogen` |
| `"langgraph"` | `commonadk.adapters.langgraph_adapter` | `LangGraphAdapter` | `langgraph` |

**`get_adapter(target) -> BaseAdapter`** — two error modes:

1. `target not in _REGISTRY` → `ValueError(f"Unknown build target
   {target!r}. Known targets: {sorted(_REGISTRY)}")`.
2. `target` is known but `import_module(module_path)` raises `ImportError`
   (the SDK isn't installed) → re-raised as `ImportError(f"target
   {target!r} requires its SDK to be installed. Install it with: pip
   install \"commonadk[{extra}]\" (underlying import error: {e})")`,
   chained (`from e`).

On success, `getattr(module, class_name)()` — the adapter class is
instantiated with no arguments.

### `google_adk.py` — `GoogleADKAdapter`

`target = "google-adk"`.

**`build(project, agent_name)`** calls `self._check_env(...)` first, then
`self._build_agent(project, agent_name, claimed={}, ancestors=(),
parent=None)`.

**`_build_agent(project, name, claimed, ancestors, parent) -> Agent`** —
recursive tree construction, checked *before* any `Agent(...)` is built at
this node:

- If `name in ancestors`: raises `ValueError` reporting the full cycle
  chain (`" -> ".join((*ancestors, name))`) — "Google ADK's sub_agents
  cannot represent a cycle."
- Elif `name in claimed`: raises `ValueError` naming both the agent
  already claiming it (`claimed[name]`) and the conflicting edge
  (`f"'{parent}' -> '{name}'"`) — "Google ADK's sub_agents form a tree (an
  agent can only have one parent)."
- Otherwise: records `claimed[name] = parent or "<build root>"`, extends
  `ancestors` with `name`, recursively builds every `edge.to` where
  `edge.from_ == name` as a sub-agent (passing the extended ancestors and
  `parent=name`), then constructs and returns:

  ```python
  Agent(
      name=spec.name,
      description=spec.config.description,
      instruction=spec.instructions,
      model=self._model_for(project, spec),
      tools=[tool.func for tool in spec.tools],
      generate_content_config=self._generate_content_config(spec),
      sub_agents=sub_agents,
  )
  ```

**Why the pre-check exists** (not just relying on ADK itself): ADK's
`BaseAgent.model_post_init` → `__set_parent_agent_for_sub_agents` raises if
a sub-agent **instance** already has `parent_agent` set — but that only
fires when the *same object* is passed as a sub-agent twice. Building a
second, independent `Agent` instance for the same logical name under a
second parent would sail straight past that guard and silently duplicate
the agent in the tree rather than error. Since `_build_agent` constructs a
fresh instance per recursive call, commonadk has to track "already claimed"
itself, by logical name, before construction — which is exactly what
`claimed` does.

**Model routing (`_model_for`)**:

1. If `spec.config.targets.get("google-adk", {})` has a `"model"` key,
   return it verbatim (already assumed SDK-native — no further
   processing).
2. Else `resolved = project.resolve_model(spec.name)`
   (LiteLLM-format). Split on the first `/`. If the provider is
   `"gemini"`, return the bare model id (the part after `/`) — ADK speaks
   Gemini natively.
3. Otherwise, `from google.adk.models.lite_llm import LiteLlm; return
   LiteLlm(model=resolved)` — wraps the full LiteLLM string for every
   other provider (e.g. `anthropic/claude-sonnet-5`).

**`model_params` → `GenerateContentConfig` (`_generate_content_config`)**:
`_MODEL_PARAM_MAP = {"temperature": "temperature", "max_tokens":
"max_output_tokens"}`. Returns `None` immediately if `model_params` is
empty. For each key present: if it's not in the map, `warnings.warn(...)`
("... is not supported by the Google ADK adapter and will be ignored") and
skip it; otherwise map the key and carry the value through. Returns `None`
if nothing mapped, else `google.genai.types.GenerateContentConfig(**kwargs)`.

### `openai_agents.py` — `OpenAIAgentsAdapter`

`target = "openai"`.

**`build(project, agent_name)`** calls `self._check_env(...)`, then
`self._get_or_build(project, agent_name, memo={})` and returns `memo[agent_name]`.

**`_get_or_build(project, name, memo) -> Agent`** — memoized two-pass
construction:

1. If `name in memo`, return the existing instance immediately (no
   rebuild).
2. Otherwise construct the `Agent` with `handoffs=[]`:

   ```python
   Agent(
       name=spec.name,
       handoff_description=spec.config.description or None,
       instructions=spec.instructions,
       model=self._model_for(project, spec),
       model_settings=self._model_settings(spec),
       tools=[function_tool(tool.func) for tool in spec.tools],
       handoffs=[],
   )
   ```

3. **Records it in `memo[name]` immediately — before recursing** into its
   own handoff targets.
4. *Then* recurses: `agent.handoffs = [self._get_or_build(project, edge.to,
   memo) for edge in project.graph.edges if edge.from_ == name]`,
   mutating `.handoffs` on the already-memoized instance.

**Why this makes multi-parent and cycles safe**: step 3 happens *before*
step 4. If B's handoff graph loops back to A while A is still being built,
the recursive call for A finds A already in `memo` (from before A started
recursing into B) and returns that same reference instead of recursing
forever or building a duplicate. And because `agents.Agent.handoffs` is
just `list[Agent | Handoff]` with no parent-tracking (`Agent.__post_init__`
only validates field *types*), the same `Agent` instance can legitimately
be appended to more than one parent's `.handoffs` list — so, unlike
`GoogleADKAdapter`, this adapter never needs to reject a multi-parent or
cyclic graph; it just shares references
(`test_multi_parent_graph_builds_with_shared_instance`,
`test_cyclic_graph_builds_with_wired_handoff_references`,
`tests/test_adapter_openai.py`).

**Model routing (`_model_for`)** — same shape as the Google adapter,
mirrored provider check: `spec.config.targets.get("openai", {})["model"]`
override wins verbatim; else resolve and check for the `"openai"` provider
prefix (bare model id passed through natively, e.g. `gpt-4o`); else `from
agents.extensions.models.litellm_model import LitellmModel; return
LitellmModel(model=resolved)`.

**`model_params` → `ModelSettings` (`_model_settings`)**:
`_MODEL_PARAM_MAP = {"temperature": "temperature", "max_tokens":
"max_tokens"}`. Same unknown-key warning pattern as the Google adapter.
Unlike `_generate_content_config`, this always returns a `ModelSettings(...)`
instance (constructed even with an empty `kwargs`), never `None` —
`ModelSettings()` with no args is `agents`'s own default.

### `claude_agent.py` — `ClaudeAgentSDKAdapter`

`target = "claude"`. Verified against claude-agent-sdk 0.2.144.

**What `build()` returns**: unlike every adapter above, this SDK has no
persistent "agent object" — it's session/query-based, driven by a
`claude_agent_sdk.query(prompt=..., options=...)` call. So `build()`
returns a fully-wired `claude_agent_sdk.ClaudeAgentOptions`: the requested
agent's `instructions` as `system_prompt`, its resolved model as `model`,
its own `tools.py` functions registered as an in-process MCP server, and
every other reachable agent wired into `options.agents` as
`claude_agent_sdk.AgentDefinition` subagents.

**`build(project, agent_name)`**:

1. `self._check_env(...)`, then `reachable = self._reachable_agents(...)`
   (includes `agent_name`) and `has_children[name]` — whether `name` has
   any outgoing edge.
2. For every reachable agent with `spec.tools`: wraps each `ToolSpec` with
   `claude_agent_sdk.tool(name, description, input_schema)` (a JSON Schema
   built from `ToolSpec.parameters`, `_JSON_SCHEMA_TYPE` mapping
   `str/int/float/bool` → `string/integer/number/boolean`, anything else
   falling back to `"string"`), bundles them into one
   `create_sdk_mcp_server(name=f"{name}_tools", tools=[...])`, and records
   its own tool names as `f"mcp__{name}_tools__{tool.name}"`. An agent with
   no tools gets an empty own-tool-names list and no server.
3. Builds `options.agents`: one `AgentDefinition` per reachable agent
   *other than* `agent_name` itself (the root isn't a subagent of itself —
   it *is* `options`), each with `tools = own_tool_names[name] +
   (["Agent"] if has_children[name] else [])` and `mcpServers =
   [f"{name}_tools"]` if it has tools, else `[]`.
4. Builds the root's own tool list the same way (`root_tools`, with
   `"Agent"` appended iff the root has outgoing edges) and a `disallowed`
   list — every *other* reachable agent's own `mcp__..__` tool names —
   removed from the root's built-in-tool surface.
5. Returns:

   ```python
   ClaudeAgentOptions(
       system_prompt=root_spec.instructions,
       model=self._model_for(project, root_spec),
       tools=[],                       # no built-in Claude Code tools
       allowed_tools=root_tools,
       disallowed_tools=disallowed,
       mcp_servers=mcp_servers,
       agents=agents,
   )
   ```

**Flat subagent registry, gated by edges** — the notable verified upstream
constraint: `AgentDefinition` has no nested "agents" field of its own —
subagents live in one flat `dict[str, AgentDefinition]` on
`ClaudeAgentOptions`. Per the SDK's docs
(`code.claude.com/docs/en/agent-sdk/subagents`), any agent whose `tools`
includes `"Agent"` can invoke *any* name in that flat registry via the
Agent tool's `subagent_type` argument — a lookup against the whole
registry, not scoped to a parent-declared child list (subagents can spawn
subagents up to `CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH`, default 3). So a
deep `interactions.yaml` edge (e.g. `researcher -> writer` when building
`coordinator`) is honestly representable: every reachable agent (the full
transitive closure) is registered, and `"Agent"` access is granted only to
agents that actually have an outgoing edge — delegation happens exactly
where the graph says it can, even though the underlying lookup mechanism
is more permissive. This also means multi-parent graphs and cycles need no
special handling: `options.agents` is a plain dict keyed by logical agent
name, so a name reached by two paths, or a cycle back to the build root
(excluded from `options.agents` since it *is* `options`), is simply the
same dict entry, or a no-op.

**Tool restriction detail**: `AgentDefinition.tools`, when set, is a
restrictive allowlist (not additive to some base set — the SDK's own docs
say an *omitted* `tools` field inherits everything). `ClaudeAgentOptions`
has no equivalent restrictive field for the main session, so the build
root is restricted differently: `tools=[]` turns off every built-in Claude
Code tool (Read/Write/Bash/...; commonadk agents are pure custom-tool
agents), and `disallowed_tools` explicitly strips every other reachable
agent's own `mcp__..__` names from the root's visibility — `disallowed_tools`
actually removes a tool from the model's context, unlike `allowed_tools`,
which only pre-approves without restricting.

**Model routing (`_model_for`)**: Anthropic-native only — no LiteLLM path
exists anywhere in this SDK. `targets.claude.model` override wins verbatim
(already SDK-native — a bare model id or alias like `"sonnet"`, `"opus"`,
`"haiku"`, `"inherit"`). Else, `resolved = project.resolve_model(spec.name)`;
if the provider is `"anthropic"`, return the bare id after the `/`.
Otherwise raises:

```
ValueError(
    f"commonadk: agent {spec.name!r} resolves to model {resolved!r}, "
    f"but the Claude Agent SDK target ('claude') runs Anthropic "
    f"models only -- there is no LiteLLM path for this target. Fix "
    f"this by either: using an 'anthropic/...' model (e.g. "
    f"'anthropic/claude-sonnet-5'), changing {spec.name}'s model "
    f"alias in config.yaml to one that resolves to an "
    f"'anthropic/...' string, or adding a `targets.claude.model` "
    f"override to {spec.name}/agent-config.yaml with an SDK-native "
    f"model id or alias (e.g. 'claude-sonnet-5', 'sonnet', 'opus', "
    f"'haiku')."
)
```

This is why the shipped research-crew example (gemini-default models)
needs `targets.claude.model: claude-sonnet-5` added to every agent to build
for this target at all — verified and tested
(`test_shipped_example_without_claude_overrides_fails_with_clear_error`,
`tests/test_adapter_claude.py`).

**`model_params`**: `_MODEL_PARAM_MAP = {}` — this SDK exposes no
per-request sampling controls analogous to `temperature`/`max_tokens`
anywhere in `claude_agent_sdk.types` (grepped directly; the closest fields,
`thinking`/`effort`/`max_thinking_tokens`/`max_turns`, are reasoning-effort
and turn-budget controls, not sampling params). Every `model_params` key is
therefore warned-and-ignored, unconditionally.

**Warn-vs-error**: model-provider mismatch is the only hard error (build
root and every reachable agent must resolve to `anthropic/...` or carry a
`targets.claude.model` override); every `model_params` key is a warning,
same policy as every other adapter.

### `crewai_adapter.py` — `CrewAIAdapter`

`target = "crewai"`. Verified against crewai 1.15.16.

**What `build()` returns**: a live `crewai.Crew` with `tasks=[]` (verified
directly: `Crew(agents=[...], tasks=[], process=...)` constructs
successfully with an empty task list — a `Task` needs a
`description`/`expected_output` that only exist once the caller knows the
real prompt, which is `commonadk run`'s job, not `build()`'s). The caller
supplies the one `Task` at kickoff time.

**`build(project, agent_name)`**:

1. `self._check_env(...)`; `reachable = self._reachable_agents(...)`;
   `has_outgoing[name]` per reachable agent; `member_names = reachable
   minus agent_name`; `is_manager = bool(member_names)`.
2. Builds the root agent (`allow_delegation=has_outgoing[agent_name],
   as_manager=is_manager`).
3. If not `is_manager` (the root has no reachable agents — a leaf built
   directly, e.g. `writer` in the shipped example): returns
   `Crew(agents=[root_agent], tasks=[], process=Process.sequential)` — the
   solo-member fallback (an empty, non-manager `agents=[]` combined with
   `tasks=[]` is rejected by a different pydantic validator, so this
   fallback, not an empty crew, is the only viable shape).
4. Else: builds every other member (`allow_delegation=has_outgoing[name],
   as_manager=False`) and returns `Crew(agents=members, tasks=[],
   process=Process.hierarchical, manager_agent=root_agent)`.

**Manager-tools constraint** (`_build_agent`): `Crew._create_manager_agent`
(called at `kickoff()`, not `build()`) raises `"Manager agent should not
have tools"` if the manager has any. So when this adapter builds
`agent_name` into the manager role, it forces `tools=[]` regardless of
`agent-config.yaml`, and **warns** if that agent actually declared tools —
they'd otherwise be silently unusable there, and `kickoff()` would
hard-crash. This only affects the root when it becomes manager; every
other crew member (and the root itself when it's the solo sequential
member) keeps its own tools.

**Edge mapping — coarsened, verified not assumed**: both `delegate` and
`handoff` map to CrewAI's one delegation mechanism
(`allow_delegation=True`), which is **crew-wide**:
`Crew._add_delegation_tools` targets `[agent for agent in self.agents if
agent != task.agent]` — every *other* crew member, with no concept of "only
the agents `interactions.yaml` points this agent at." So edge *targets*
are not representable here. What the graph still controls: (1) *whether*
an agent can delegate at all (`allow_delegation=True` only for agents with
≥1 outgoing edge); (2) *scope* (only agents reachable from the build root
join the crew via `_reachable_agents`). Multi-parent graphs and cycles need
no special handling — `crew.agents` is a flat, already-deduped list, so a
shared or cyclic destination is simply the same `Agent` instance appearing
once.

**AgentSpec mapping**: `role=spec.name`, `goal=spec.config.description`,
`backstory=spec.instructions` — `role` is CrewAI's only identifying string
field, mirroring `name` elsewhere; `goal` mirrors `description`; `backstory`
is where CrewAI expects an agent's persona/instructions, mirroring
`instructions` elsewhere.

**Model routing (`_llm_for`)**: `targets.crewai.model` override wins
verbatim; else `model = project.resolve_model(spec.name)` (LiteLLM-format).
Either way, `LLM(model=model, **model_param_kwargs)` — `crewai.LLM.__new__`
is a factory that parses the `"provider/..."` prefix itself, routing known
providers (openai, anthropic, azure, bedrock, gemini, openrouter,
deepseek, ollama, cerebras, ...) to a native client and falling back to
litellm's `completion()` for everything else. **No unsupported-provider
error exists for this target** — the one adapter in this codebase where
every LiteLLM-format string just works.

**`model_params` → `LLM` kwargs**: `_MODEL_PARAM_MAP = {"temperature":
"temperature", "max_tokens": "max_tokens"}` — real constructor fields on
`crewai.LLM` (and every native provider subclass it resolves to). Anything
else (e.g. `top_p`, which `LLM` also exposes but this adapter doesn't wire
up) is warned-and-ignored.

**Tool wiring**: `crewai.tools.tool(func)` — the SDK's own decorator,
builds a pydantic `args_schema` from the function signature; no extra
name/description plumbing needed since `tool_spec.name` is always
`func.__name__`, which is exactly what the decorator derives its name from.

**Telemetry**: CrewAI ships opt-out telemetry, gated dynamically (not just
at import) by `CREWAI_DISABLE_TELEMETRY`/`OTEL_SDK_DISABLED`/
`CREWAI_DISABLE_TRACKING`. `build()` itself never emits telemetry
(`kickoff()` does); the test suite sets those env vars anyway for
belt-and-suspenders offline safety.

**Warn-vs-error**: only the manager-tools case is a warning (tools
silently dropped, not an error) — there is no unsupported-provider or
unsupported-graph-shape error at all for this target; every reachable
agent always builds.

### `autogen_adapter.py` — `AutoGenAdapter`

`target = "autogen"`. Targets Microsoft's current stack
(`autogen-agentchat`/`autogen-core`/`autogen-ext`, 0.4+) — **not** the
community `ag2` fork, an unrelated API despite the shared origin. Verified
against autogen-agentchat/-core/-ext 0.7.5.

**What `build()` returns**: real persistent `AssistantAgent` objects (like
Google ADK and OpenAI Agents), each wired with a `model_client`, its own
`tools.py` functions, and a `handoffs: list[str]`. But a lone
`AssistantAgent`'s handoffs only do anything inside a *team* — verified
directly: a bare `AssistantAgent.run()` answers once and never consults
`.handoffs`. So `build()` picks its return shape based on the root:

- No outgoing edges (a leaf, e.g. `writer`): returns the bare
  `AssistantAgent` — nothing to route to.
- ≥1 outgoing edge: returns a ready-to-run `autogen_agentchat.teams.Swarm`
  whose `participants` are every reachable agent, build root first
  (`_reachable_agents` already returns it at index 0, exactly what `Swarm`
  requires as the initial speaker).

**`build(project, agent_name)`**: builds one `AssistantAgent` per reachable
name (`handoffs = [edge.to for edge in project.graph.edges if edge.from_
== name]`, plain strings), then branches on `has_outgoing` as above,
setting `max_turns=len(reachable)` on any `Swarm` it returns.

**`max_turns` heuristic — a documented v1 limitation, not an SDK
requirement**: with no `termination_condition`/`max_turns`, a `Swarm` "runs
indefinitely" per its own docs — `max_turns`/`termination_condition` are
constructor-only, no per-call override exists on `run`/`run_stream`. Since
`commonadk run` needs one execution that reliably terminates,
`max_turns=len(reachable)` gives exactly enough speaker-turns for one full
pass down a linear chain; it's a heuristic for branchier graphs
(multi-parent, cycles), not a guarantee.

**Handoff targets are name strings, not references** — the notable
verified property: `AssistantAgent.__init__` accepts `handoffs:
List[HandoffBase | str] | None`, wrapping a bare `str` as
`HandoffBase(target=that_string)`; `Swarm` resolves those names against its
own `participants` by name at run time, with no parent-tracking anywhere in
construction. This makes multi-parent graphs and cycles even more trivially
fine than OpenAI Agents' memoized-reference approach: one `AssistantAgent`
per logical name (memoized), each `handoffs` list just plain strings — no
recursion hazard at all, since a name reference carries none.

**Model routing (`_client_for`)** — three native providers, everything
else a clear error:

- `openai/<model>` → `OpenAIChatCompletionClient(model=<bare id>)`.
- `anthropic/<model>` → `AnthropicChatCompletionClient(model=<bare id>,
  model_info=_ANTHROPIC_MODEL_INFO)`. **Verified landmine**: this client's
  bundled model table only knows a handful of dated hardcoded ids and falls
  back to buggy prefix-matching otherwise — `"claude-sonnet-5"` silently
  matches a legacy `claude-2.0` table entry (`function_calling: False`),
  which then makes `AssistantAgent.__init__` raise "The model does not
  support function calling" as soon as tools/handoffs are passed. This
  adapter always passes an explicit `model_info`
  (`function_calling: True, family: ModelFamily.UNKNOWN`) to bypass the
  stale table entirely.
- `gemini/<model>` → also `OpenAIChatCompletionClient(model=<bare id>,
  model_info=_GEMINI_MODEL_INFO)`. No separate native Gemini client exists
  in `autogen_ext.models`; `OpenAIChatCompletionClient.__init__` itself
  special-cases a `"gemini-"`-prefixed model name, pointing `base_url` at
  Gemini's OpenAI-compatible endpoint and reading `GEMINI_API_KEY`. Its
  bundled table is *incomplete* (`gemini-2.5-pro`, used by the shipped
  example's `researcher`, isn't in it), so this adapter always supplies
  explicit `model_info` here too — the base-url/api-key special-casing
  still runs regardless.
- Per-target override (`targets.autogen.model`): passed to
  `OpenAIChatCompletionClient(model=override, **kwargs)` with **no**
  explicit `model_info` — assumed to already be a model that client's own
  table recognizes.
- Anything else (azure, bedrock, ollama, cohere, mistral, ...): clear
  `ValueError` naming the agent, its resolved model, and the fix options.

No override is needed for the shipped research-crew example — `fast`
(gemini), `smart` (anthropic), and researcher's own gemini model are all
covered by the native paths above.

**`model_params`**: `_MODEL_PARAM_MAP = {"temperature": "temperature",
"max_tokens": "max_tokens"}` — both native clients accept these directly
(`TypedDict` `CreateArguments` fields, shared shape). Anything else is
warned-and-ignored.

**Tool wiring**: `AssistantAgent(tools=[...])` accepts plain callables
directly — it wraps each with `autogen_core.tools.FunctionTool` itself. So
`[t.func for t in spec.tools]` passes straight through, no wrapping needed.

**Offline construction — a real difference from Google ADK/OpenAI
Agents/CrewAI/Claude**: `OpenAIChatCompletionClient`/
`AnthropicChatCompletionClient.__init__` eagerly construct the underlying
`openai.AsyncOpenAI`/`anthropic.AsyncAnthropic` client right there in
`build()`, raising immediately (`openai.OpenAIError: "Missing
credentials..."`) if no `api_key` kwarg is given and the matching env var
(`OPENAI_API_KEY`/`ANTHROPIC_API_KEY`/`GEMINI_API_KEY`) isn't set — verified
directly with the vars cleared. `requires.env` has never covered
model-provider auth (it's for an agent's own tool-level vars); this is the
SDK's own eagerness, unwrapped. Tests set fake key-shaped values up front
for exactly this reason (`test_adapter_autogen.py`'s `provider_keys_env`
fixture) — no network call is ever made.

**Warn-vs-error**: unsupported provider is a hard `ValueError`; every
`model_params` key outside the map is a warning; a missing provider API
key is the SDK's own unwrapped `OSError`/`openai.OpenAIError`, surfaced
as-is, not an adapter-specific error.

### `langgraph_adapter.py` — `LangGraphAdapter`

`target = "langgraph"`. Verified against langgraph 1.2.11, langchain
1.3.17, langchain-core 1.6.0, langchain-google-genai 4.3.6,
langchain-anthropic 1.6.1, langchain-openai 1.6.0.

**What `build()` returns**: every reachable agent is its own prebuilt
react agent — a `CompiledStateGraph` from `langchain.agents.create_agent`
(the *current* idiomatic entry point; the older
`langgraph.prebuilt.create_react_agent` still works in this stack but emits
a `LangGraphDeprecatedSinceV10` warning, verified directly, so this adapter
uses `create_agent` throughout).

- No outgoing edges (a leaf, e.g. `writer`): returns that agent's own
  compiled react graph directly.
- ≥1 outgoing edge: every reachable agent is built the same way, each
  wired with one handoff tool per outgoing edge (see below), and all of
  them become nodes of one parent `StateGraph(MessagesState)` with a
  single `START -> <build root>` entry edge. `builder.compile()` returns
  the whole multi-agent `CompiledStateGraph` — routing between nodes at run
  time is driven entirely by the handoff tools, not by any static edges
  added to `builder` beyond that one entry point.

**`build(project, agent_name)`**: `self._check_env(...)`; builds every
reachable agent's node via `_build_agent_node`; if the root has no
outgoing edges, returns its node directly; else adds every node to a
`StateGraph(MessagesState)`, adds `START -> agent_name`, and compiles.

**`_build_agent_node(project, name)`**: `destinations =
dict.fromkeys(edge.to for edge in project.graph.edges if edge.from_ ==
name)` — deduped, one handoff tool per *distinct* destination even if two
edges (e.g. one `delegate`, one `handoff`) point at the same agent.
`create_agent(self._model_for(...), tools=[t.func for t in spec.tools] +
handoff_tools, system_prompt=spec.instructions, name=name)`.

**Edge mapping — precise, per-edge, the notable verified upstream
mechanism**: this is the one target where edge *targets* are fully,
precisely expressible. `_make_handoff_tool(destination)` hand-rolls a
`transfer_to_<destination>` tool using LangGraph's own `Command` primitive
(no `langgraph-supervisor`/`langgraph-swarm` dependency — neither is
installed, and a `Command`-returning tool is LangGraph's own documented,
dependency-free multi-agent handoff mechanism). The tool is annotated with
`InjectedState`/`InjectedToolCallId`, and returns `Command(goto=destination,
update={"messages": [...]}, graph=Command.PARENT)` — `graph=Command.PARENT`
is what makes the jump target a *sibling* node in the parent `StateGraph`
rather than a node inside the calling agent's own react-agent subgraph
(verified via `Command`'s own docstring). An agent with edges to `x` and
`y` gets exactly `transfer_to_x` and `transfer_to_y` and can reach nothing
else. Multi-parent graphs and cycles build successfully with no special
handling: every reachable agent is built exactly once into a
`dict[str, CompiledStateGraph]` (mirroring `_reachable_agents`'s dedup),
`StateGraph.add_node` runs once per entry, and a shared or cyclic
destination is just another `transfer_to_*` tool pointing at a node that
already exists — LangGraph's own execution loop resolves it by name at run
time, not this adapter.

**Model routing (`_model_for`)** — LiteLLM `"provider/model"` maps onto
langchain's `init_chat_model` `"provider:model"` convention
(`_PROVIDER_MAP = {"gemini": "google_genai", "openai": "openai",
"anthropic": "anthropic"}` — only the three providers this adapter's extra
ships an integration package for):

- `gemini/<model>` → `init_chat_model(f"google_genai:{model}")`. Eager:
  raises a pydantic `ValidationError` ("API key required for Gemini
  Developer API") immediately if neither `GOOGLE_API_KEY` nor
  `GEMINI_API_KEY` is set.
- `openai/<model>` → `init_chat_model(f"openai:{model}")`. Also eager:
  raises `openai.OpenAIError: "Missing credentials..."` if `OPENAI_API_KEY`
  isn't set.
- `anthropic/<model>` → `init_chat_model(f"anthropic:{model}")`. **Lazy**,
  unlike the other two — constructs successfully with no
  `ANTHROPIC_API_KEY` set at all, no error until an actual API call.
  Documented as a deliberate asymmetry, not papered over.
- Per-target override (`targets.langgraph.model`): passed to
  `init_chat_model(override, **kwargs)` as-is — its *expected form* is
  already langchain-native `"provider:model"` (e.g.
  `"anthropic:claude-opus-4"`), **not** a bare id and **not** a LiteLLM
  `"provider/model"` string.
- Anything else: clear `ValueError` naming the agent, its resolved model,
  and the fix options — mirroring the Claude/AutoGen adapters' error
  style.

The shipped research-crew example builds unmodified for this target (same
as AutoGen and CrewAI, unlike Claude) — its gemini-default models are all
native `google_genai` paths.

**`model_params`**: `_MODEL_PARAM_MAP = {"temperature": "temperature",
"max_tokens": "max_tokens"}` — one flat map for all three providers.
Verified `ChatGoogleGenerativeAI` declares `max_output_tokens` with a
pydantic `validation_alias="max_tokens"` and `populate_by_name=True`, so
the same `max_tokens=` keyword resolves correctly across all three chat
model classes with no per-provider remapping (unlike AutoGen's
separately-typed clients). Anything else is warned-and-ignored.

**Tool wiring**: bare callables pass straight through in `tools=[...]` —
`create_agent`/`ToolNode` wraps them into a `StructuredTool` itself,
introspecting signature and docstring exactly like `tools.py`'s own
contract already guarantees.

**Offline construction**: like AutoGen, the `google_genai` and `openai`
chat model classes construct their provider client eagerly and fail
immediately on a missing key; `anthropic` is the one of the three that does
not. Tests set fake `GOOGLE_API_KEY`/`OPENAI_API_KEY`/`ANTHROPIC_API_KEY`
up front (`test_adapter_langgraph.py`'s `provider_keys_env` fixture,
mirroring AutoGen's).

**Warn-vs-error**: unsupported provider is a hard `ValueError`; every
`model_params` key outside the map is a warning; a missing key for the two
eager providers (`gemini`, `openai`) is that provider's own unwrapped SDK
error, not an adapter-specific one.

## `cli.py`

Four subcommands plus `--version`, built with `argparse`
(`subparsers.add_parser`, `dest="command", required=True`).

| Command | Args | Behavior |
|---|---|---|
| `validate` | `common_dir` | Loads + validates; prints project name, entry agent, and per-agent model/tools/env status (env vars flagged `set`/`not set` against the current shell, `required`/`optional`) |
| `render` | `common_dir` | Loads + validates, then `write_interaction_layer(common_dir, project.graph)`; prints the output path |
| `run` | `common_dir --target {google-adk,openai,claude,crewai,autogen,langgraph} [--agent NAME] prompt` | Loads, builds one agent for `target`, executes a single turn, prints the final text output |
| `new` | `common_dir agent_name [--from AGENT --type {delegate,handoff}]` | Scaffolds `<agent_name>/{skill.md,tools.py,agent-config.yaml}` under `common_dir`; with `--from`, also appends an edge to `interactions.yaml` and regenerates `interaction-layer.md` (see below) |
| `--version` | — | `argparse`'s built-in `action="version"`; prints `commonadk {version}` (via `importlib.metadata.version("commonadk")`, falling back to `"0.0.0+unknown"` if the package metadata isn't found) and exits `0` via `SystemExit` |

**`new` in detail.** `_cmd_new` (1) refuses outright if `common_dir/
agent_name` already exists (`ValueError`, before touching anything else on
disk); (2) calls `_load_project` on the project AS IT STANDS first — this
both fails loudly if the project is already broken (matching every other
command) and, when `--from` is given, is how that agent name is checked
against the real, loaded agent list (`ValueError` naming the known agents
if `--from` doesn't resolve); (3) writes the three new files from
module-level `str.format` templates (`_NEW_AGENT_SKILL_MD`,
`_NEW_AGENT_TOOLS_PY`, `_NEW_AGENT_CONFIG_YAML`) — the generated
`agent-config.yaml` deliberately omits `model:` so the new agent falls back
to `config.yaml`'s `default_model` (already required to be resolvable by
`validation._check_models`, so this alone can never be what breaks
`commonadk validate`), and its `name:` always matches the folder name it
was written into; (4) when `--from` is given, reads-and-rewrites
`interactions.yaml` (`yaml.safe_load`/`yaml.safe_dump(sort_keys=False)`,
preserving existing key order) to append one edge, `edge_type` defaulting to
`"delegate"`, then reloads the project and calls
`write_interaction_layer(common_dir, project.graph)` — the SAME renderer
`commonadk render` uses, so `interaction-layer.md` is never hand-edited,
here or anywhere else. `--type` without `--from` is rejected up front
(`ValueError`, "--type requires --from") before any file is touched.
Warnings from the final reload are surfaced via `_print_warnings`, matching
`validate`/`render`/`run`.

**Lazy SDK imports.** `validate` and `render` only touch `loader.py` and
`mermaid.py`, neither of which imports any agent SDK at module scope, so
both commands work with zero SDKs installed. `run` needs exactly one SDK —
its imports live inside each target's own `_run_*` function
(`_run_google_adk`, `_run_openai`, `_run_claude`, `_run_crewai`,
`_run_autogen`, `_run_langgraph`), never at module scope, so
`commonadk run ... --target openai` never imports `google.adk`, `claude_agent_sdk`,
or any other target's SDK.

**Per-target `run` behavior**, `_RUN_TARGETS` dict-dispatched off `--target`:

| target | key preflight | execution shape |
|---|---|---|
| `google-adk` | (env preflight only, via `_check_env`) | `InMemoryRunner(agent=..., app_name=...)`, one `run_async` turn, joins final-response text parts |
| `openai` | (env preflight only) | `Runner.run_sync(agent, prompt)`, prints `.final_output` |
| `claude` | **`_run_claude` preflights `ANTHROPIC_API_KEY` itself** — raises the same `OSError` shape as `_check_env` if unset, *before* calling `project.build()`. Nothing in the SDK declares or checks for it up front the way `requires.env` does for tool-level vars | `query(prompt=..., options=...)`, joins every `ResultMessage.result` chunk |
| `crewai` | (env preflight only) | assigns the one `Task` to `crew.agents[0]` for a sequential (solo-member) crew, leaves it unassigned for a hierarchical crew (the manager picks who executes it); `crew.kickoff()`, prints `.raw` |
| `autogen` | (env preflight only — provider API keys are the SDK's own eager-construction concern, inside `project.build()`, not the CLI's) | `built.run(task=prompt)` — `built` is a bare `AssistantAgent` or a `Swarm`, both exposing the same async `.run(task=...)` shape, no branching needed |
| `langgraph` | (env preflight only, same eager-construction note as autogen) | `graph.invoke({"messages": [{"role": "user", "content": prompt}]})`, prints the last message's `.content` — same call shape whether `graph` is a lone react agent or the compiled multi-agent graph |

Every target funnels through the *shared* `_check_env` preflight
(`adapters/base.py`, tool-level `requires.env` only) before its adapter's
`build()` runs; `claude` additionally has its own CLI-level preflight for
`ANTHROPIC_API_KEY` specifically, since that's an SDK-authentication
requirement `requires.env` was never meant to model.

**Warnings surfaced, not swallowed.** `_load_project(common_dir)` wraps
`loader.load()` in `warnings.catch_warnings(record=True)` +
`warnings.simplefilter("always")`, returning `(project, caught)` instead of
letting Python's default warning machinery print to stderr on its own.
`_print_warnings(caught)` then prints a `"Warnings:"` header followed by
one `"  - {message}"` line per warning — called by `validate` and `render`
after their main output, and by `run` immediately after loading (before
attempting to build), so a `runtime:`-key warning or a missing-return-hint
warning is visible in every command's output rather than silently lost.

**Exit-code behavior.** `main()` wraps the dispatched command in one
`try/except` covering `ValidationError`, then `(OSError, ValueError,
ImportError)`: any of those prints one clean message to `stderr` (via
`str(e)` for `ValidationError`, prefixed with `"commonadk: "` for the
others unless the message already starts with `"commonadk"`) and returns
`1`. Every other path returns `0`. `_cmd_run` resolves `--agent` to
`project.config.entry or project.graph.entry` if omitted, raising
`ValueError` (caught by the same handler) if neither is set or if the
resolved name isn't in `project.agents`. An unrecognized `--target` isn't
special-cased in the CLI itself — `_cmd_run` looks it up in its own
`_RUN_TARGETS` dict (six entries, one per adapter registry entry); on a
miss it calls `adapters.get_adapter(target)` purely to raise that
function's `ValueError`, so the CLI never hand-maintains a second "known
targets" list that could drift from `adapters/__init__.py`'s registry.

## Error taxonomy

| Situation | Raised as | Where |
|---|---|---|
| `common/` folder doesn't exist | `ValidationError` (raised immediately, not accumulated) | `loader.load` |
| `config.yaml` missing, invalid YAML, or fails `ProjectConfig` validation | `ValidationError` (accumulated) | `loader._load_project_config` |
| `interactions.yaml` missing, invalid YAML, or fails `InteractionGraph`/edge-type validation | `ValidationError` (accumulated) | `loader._load_interactions` |
| `agent-config.yaml` missing, invalid YAML, unknown key, or fails `AgentConfig` validation | `ValidationError` (accumulated) | `loader._load_agent_config` |
| Two agent folders declare the same `name:` | `ValidationError` (accumulated) | `loader.load` |
| `skill.md` missing | `ValidationError` (accumulated) | `loader._load_skill` |
| `tools.py` missing, or raises while importing | `ValidationError` (accumulated) | `loader._load_tools` |
| Tool in `agent-config.yaml` not defined in `tools.py` | `ValidationError` (accumulated) | `validation._check_tools` |
| Tool missing a docstring or a parameter type hint | `ValidationError` (accumulated) | `validation._check_tools` |
| Tool has no return type hint | `UserWarning` (non-fatal) | `validation._check_tools` |
| Agent folder name ≠ its `agent-config.yaml` `name:` | `ValidationError` (accumulated) | `validation._check_folder_names` |
| Edge names an unknown agent | `ValidationError` (accumulated) | `validation._check_edges` |
| No resolvable entry agent, or `config.yaml`/`interactions.yaml` entries disagree | `ValidationError` (accumulated) | `validation._check_entry` |
| `default_model` or an agent's `model` isn't a LiteLLM string or a known alias | `ValidationError` (accumulated) | `validation._check_models` |
| `runtime:` set on an agent | `UserWarning` (non-fatal) | `validation._check_runtime` |
| `resolve_model`/`resolve_model_string` called post-load with an alias not in `model_aliases` | `ValueError` | `models.Project.resolve_model_string` |
| `resolve_model`/`check_env` called with an unknown agent name | `KeyError` | `models.Project._require_agent` |
| Required env var missing at build time (`_check_env`, transitively over reachable agents) | `OSError` | `adapters.base.BaseAdapter._check_env` |
| `ANTHROPIC_API_KEY` unset for `commonadk run --target claude` | `OSError` (raised by the CLI itself, before `project.build()`) | `cli._run_claude` |
| Unknown `target=` string | `ValueError` | `adapters.get_adapter` |
| Target's SDK not installed | `ImportError` (with `pip install "commonadk[...]"` hint) | `adapters.get_adapter` |
| Same agent reachable from two parents (Google ADK build) | `ValueError` | `adapters.google_adk.GoogleADKAdapter._build_agent` |
| Cycle in the reachable graph (Google ADK build) | `ValueError` | `adapters.google_adk.GoogleADKAdapter._build_agent` |
| Agent's resolved model isn't `anthropic/...` and no `targets.claude.model` override (Claude Agent SDK build) | `ValueError` | `adapters.claude_agent.ClaudeAgentSDKAdapter._model_for` |
| `model_params` key unsupported by the Claude Agent SDK (all keys, since none map) | `UserWarning` (non-fatal) | `adapters.claude_agent.ClaudeAgentSDKAdapter._warn_unsupported_model_params` |
| Build root built as CrewAI hierarchical manager but has declared tools (dropped, not passed through) | `UserWarning` (non-fatal) | `adapters.crewai_adapter.CrewAIAdapter._build_agent` |
| `model_params` key unsupported by the CrewAI adapter | `UserWarning` (non-fatal) | `adapters.crewai_adapter.CrewAIAdapter._model_param_kwargs` |
| Agent's resolved model isn't `openai/anthropic/gemini` and no `targets.autogen.model` override (AutoGen build) | `ValueError` | `adapters.autogen_adapter.AutoGenAdapter._client_for` |
| `model_params` key unsupported by the AutoGen adapter | `UserWarning` (non-fatal) | `adapters.autogen_adapter.AutoGenAdapter._model_param_kwargs` |
| Missing provider API key for AutoGen's `openai`/`anthropic`/`gemini` native model clients (eager client construction) | the underlying SDK's own error (e.g. `openai.OpenAIError`), unwrapped | `autogen_ext`'s model client `__init__`, called from `adapters.autogen_adapter.AutoGenAdapter._client_for` |
| Agent's resolved model isn't `gemini/openai/anthropic` and no `targets.langgraph.model` override (LangGraph build) | `ValueError` | `adapters.langgraph_adapter.LangGraphAdapter._model_for` |
| `model_params` key unsupported by the LangGraph adapter | `UserWarning` (non-fatal) | `adapters.langgraph_adapter.LangGraphAdapter._model_param_kwargs` |
| Missing provider API key for LangGraph's `openai`/`google_genai` chat models (eager; `anthropic` is lazy and does *not* raise here) | the underlying SDK's own error (e.g. `openai.OpenAIError`, a pydantic `ValidationError`), unwrapped | `langchain.chat_models.init_chat_model`, called from `adapters.langgraph_adapter.LangGraphAdapter._model_for` |
| `commonadk new`'s target agent folder already exists | `ValueError` (before any file is written) | `cli._cmd_new` |
| `commonadk new --type` given without `--from` | `ValueError` (before any file is written) | `cli._cmd_new` |
| `commonadk new --from` names an agent not in the (already-loaded) project | `ValueError` naming the known agents | `cli._cmd_new` |
| Any of the above surfacing through the CLI | printed to `stderr`, exit code `1` | `cli.main`'s `try/except` |

## Testing layout

All 124 tests live under `tests/`, sharing two fixtures from
`tests/conftest.py`: `example_common_dir` (path to
`examples/research-crew/common`, read-only) and `tmp_project` (a
`tmp_path`-backed mutable copy of the same, for tests that deliberately
break the project).

**Importorskip gating.** Every adapter-specific test file (`test_adapter_*.py`)
calls `pytest.importorskip("<module>")` **at module scope**, immediately
after the file's docstring and before importing `commonadk` — so the whole
file is skipped, not just individual tests, when that target's SDK isn't
installed, and the `import commonadk` line itself carries a `# noqa: E402`
comment marking the post-importorskip placement as deliberate. Module
names: `google.adk` (google), `agents` (openai), `claude_agent_sdk`
(claude), `crewai` (crewai), `autogen_agentchat` (autogen), and both
`langgraph`/`langchain` (langgraph — two separate `importorskip` calls,
since this adapter needs both packages). `test_hypothesis.py` gates
per-target *inside* the parametrized test body instead, via the same
module-name mapping, so that one file always collects regardless of which
subset of SDKs is installed, skipping only the individual parametrized
cases whose SDK is missing rather than the whole file.

**Fake-key fixture patterns.** Two fixtures recur across every adapter test
file (each file defines its own local copy; `test_hypothesis.py`'s
versions are shown below and are representative):

- `tavily_env` — `monkeypatch.setenv("TAVILY_API_KEY", "test-key")` and
  `monkeypatch.delenv("POSTGRES_DSN", raising=False)`, satisfying
  researcher's one *required* `requires.env` entry while leaving the
  optional one unset on purpose (build must not block on it).
- `provider_keys_env` — fake `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
  `GEMINI_API_KEY`, `GOOGLE_API_KEY` values, needed only by the `autogen`
  and `langgraph` targets (their model clients construct the underlying
  provider SDK client eagerly — see those adapters' "Offline construction"
  above) but harmless to set for every other target's build in the same
  parametrized test, since none of them read those vars. Used by
  `test_adapter_autogen.py`, `test_adapter_langgraph.py`, and
  `test_hypothesis.py`; not needed by `test_adapter_claude.py` or
  `test_adapter_crewai.py`, whose targets don't construct a provider client
  eagerly at `build()` time.

| File | Covers |
|---|---|
| `test_models.py` | `resolve_model` (alias, literal passthrough, default fallback, unknown-alias `ValueError`, unknown-agent `KeyError`); `check_env` (missing required, satisfied, no requirements) |
| `test_loader.py` | Full `load()` happy path (config/entry/agents), instructions + tools populated and callable, `ToolSpec` schema metadata, edges present, frontmatter stripped from `skill.md`, missing folder raises `ValidationError` |
| `test_validation.py` | Each check individually — unknown tool name, untyped param, missing docstring, edge to unknown agent, bad edge type, folder/name mismatch, unknown model alias, entry mismatch, missing `config.yaml`, unknown YAML key, `runtime:` warns when set / silent when unset — plus one test asserting multiple unrelated problems are *all* collected into one `ValidationError.errors` |
| `test_mermaid.py` | Node/edge rendering, entry-node marking, delegate vs. handoff arrow styles, `write_interaction_layer` output shape, and a drift guard (`test_example_interaction_layer_matches_current_graph`) asserting the committed `interaction-layer.md` still matches a fresh render of `interactions.yaml` |
| `test_adapter_google.py` | Gated by `pytest.importorskip("google.adk")`. Happy-path tree build, multi-parent rejection, cycle rejection, gemini-native vs. `LiteLlm`-wrapped model routing, per-target override precedence, env preflight (missing required blocks, optional doesn't, checks agents reachable via edges — not just direct sub_agents), unknown-target error, missing-SDK install-hint error |
| `test_adapter_openai.py` | Gated by `pytest.importorskip("agents")`. Happy-path build, multi-parent graph building via a shared instance, cyclic graph building via post-hoc wiring, openai-native vs. `LitellmModel`-wrapped routing, per-target override, env preflight |
| `test_adapter_claude.py` | Gated by `pytest.importorskip("claude_agent_sdk")`. Happy-path build (flat `options.agents` registry, tool/MCP-server wiring), multi-parent and cyclic graphs building without special handling, anthropic-native model resolution, non-anthropic provider raising a clear error, the shipped example failing *without* `targets.claude.model` overrides and succeeding *with* them, per-target override precedence, `model_params` all warned-and-ignored, env preflight (including the reachable-via-edges case) |
| `test_adapter_crewai.py` | Gated by `pytest.importorskip("crewai")`. Happy-path hierarchical build, solo-sequential fallback for a leaf build, multi-parent and cyclic graphs building via a flat deduped member list, LiteLLM-string-passthrough model routing (including a non-native provider falling back to litellm with *no* error), per-target override, unsupported `model_params` key warned, env preflight |
| `test_adapter_autogen.py` | Gated by `pytest.importorskip("autogen_agentchat")`. Happy-path `Swarm` build, bare-`AssistantAgent` return for a leaf build (not a team), multi-parent and cyclic graphs via shared participants, openai/gemini/anthropic native model-client routing (including the explicit `model_info` workaround), unsupported-provider error, per-target override, unsupported `model_params` key warned, env preflight |
| `test_adapter_langgraph.py` | Gated by `pytest.importorskip("langgraph")` and `pytest.importorskip("langchain")`. Happy-path multi-agent `StateGraph` build, bare-react-agent return for a leaf build, multi-parent and cyclic graphs via a flat node registry, duplicate edges to the same destination yielding exactly one handoff tool, gemini/openai/anthropic native routing, unsupported-provider error, per-target override, unsupported `model_params` key warned, env preflight |
| `test_hypothesis.py` | `test_same_project_builds_on_every_installed_target`, parametrized over all six `target` strings — the v1 success criterion made executable: one `Project`, loaded once, built under whichever targets are actually installed (each case individually skipped via inline `pytest.importorskip`, not the whole file). Asserts each target's return value carries the build root's identity somewhere, under that SDK's own attribute shape (`.name` for google-adk/openai; `.system_prompt` for claude; `.manager_agent.role` for crewai; `._participant_names[0]` for autogen; `"coordinator" in .nodes` for langgraph) |
| `test_cli.py` | In-process `cli.main(argv)` calls (no subprocess) asserting exit codes and captured stdout/stderr: `validate` (success summary incl. env set/not-set, broken project exits 1, missing folder exits 1), `render` (writes file, broken project exits 1), `run` (missing `requires.env` var, missing `ANTHROPIC_API_KEY` for `--target claude` naming that var in stderr, unknown target, unknown agent, broken project exits 1 before touching any SDK), `--version` |
| `test_cli_new.py` | `commonadk new`: happy path (scaffolded files' shape, output passes `commonadk validate`, `interactions.yaml`/`interaction-layer.md` untouched with no `--from`), refuse-to-overwrite (an existing shipped agent and a just-scaffolded one, files left untouched either way), the `--from`/`--type` edge variant (edge appended, `interaction-layer.md` regenerated via the real renderer, default edge type is `delegate`, output passes `commonadk validate`), `--type` without `--from` rejected, unknown `--from` agent rejected (naming the known agents), broken project exits 1 before scaffolding anything, missing project folder exits 1 |
