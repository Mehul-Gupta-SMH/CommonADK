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
├── cli.py                  # argparse CLI: validate | render | run | --version
└── adapters/
    ├── __init__.py         # target -> adapter registry, lazy SDK imports
    ├── base.py               # BaseAdapter ABC + shared env-preflight/BFS
    ├── google_adk.py           # AgentSpec -> google.adk.agents.Agent
    └── openai_agents.py         # AgentSpec -> agents.Agent
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
     (or `commonadk render`, once the CLI lands) from interactions.yaml. -->
```

This is verbatim what ships in `mermaid.py` and in the committed
`examples/research-crew/common/interaction-layer.md` — note the parenthetical
still reads "once the CLI lands" even though `commonadk render` (`cli.py`)
has existed since M4; it's a stale comment carried over from before the CLI
landed, not a sign the CLI is missing.

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
path, class name, pip extra)`:

| target | module | class | extra |
|---|---|---|---|
| `"google-adk"` | `commonadk.adapters.google_adk` | `GoogleADKAdapter` | `google` |
| `"openai"` | `commonadk.adapters.openai_agents` | `OpenAIAgentsAdapter` | `openai` |

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

## `cli.py`

Three subcommands plus `--version`, built with `argparse`
(`subparsers.add_parser`, `dest="command", required=True`).

| Command | Args | Behavior |
|---|---|---|
| `validate` | `common_dir` | Loads + validates; prints project name, entry agent, and per-agent model/tools/env status (env vars flagged `set`/`not set` against the current shell, `required`/`optional`) |
| `render` | `common_dir` | Loads + validates, then `write_interaction_layer(common_dir, project.graph)`; prints the output path |
| `run` | `common_dir --target {google-adk,openai} [--agent NAME] prompt` | Loads, builds one agent for `target`, executes a single turn, prints the final text output |
| `--version` | — | `argparse`'s built-in `action="version"`; prints `commonadk {version}` (via `importlib.metadata.version("commonadk")`, falling back to `"0.0.0+unknown"` if the package metadata isn't found) and exits `0` via `SystemExit` |

**Lazy SDK imports.** `validate` and `render` only touch `loader.py` and
`mermaid.py`, neither of which imports any agent SDK at module scope, so
both commands work with zero SDKs installed. `run` needs exactly one SDK —
its imports (`google.adk.runners.InMemoryRunner`, `google.genai.types`, or
`agents.Runner`) live inside `_run_google_adk`/`_run_openai` respectively,
never at module scope, so `commonadk run ... --target openai` never
imports `google.adk` and vice versa.

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
`_RUN_TARGETS` dict (`{"google-adk": ..., "openai": ...}`); on a miss it
calls `adapters.get_adapter(target)` purely to raise that function's
`ValueError`, so the CLI never hand-maintains a second "known targets"
list that could drift from `adapters/__init__.py`'s registry.

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
| Unknown `target=` string | `ValueError` | `adapters.get_adapter` |
| Target's SDK not installed | `ImportError` (with `pip install "commonadk[...]"` hint) | `adapters.get_adapter` |
| Same agent reachable from two parents (Google ADK build) | `ValueError` | `adapters.google_adk.GoogleADKAdapter._build_agent` |
| Cycle in the reachable graph (Google ADK build) | `ValueError` | `adapters.google_adk.GoogleADKAdapter._build_agent` |
| Any of the above surfacing through the CLI | printed to `stderr`, exit code `1` | `cli.main`'s `try/except` |

## Testing layout

All 64 tests (per `tasks.md`) live under `tests/`, sharing two fixtures
from `tests/conftest.py`: `example_common_dir` (path to
`examples/research-crew/common`, read-only) and `tmp_project` (a
`tmp_path`-backed mutable copy of the same, for tests that deliberately
break the project).

| File | Covers |
|---|---|
| `test_models.py` | `resolve_model` (alias, literal passthrough, default fallback, unknown-alias `ValueError`, unknown-agent `KeyError`); `check_env` (missing required, satisfied, no requirements) |
| `test_loader.py` | Full `load()` happy path (config/entry/agents), instructions + tools populated and callable, `ToolSpec` schema metadata, edges present, frontmatter stripped from `skill.md`, missing folder raises `ValidationError` |
| `test_validation.py` | Each check individually — unknown tool name, untyped param, missing docstring, edge to unknown agent, bad edge type, folder/name mismatch, unknown model alias, entry mismatch, missing `config.yaml`, unknown YAML key, `runtime:` warns when set / silent when unset — plus one test asserting multiple unrelated problems are *all* collected into one `ValidationError.errors` |
| `test_mermaid.py` | Node/edge rendering, entry-node marking, delegate vs. handoff arrow styles, `write_interaction_layer` output shape, and a drift guard (`test_example_interaction_layer_matches_current_graph`) asserting the committed `interaction-layer.md` still matches a fresh render of `interactions.yaml` |
| `test_adapter_google.py` | Gated by `pytest.importorskip("google.adk")` at module scope (whole file skipped, not just individual tests, if `google-adk` isn't installed). Happy-path tree build, multi-parent rejection, cycle rejection, gemini-native vs. `LiteLlm`-wrapped model routing, per-target override precedence, env preflight (missing required blocks, optional doesn't, checks agents reachable via edges — not just direct sub_agents), unknown-target error, missing-SDK install-hint error |
| `test_adapter_openai.py` | Gated by `pytest.importorskip("agents")` at module scope. Happy-path build, multi-parent graph building via a shared instance, cyclic graph building via post-hoc wiring, openai-native vs. `LitellmModel`-wrapped routing, per-target override, env preflight, and `test_same_project_builds_on_both_targets` — the hypothesis test, which additionally does `pytest.importorskip("google.adk")` inline to build the *same* `Project` on both targets in one test |
| `test_cli.py` | In-process `cli.main(argv)` calls (no subprocess) asserting exit codes and captured stdout/stderr: `validate` (success summary incl. env set/not-set, broken project exits 1, missing folder exits 1), `render` (writes file, broken project exits 1), `run` (missing env var, unknown target, unknown agent, broken project exits 1 before touching any SDK), `--version` |
