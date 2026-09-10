"""commonadk's command-line interface.

Five subcommands, mirroring plan.md ("M4 -- CLI & docs") plus the `new`
scaffolding command added for issue #12 and the `import` command added for
SKILL.md interoperability:

- `commonadk validate <common-dir>` -- load + validate a `common/` project and
  print a human-readable summary (or the error list, on failure).
- `commonadk render <common-dir>` -- regenerate `interaction-layer.md` from
  `interactions.yaml`.
- `commonadk run <common-dir> --target
  {google-adk,openai,claude,crewai,autogen,langgraph} PROMPT` -- build an
  agent for a target SDK and execute one turn.
- `commonadk new <common-dir> <agent-name> [--from AGENT --type
  {delegate,handoff}]` -- scaffold a new, conforming agent folder
  (`skill.md`, `tools.py`, `agent-config.yaml`) inside an existing `common/`
  project, optionally wiring an edge into `interactions.yaml` from an
  existing agent and regenerating `interaction-layer.md` through
  `mermaid.write_interaction_layer` (never hand-edited). Refuses to
  overwrite an existing folder. The scaffolded output is designed to pass
  `commonadk validate` immediately: no `model:` override (falls back to
  `config.yaml`'s `default_model`, which validation already requires to be
  resolvable), and a folder name that matches the generated `name:`.
- `commonadk import <skills-dir> <common-dir> [--entry NAME] [--name
  PROJECT] [--model ALIAS-OR-STRING]` -- turn a directory of SKILL.md files
  (Spotify `portal-ai-plugins`-style: nested `<name>/SKILL.md`, or a flat
  `*.md` directory) into a conforming `common/` project: one agent folder
  per skill (original `skill.md` preserved verbatim, frontmatter and all;
  `agent-config.yaml` with an empty `tools:`; a stub `tools.py`), plus
  `config.yaml`/`interactions.yaml` (no invented edges) when creating a new
  project. See `_cmd_import`'s docstring for the full behavior, including
  name normalization and the refuse-to-clobber rule for an existing
  `common-dir`.

`validate` and `render` never need an agent SDK installed -- they only touch
`loader.py`/`mermaid.py`, which have no SDK imports at module scope. `new`
and `import` are the same: scaffolding and rewiring `interactions.yaml`
only ever touches those two modules too. `run` does need an SDK, but only
for the target actually requested, so every SDK-touching import in this
module lives inside the function that uses it (see `_run_google_adk` /
`_run_openai` / `_run_claude` / `_run_crewai` / `_run_autogen` /
`_run_langgraph`), never at module scope. `_run_claude` additionally
preflights `ANTHROPIC_API_KEY` itself -- the Claude Agent SDK's bundled CLI
needs it to authenticate, but (unlike `requires.env` in
`agent-config.yaml`) nothing in the SDK declares or checks for it up front.

Every command funnels its expected failure modes -- `ValidationError` (bad
project), `OSError` (missing required env var, from the adapters' env
preflight), `ValueError` (unknown build target, unbuildable graph, `new`'s
own refuse-to-overwrite / unknown `--from` agent errors, `import`'s own
refuse-to-clobber / unknown `--entry` errors), and `ImportError` (target SDK
not installed) -- through `main`'s top-level `try/except`, so the CLI
always prints one clean message and a non-zero exit code instead of a
Python traceback.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

import yaml

from .loader import _FRONTMATTER_RE, load
from .mermaid import write_interaction_layer
from .validation import ValidationError

if TYPE_CHECKING:
    from .models import Project


# ---------------------------------------------------------------------------
# `new` scaffolding templates
# ---------------------------------------------------------------------------
#
# Plain `str.format(agent=..., title=...)` templates -- literal `{`/`}` that
# must survive into the generated file (the example tool's own f-string) are
# doubled (`{{`/`}}`), the standard str.format escaping, since these are NOT
# f-strings themselves: the agent name is only known at scaffold time, not
# when this module is imported.

_NEW_AGENT_SKILL_MD = """# {title}

TODO: describe {agent}'s persona and what it should do when invoked.

Use `example_tool` to process input before returning a final answer.
"""

_NEW_AGENT_TOOLS_PY = '''"""Tools available to the {agent} agent."""

from __future__ import annotations


def example_tool(text: str) -> str:
    """Example tool -- replace with real logic specific to this agent.

    Args:
        text: Input text to process.

    Returns:
        A short, deterministic transformation of the input (replace with
        real logic).
    """
    return f"processed: {{text}}"
'''

_NEW_AGENT_CONFIG_YAML = """name: {agent}
description: "TODO: describe what {agent} does."

tools:
  - example_tool

requires:
  env: []
"""


def _title_from_agent_name(agent_name: str) -> str:
    return agent_name.replace("_", " ").replace("-", " ").title()


# ---------------------------------------------------------------------------
# `import` scaffolding
# ---------------------------------------------------------------------------
#
# `commonadk import <skills-dir> <common-dir>` turns a directory of SKILL.md
# files into a conforming `common/` project. Unlike `new`'s templates above,
# each agent's `skill.md` is NOT a template -- it's the original SKILL.md
# file, byte for byte, frontmatter included (docs/file-contracts.md now
# documents what commonadk does with that frontmatter at load time). Only
# `agent-config.yaml` and `tools.py` are generated.

_IMPORTED_AGENT_TOOLS_PY = '''"""Tools available to the {agent} agent.

Imported by `commonadk import` from a SKILL.md file. This skill declared no
commonadk-native tools -- its own instructions (see skill.md) may already
describe CLI commands or other actions for a host to run directly. Add
typed, docstringed functions here and list their names in this agent's
`agent-config.yaml` `tools:` to give it real commonadk tools; an agent with
no tools listed is a legal, valid commonadk agent as-is.
"""

from __future__ import annotations
'''

# A real, always-resolvable LiteLLM-format model string (contains "/", so it
# needs no `model_aliases` entry to pass `validation._check_models`) used as
# `--model`'s default, and as the alias target when `--model` is given a
# bare alias name instead of a literal model string (see `_cmd_import`).
_DEFAULT_IMPORT_MODEL = "anthropic/claude-sonnet-5"

# Deliberately more permissive than a strict slug (underscores and existing
# hyphens pass through untouched -- e.g. Spotify's own `bulk-reader` needs no
# normalization at all) but still lands on something commonadk's own rules
# actually require: a single path segment, lowercase, no leading dot (an
# agent folder starting with "." would be silently skipped by
# `loader._discover_agent_folders`), and not empty.
_AGENT_NAME_INVALID_CHARS_RE = re.compile(r"[^a-z0-9_-]+")
_AGENT_NAME_COLLAPSE_DASHES_RE = re.compile(r"-{2,}")


def _normalize_agent_name(raw: str) -> str:
    """Normalize an arbitrary skill name into a valid commonadk agent name.

    commonadk itself places no character constraint on `AgentConfig.name`
    beyond being a `str` (models.py) -- the only real constraints are
    structural: the folder name and `name:` must agree
    (`validation._check_folder_names`), names must be unique within a
    project (`loader.load`'s duplicate check), and a folder name is a single
    filesystem path segment that must not start with "." (dot-folders are
    silently skipped by `loader._discover_agent_folders`, so a name that
    normalized to one would vanish rather than error). This function picks
    one conservative, deterministic target shape satisfying all of that:
    lowercase, ASCII `[a-z0-9_-]`, no leading/trailing "-"/"_"/".", no
    run of repeated "-". Already-clean names (Spotify's `actions`,
    `bulk-reader`, ...) pass through byte-for-byte.
    """
    name = raw.strip().lower().replace(" ", "-")
    name = _AGENT_NAME_INVALID_CHARS_RE.sub("-", name)
    name = _AGENT_NAME_COLLAPSE_DASHES_RE.sub("-", name)
    name = name.strip("-_.")
    return name or "agent"


@dataclass(frozen=True)
class _SkillSource:
    """One discovered SKILL.md (or flat `*.md`) file, pre-parsed."""

    path: Path
    raw_name: str
    description: str


def _parse_skill_frontmatter(path: Path) -> tuple[Optional[str], str]:
    """Best-effort read of a skill file's `name`/`description` frontmatter.

    Reuses `loader._FRONTMATTER_RE`, the same anchored-at-start regex
    `loader._load_skill` matches against, so "what counts as frontmatter"
    can't drift between import-time discovery and load-time parsing. Never
    raises -- a missing, malformed, or non-mapping frontmatter block here
    just means "nothing to derive a name/description from"; `loader.load`
    is what actually enforces frontmatter validity once the file has been
    copied into the new agent folder (`_cmd_import` relies on the final
    `commonadk validate`-equivalent reload to surface a real problem, e.g. a
    frontmatter `name` that ends up disagreeing with the normalized agent
    name after all).
    """
    try:
        text = path.read_text()
    except OSError:
        return None, ""
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return None, ""
    try:
        frontmatter = yaml.safe_load(match.group("yaml"))
    except yaml.YAMLError:
        return None, ""
    if not isinstance(frontmatter, dict):
        return None, ""
    name = frontmatter.get("name")
    description = frontmatter.get("description")
    return (
        str(name) if isinstance(name, str) and name.strip() else None,
        str(description) if isinstance(description, str) else "",
    )


def _discover_skill_sources(skills_dir: Path) -> list[_SkillSource]:
    """Find every skill file under `skills_dir`, both supported layouts.

    Nested (Spotify's `portal-ai-plugins`): `<skills-dir>/<name>/SKILL.md`,
    one per direct subdirectory. Flat: `<skills-dir>/*.md`, directly inside
    `skills_dir` itself. Both are scanned unconditionally (not "detect one
    layout and use only that"), so a directory that happens to mix the two
    shapes still picks up every skill, deterministically ordered: nested
    skills first (subdirectories sorted by name), then flat files (sorted by
    name).
    """
    sources: list[_SkillSource] = []

    for sub in sorted(p for p in skills_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
        candidate = sub / "SKILL.md"
        if candidate.is_file():
            fm_name, fm_description = _parse_skill_frontmatter(candidate)
            sources.append(
                _SkillSource(path=candidate, raw_name=fm_name or sub.name, description=fm_description)
            )

    for md in sorted(p for p in skills_dir.iterdir() if p.is_file() and p.suffix.lower() == ".md"):
        fm_name, fm_description = _parse_skill_frontmatter(md)
        sources.append(_SkillSource(path=md, raw_name=fm_name or md.stem, description=fm_description))

    return sources


_FRONTMATTER_NAME_LINE_RE = re.compile(r"(?m)^name:[ \t]*.*$")


def _reconcile_skill_name(text: str, final_name: str) -> str:
    """Rewrite a copied skill file's frontmatter `name:` value to `final_name`.

    Only needed when name normalization actually changed the name (see
    `_normalize_agent_name`): loader.py's frontmatter handling (feature 1)
    requires a frontmatter `name`, when present, to agree with the agent's
    real name -- so a skill whose declared name isn't already a valid,
    unique commonadk agent name would otherwise make this importer's own
    output fail the "must pass `commonadk validate` immediately" contract.
    Every other byte (other frontmatter keys, the body) is left untouched --
    this is a targeted single-line substitution, not a full YAML
    re-serialization, specifically so it doesn't reformat quoting/spacing
    the original author chose. Returns `text` unchanged when there's no
    frontmatter, it declares no `name`, that `name` already matches
    `final_name`, or (defensively) the `name:` key isn't on its own simple
    `name: <value>` line (e.g. YAML flow-mapping syntax) -- in that last
    case the resulting name-mismatch error surfaces clearly on reload
    instead of this function risking a bad rewrite.
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return text
    try:
        frontmatter = yaml.safe_load(match.group("yaml"))
    except yaml.YAMLError:
        return text
    if not isinstance(frontmatter, dict):
        return text
    fm_name = frontmatter.get("name")
    if fm_name is None or str(fm_name) == final_name:
        return text

    frontmatter_block = match.group(0)
    new_block, n = _FRONTMATTER_NAME_LINE_RE.subn(f"name: {final_name}", frontmatter_block, count=1)
    if n == 0:
        return text
    return new_block + text[match.end() :]


def _version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("commonadk")
    except PackageNotFoundError:
        return "0.0.0+unknown"


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="commonadk",
        description=(
            "Define an agent system once, in a framework-neutral common/ "
            "folder, and build it on any supported agent SDK."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"commonadk {_version()}"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_p = subparsers.add_parser(
        "validate", help="Load and validate a common/ project folder"
    )
    validate_p.add_argument("common_dir", help="Path to the project's common/ folder")

    render_p = subparsers.add_parser(
        "render", help="Regenerate interaction-layer.md from interactions.yaml"
    )
    render_p.add_argument("common_dir", help="Path to the project's common/ folder")

    run_p = subparsers.add_parser(
        "run", help="Build an agent for a target SDK and run one turn"
    )
    run_p.add_argument("common_dir", help="Path to the project's common/ folder")
    run_p.add_argument(
        "--target",
        required=True,
        metavar="{google-adk,openai,claude,crewai,autogen,langgraph}",
        help="Agent SDK to build against: google-adk, openai, claude, crewai, autogen, or langgraph",
    )
    run_p.add_argument(
        "--agent",
        default=None,
        help="Agent to run (default: the project's entry agent)",
    )
    run_p.add_argument("prompt", help="The user message to send")

    new_p = subparsers.add_parser(
        "new", help="Scaffold a new agent folder inside an existing common/ project"
    )
    new_p.add_argument("common_dir", help="Path to the project's common/ folder")
    new_p.add_argument("agent_name", help="Name for the new agent (becomes its folder name)")
    new_p.add_argument(
        "--from",
        dest="from_agent",
        default=None,
        metavar="AGENT",
        help="Existing agent to add an outgoing edge from, into the new agent",
    )
    new_p.add_argument(
        "--type",
        dest="edge_type",
        choices=["delegate", "handoff"],
        default=None,
        metavar="{delegate,handoff}",
        help="Edge type for --from (default: delegate); requires --from",
    )

    import_p = subparsers.add_parser(
        "import",
        help="Import a directory of SKILL.md files into a new or existing common/ project",
    )
    import_p.add_argument(
        "skills_dir",
        help=(
            "Directory containing SKILL.md files: either nested "
            "<name>/SKILL.md (e.g. Spotify's portal-ai-plugins) or flat *.md"
        ),
    )
    import_p.add_argument("common_dir", help="Path to the common/ project folder to create or extend")
    import_p.add_argument(
        "--entry",
        default=None,
        metavar="NAME",
        help=(
            "Entry agent name (default: the alphabetically-first imported "
            "agent for a new project, or the existing project's own entry "
            "when extending one)"
        ),
    )
    import_p.add_argument(
        "--name",
        dest="project_name",
        default=None,
        metavar="PROJECT",
        help="Project name for a newly created config.yaml (default: the skills directory's own name)",
    )
    import_p.add_argument(
        "--model",
        default=None,
        metavar="ALIAS-OR-STRING",
        help=(
            "default_model for a newly created config.yaml: a literal "
            "'provider/model' LiteLLM string used as-is, or a bare alias "
            f"name defined to resolve to {_DEFAULT_IMPORT_MODEL!r} "
            f"(default: {_DEFAULT_IMPORT_MODEL!r} directly)"
        ),
    )

    return parser


# ---------------------------------------------------------------------------
# shared loading helper
# ---------------------------------------------------------------------------


def _load_project(common_dir: str) -> tuple["Project", list[warnings.WarningMessage]]:
    """`loader.load`, with every warning it raises captured instead of just
    printed to stderr by Python's default warning machinery -- so callers can
    fold them into the CLI's own output ("show them, don't swallow them")."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        project = load(common_dir)
    return project, list(caught)


def _print_warnings(caught: list[warnings.WarningMessage]) -> None:
    if not caught:
        return
    print("\nWarnings:")
    for w in caught:
        print(f"  - {w.message}")


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def _cmd_validate(common_dir: str) -> int:
    project, caught_warnings = _load_project(common_dir)

    lines: list[str] = []
    entry = project.config.entry or project.graph.entry
    lines.append(f"Project: {project.config.name}  (entry agent: {entry})")
    lines.append("")
    lines.append("Agents:")

    for name in sorted(project.agents):
        spec = project.agents[name]
        raw_model = spec.config.model or f"{project.config.default_model} (project default)"
        resolved_model = project.resolve_model(name)
        lines.append(f"  {name}")
        lines.append(f"    model: {raw_model} -> {resolved_model}")

        tool_names = ", ".join(sorted(t.name for t in spec.tools))
        lines.append(f"    tools: {tool_names or '(none)'}")

        env_reqs = spec.config.requires.env
        if not env_reqs:
            lines.append("    env: (none required)")
        else:
            lines.append("    env:")
            for req in env_reqs:
                is_set = bool(os.environ.get(req.name))
                status = "set" if is_set else "not set"
                required = "required" if req.required else "optional"
                desc = f" -- {req.description}" if req.description else ""
                lines.append(f"      {req.name}: {status} ({required}){desc}")

    print("\n".join(lines))
    _print_warnings(caught_warnings)
    return 0


def _cmd_render(common_dir: str) -> int:
    project, caught_warnings = _load_project(common_dir)
    out_path = write_interaction_layer(common_dir, project.graph)
    print(f"Wrote {out_path}")
    _print_warnings(caught_warnings)
    return 0


def _cmd_new(
    common_dir: str,
    agent_name: str,
    from_agent: Optional[str],
    edge_type: Optional[str],
) -> int:
    if edge_type is not None and from_agent is None:
        raise ValueError("commonadk: --type requires --from")

    common_path = Path(common_dir)
    agent_dir = common_path / agent_name
    if agent_dir.exists():
        raise ValueError(
            f"commonadk: refusing to overwrite existing agent folder "
            f"{agent_dir} -- choose a different name or remove it first"
        )

    # Load (and therefore validate) the project as it stands BEFORE
    # scaffolding anything -- this both fails loudly if the project is
    # already broken (matching every other command's behavior) and, when
    # --from is given, is how its agent name is checked against the real
    # agent list.
    project, _ = _load_project(common_dir)
    if from_agent is not None and from_agent not in project.agents:
        raise ValueError(
            f"commonadk: unknown --from agent {from_agent!r}. Known agents: "
            f"{sorted(project.agents)}"
        )

    agent_dir.mkdir(parents=True)
    (agent_dir / "skill.md").write_text(
        _NEW_AGENT_SKILL_MD.format(
            agent=agent_name, title=_title_from_agent_name(agent_name)
        )
    )
    (agent_dir / "tools.py").write_text(_NEW_AGENT_TOOLS_PY.format(agent=agent_name))
    (agent_dir / "agent-config.yaml").write_text(
        _NEW_AGENT_CONFIG_YAML.format(agent=agent_name)
    )

    created = [
        agent_dir / "skill.md",
        agent_dir / "tools.py",
        agent_dir / "agent-config.yaml",
    ]

    if from_agent is not None:
        interactions_path = common_path / "interactions.yaml"
        data = yaml.safe_load(interactions_path.read_text()) or {}
        data.setdefault("edges", []).append(
            {"from": from_agent, "to": agent_name, "type": edge_type or "delegate"}
        )
        interactions_path.write_text(yaml.safe_dump(data, sort_keys=False))
        created.append(interactions_path)

    # Reload the now-scaffolded project (never hand-edit the generated
    # interaction-layer.md -- regenerate it through the same renderer
    # `commonadk render` uses) and surface any warnings, matching every
    # other command's pattern.
    project, caught_warnings = _load_project(common_dir)
    if from_agent is not None:
        out_path = write_interaction_layer(common_dir, project.graph)
        created.append(out_path)

    print(f"Created agent '{agent_name}' in {agent_dir}:")
    for path in created:
        print(f"  {path}")
    _print_warnings(caught_warnings)
    return 0


def _cmd_import(
    skills_dir: str,
    common_dir: str,
    entry: Optional[str],
    project_name: Optional[str],
    model: Optional[str],
) -> int:
    """Turn a directory of SKILL.md files into a conforming `common/` project.

    One agent folder per discovered skill (see `_discover_skill_sources` for
    the two supported layouts): `skill.md` is the original file, copied
    byte-for-byte (frontmatter included); `agent-config.yaml` gets `name` +
    `description` derived from that frontmatter (`AgentConfig.description`
    defaults to `""`, so a skill with no frontmatter `description` just gets
    one) and an empty `tools: []` (a real, valid commonadk agent -- verified
    directly: `validation._check_tools` only ever iterates
    `agent-config.yaml`'s `tools:` list, so an empty one is never checked
    against `tools.py` at all); `tools.py` is a docstring-only stub (no
    functions -- nothing in `loader._load_tools` requires any).

    **Name normalization** (`_normalize_agent_name`): a skill's raw name
    (its frontmatter `name`, else its folder/file name) is normalized into
    a valid, unique agent name; collisions (after normalization, or against
    an existing project's own agents in extend mode) are resolved with a
    deterministic `-2`, `-3`, ... suffix. Every mapping is printed, even
    identity ones, so the caller can see exactly what commonadk imported.

    **New project vs. extend** (the refuse-to-clobber rule): if `common_dir`
    doesn't exist, or exists and is empty, this scaffolds a brand-new
    project -- `config.yaml` (`--name`/`--model`-derived) and
    `interactions.yaml` (`entry:` only, `edges: []` -- imported skills carry
    no declared relationship to each other, so none is invented) are written
    alongside the agent folders. If `common_dir` exists and is NOT empty,
    this refuses to touch it UNLESS it already loads as a valid commonadk
    project (`loader.load` succeeds) -- in which case it's extended: new
    agent folders are added (never overwriting an existing one -- a name
    collision is resolved by the same normalization-suffix logic used
    between two freshly-imported skills), `config.yaml`/`interactions.yaml`
    are left alone (entry included, unless `--entry` explicitly asks for a
    different one), and only `interaction-layer.md` is regenerated at the
    end, same as every other command that changes the graph. A non-empty
    `common_dir` that does NOT load as a valid project is refused outright
    -- writing into a directory this tool doesn't understand risks silently
    corrupting whatever's actually there.

    Either way, the output is reloaded through the same `loader.load` path
    `commonadk validate` uses before this function returns, so a caller
    immediately running `commonadk validate` on the result is checking
    something this function has already checked once itself.
    """
    skills_path = Path(skills_dir)
    if not skills_path.is_dir():
        raise ValueError(f"commonadk: skills directory not found: {skills_path}")

    sources = _discover_skill_sources(skills_path)
    if not sources:
        raise ValueError(
            f"commonadk: no SKILL.md files found under {skills_path} (expected "
            f"either <skills-dir>/<name>/SKILL.md or <skills-dir>/*.md)"
        )

    common_path = Path(common_dir)
    existing_project: Optional["Project"] = None
    if common_path.exists():
        if not common_path.is_dir():
            raise ValueError(f"commonadk: {common_path} exists and is not a directory")
        if any(common_path.iterdir()):
            try:
                existing_project, _ = _load_project(common_path)
            except ValidationError as e:
                raise ValueError(
                    f"commonadk: refusing to import into {common_path} -- it "
                    f"already exists, is not empty, and is not a valid "
                    f"commonadk project ({e}). Point common-dir at an empty "
                    f"or nonexistent directory to create a new project, or "
                    f"at an existing valid commonadk project to extend it."
                ) from e

    # Normalize names, deterministically resolving collisions -- both among
    # the newly discovered skills and against any pre-existing project
    # agents (extend mode) -- with a "-2", "-3", ... suffix.
    taken = set(existing_project.agents) if existing_project is not None else set()
    imports: list[tuple[_SkillSource, str, str]] = []  # (source, raw_name, final_name)
    for source in sources:
        final_name = _normalize_agent_name(source.raw_name)
        candidate = final_name
        suffix = 2
        while candidate in taken:
            candidate = f"{final_name}-{suffix}"
            suffix += 1
        taken.add(candidate)
        imports.append((source, source.raw_name, candidate))

    imported_names = sorted(final for _, _, final in imports)
    existing_entry = (
        (existing_project.config.entry or existing_project.graph.entry)
        if existing_project is not None
        else None
    )

    if entry is not None:
        if entry not in taken:
            raise ValueError(
                f"commonadk: --entry {entry!r} does not name an imported agent "
                f"or an existing project agent. Imported: {imported_names}"
                + (
                    f"; existing: {sorted(existing_project.agents)}"
                    if existing_project is not None
                    else ""
                )
            )
        chosen_entry = entry
        entry_chosen_automatically = False
    elif existing_entry is not None:
        chosen_entry = existing_entry
        entry_chosen_automatically = False
    else:
        chosen_entry = imported_names[0]
        entry_chosen_automatically = True

    common_path.mkdir(parents=True, exist_ok=True)

    created: list[Path] = []
    renamed_frontmatter: list[str] = []
    for source, _raw_name, final_name in imports:
        agent_dir = common_path / final_name
        agent_dir.mkdir(parents=True)  # exist_ok=False -- see docstring: never clobber
        original_text = source.path.read_text()
        skill_text = _reconcile_skill_name(original_text, final_name)
        if skill_text != original_text:
            renamed_frontmatter.append(final_name)
        (agent_dir / "skill.md").write_text(skill_text)
        (agent_dir / "agent-config.yaml").write_text(
            yaml.safe_dump(
                {
                    "name": final_name,
                    "description": source.description,
                    "tools": [],
                    "requires": {"env": []},
                },
                sort_keys=False,
            )
        )
        (agent_dir / "tools.py").write_text(_IMPORTED_AGENT_TOOLS_PY.format(agent=final_name))
        created.extend(
            [
                agent_dir / "skill.md",
                agent_dir / "agent-config.yaml",
                agent_dir / "tools.py",
            ]
        )

    if existing_project is None:
        model_arg = model or _DEFAULT_IMPORT_MODEL
        if "/" in model_arg:
            default_model, model_aliases = model_arg, {}
        else:
            default_model, model_aliases = model_arg, {model_arg: _DEFAULT_IMPORT_MODEL}

        config_path = common_path / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "name": project_name or skills_path.resolve().name,
                    "entry": chosen_entry,
                    "targets": [],
                    "default_model": default_model,
                    "model_aliases": model_aliases,
                },
                sort_keys=False,
            )
        )
        created.append(config_path)

        interactions_path = common_path / "interactions.yaml"
        interactions_path.write_text(
            yaml.safe_dump({"entry": chosen_entry, "edges": []}, sort_keys=False)
        )
        created.append(interactions_path)
    elif entry is not None and entry != existing_entry:
        # Extending, and the caller explicitly asked for a different entry
        # than the project already has -- honor it, in both source-of-truth
        # files (config.yaml and interactions.yaml), same pair
        # `validation._check_entry` requires to agree.
        config_path = common_path / "config.yaml"
        config_data = yaml.safe_load(config_path.read_text()) or {}
        config_data["entry"] = entry
        config_path.write_text(yaml.safe_dump(config_data, sort_keys=False))

        interactions_path = common_path / "interactions.yaml"
        interactions_data = yaml.safe_load(interactions_path.read_text()) or {}
        interactions_data["entry"] = entry
        interactions_path.write_text(yaml.safe_dump(interactions_data, sort_keys=False))

    # Reload the now-scaffolded/extended project (this is what makes
    # `commonadk validate common_dir` immediately after this command a
    # no-op check -- the exact same load already ran here) and regenerate
    # interaction-layer.md through the same renderer `commonadk render`
    # uses, never hand-edited.
    project, caught_warnings = _load_project(common_path)
    out_path = write_interaction_layer(common_path, project.graph)
    created.append(out_path)

    verb = "Extended" if existing_project is not None else "Created"
    print(f"{verb} {common_path} with {len(imports)} imported skill(s):")
    for source, raw_name, final_name in imports:
        renamed = f"  (from '{raw_name}')" if raw_name != final_name else ""
        fm_note = " [frontmatter name rewritten to match]" if final_name in renamed_frontmatter else ""
        print(f"  {source.path} -> {final_name}{renamed}{fm_note}")
    print(
        f"Entry agent: {chosen_entry}"
        + (" (chosen automatically)" if entry_chosen_automatically else "")
    )
    for path in created:
        print(f"  {path}")
    _print_warnings(caught_warnings)
    return 0


def _cmd_run(common_dir: str, target: str, agent: Optional[str], prompt: str) -> int:
    project, caught_warnings = _load_project(common_dir)
    _print_warnings(caught_warnings)

    agent_name = agent or project.config.entry or project.graph.entry
    if agent_name is None:
        raise ValueError(
            "commonadk: no --agent given and the project has no entry agent"
        )
    if agent_name not in project.agents:
        raise ValueError(
            f"commonadk: unknown agent {agent_name!r}. Known agents: "
            f"{sorted(project.agents)}"
        )

    runner = _RUN_TARGETS.get(target)
    if runner is None:
        # Not one of this CLI's known targets -- delegate to the adapter
        # registry purely for its "unknown target, known targets are ..."
        # error message, so the CLI never hand-maintains a second list of
        # valid targets that can drift from adapters/__init__.py's.
        from .adapters import get_adapter

        get_adapter(target)
        raise AssertionError(f"unreachable: get_adapter({target!r}) did not raise")

    output = runner(project, agent_name, prompt)
    print(output)
    return 0


# -- per-target execution (SDK imports are lazy, inside these functions) ----


def _run_google_adk(project: "Project", agent_name: str, prompt: str) -> str:
    import asyncio

    from google.adk.runners import InMemoryRunner
    from google.genai import types as genai_types

    agent = project.build(agent_name, target="google-adk")
    runner = InMemoryRunner(agent=agent, app_name=project.config.name)
    user_id = "commonadk-cli"

    async def _invoke() -> str:
        session = await runner.session_service.create_session(
            app_name=runner.app_name, user_id=user_id
        )
        message = genai_types.Content(
            role="user", parts=[genai_types.Part(text=prompt)]
        )
        chunks: list[str] = []
        async for event in runner.run_async(
            user_id=user_id, session_id=session.id, new_message=message
        ):
            if event.is_final_response() and event.content and event.content.parts:
                chunks.extend(
                    part.text
                    for part in event.content.parts
                    if getattr(part, "text", None)
                )
        return "\n".join(chunks)

    return asyncio.run(_invoke())


def _run_openai(project: "Project", agent_name: str, prompt: str) -> str:
    from agents import Runner

    agent = project.build(agent_name, target="openai")
    result = Runner.run_sync(agent, prompt)
    return str(result.final_output)


def _run_claude(project: "Project", agent_name: str, prompt: str) -> str:
    import asyncio

    from claude_agent_sdk import ResultMessage, query

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise OSError(
            "commonadk: missing required environment variable for target "
            "'claude': ANTHROPIC_API_KEY (the Claude Agent SDK's bundled "
            "CLI needs it to authenticate with the Anthropic API)"
        )

    options = project.build(agent_name, target="claude")

    async def _invoke() -> str:
        chunks: list[str] = []
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, ResultMessage) and message.result:
                chunks.append(message.result)
        return "\n".join(chunks)

    return asyncio.run(_invoke())


def _run_crewai(project: "Project", agent_name: str, prompt: str) -> str:
    from crewai import Process, Task

    crew = project.build(agent_name, target="crewai")

    # Hierarchical crews (build root has outgoing edges): leave the task
    # unassigned -- the manager agent (the build root) picks which crew
    # member actually executes it. Sequential crews here are always the
    # solo-member fallback (see crewai_adapter.py's module docstring,
    # "Manager-or-solo-member decision"), so the one member must be
    # assigned explicitly.
    agent = None if crew.process == Process.hierarchical else crew.agents[0]
    crew.tasks = [
        Task(
            description=prompt,
            expected_output="A complete response to the request above.",
            agent=agent,
        )
    ]
    result = crew.kickoff()
    return str(result.raw)


def _run_autogen(project: "Project", agent_name: str, prompt: str) -> str:
    import asyncio

    built = project.build(agent_name, target="autogen")

    # `built` is either a bare `AssistantAgent` (build root has no outgoing
    # edges -- see autogen_adapter.py's module docstring, "WHAT build()
    # RETURNS") or a ready-to-run `Swarm` team (build root has at least one
    # outgoing edge). Both expose the same `async .run(task=...) ->
    # TaskResult` shape, so no branching on the return type is needed here.
    result = asyncio.run(built.run(task=prompt))
    return str(result.messages[-1].content)


def _run_langgraph(project: "Project", agent_name: str, prompt: str) -> str:
    graph = project.build(agent_name, target="langgraph")

    # `graph` is either a lone react agent's own `CompiledStateGraph` (build
    # root has no outgoing edges) or the compiled multi-agent `StateGraph`
    # (build root has at least one) -- see langgraph_adapter.py's module
    # docstring, "WHAT build() RETURNS". Both expose the same `.invoke(...)`
    # shape over a `MessagesState`-style input, so no branching on the
    # return type is needed here.
    result = graph.invoke({"messages": [{"role": "user", "content": prompt}]})
    return str(result["messages"][-1].content)


_RUN_TARGETS = {
    "google-adk": _run_google_adk,
    "openai": _run_openai,
    "claude": _run_claude,
    "crewai": _run_crewai,
    "autogen": _run_autogen,
    "langgraph": _run_langgraph,
}


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "validate":
            return _cmd_validate(args.common_dir)
        if args.command == "render":
            return _cmd_render(args.common_dir)
        if args.command == "run":
            return _cmd_run(args.common_dir, args.target, args.agent, args.prompt)
        if args.command == "new":
            return _cmd_new(
                args.common_dir, args.agent_name, args.from_agent, args.edge_type
            )
        if args.command == "import":
            return _cmd_import(
                args.skills_dir, args.common_dir, args.entry, args.project_name, args.model
            )
        parser.print_help()
        return 1
    except ValidationError as e:
        print(str(e), file=sys.stderr)
        return 1
    except (OSError, ValueError, ImportError) as e:
        print(f"commonadk: {e}" if not str(e).startswith("commonadk") else str(e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
