"""Parses a `common/` project folder into a validated `Project`.

Loading is best-effort in the collection phase: every file that can be parsed
is parsed, and every problem found (missing file, bad YAML, unknown tool,
untyped parameter, ...) is accumulated rather than raised immediately, so
`ValidationError` can report everything wrong with the project in one shot.
"""

from __future__ import annotations

import importlib.util
import inspect
import re
import warnings as _warnings
from pathlib import Path
from typing import Optional, Union

import yaml
from pydantic import ValidationError as PydanticValidationError

from .models import AgentConfig, AgentSpec, InteractionGraph, Project, ProjectConfig, ToolSpec
from .validation import ValidationError, validate

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(?P<yaml>.*?\n)---\s*\n?", re.DOTALL)


def load(path: Union[str, Path]) -> Project:
    """Load and validate a `common/` project folder.

    Raises `commonadk.validation.ValidationError` (with `.errors` listing
    every problem found) if the project is malformed in any way. Non-fatal
    issues (e.g. a tool missing a return type hint) are emitted as Python
    warnings, not raised.
    """
    root = Path(path)
    errors: list[str] = []

    if not root.is_dir():
        raise ValidationError([f"project folder not found: {root}"])

    project_config = _load_project_config(root, errors)
    graph = _load_interactions(root, errors)

    agent_configs: dict[str, AgentConfig] = {}
    agent_instructions: dict[str, str] = {}
    agent_tools: dict[str, dict[str, ToolSpec]] = {}
    agent_folder_names: dict[str, str] = {}
    skill_warnings: list[str] = []

    for folder in _discover_agent_folders(root):
        cfg = _load_agent_config(folder, errors)
        if cfg is None:
            continue
        if cfg.name in agent_configs:
            errors.append(
                f"duplicate agent name '{cfg.name}': already defined by "
                f"another folder"
            )
            continue
        agent_folder_names[cfg.name] = folder.name
        agent_configs[cfg.name] = cfg
        # `_load_skill` may mutate `cfg.description` in place (frontmatter
        # fallback -- see its docstring), so it must run before `cfg` is
        # handed to `validate()` below.
        agent_instructions[cfg.name] = _load_skill(folder, cfg, errors, skill_warnings)
        agent_tools[cfg.name] = _load_tools(folder, cfg.name, errors)

    val_errors, val_warnings = validate(
        project_config=project_config,
        graph=graph,
        agent_configs=agent_configs,
        agent_tools=agent_tools,
        agent_folder_names=agent_folder_names,
    )
    errors.extend(val_errors)

    if errors:
        raise ValidationError(errors)

    for message in [*skill_warnings, *val_warnings]:
        _warnings.warn(message, stacklevel=2)

    assert project_config is not None  # guaranteed: no errors were raised

    agents: dict[str, AgentSpec] = {}
    for name, cfg in agent_configs.items():
        tools_by_name = agent_tools.get(name, {})
        tool_specs = [
            tools_by_name[tool_name]
            for tool_name in cfg.tools
            if tool_name in tools_by_name
        ]
        agents[name] = AgentSpec(
            config=cfg,
            instructions=agent_instructions.get(name, ""),
            tools=tool_specs,
        )

    return Project(
        config=project_config,
        agents=agents,
        graph=graph if graph is not None else InteractionGraph(),
    )


def _format_pydantic_error(exc: Exception) -> str:
    if isinstance(exc, PydanticValidationError):
        parts = []
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"]) or "<root>"
            detail = f"{loc}: {err['msg']}"
            if "input" in err and err["input"] not in (None, ""):
                detail += f" (got: {err['input']!r})"
            parts.append(detail)
        return "; ".join(parts)
    return str(exc)


def _load_yaml(path: Path) -> Optional[dict]:
    return yaml.safe_load(path.read_text()) or {}


def _load_project_config(root: Path, errors: list[str]) -> Optional[ProjectConfig]:
    path = root / "config.yaml"
    if not path.is_file():
        errors.append(f"config.yaml not found (expected at {path})")
        return None
    try:
        data = _load_yaml(path)
    except yaml.YAMLError as e:
        errors.append(f"config.yaml: invalid YAML: {e}")
        return None
    try:
        return ProjectConfig.model_validate(data)
    except PydanticValidationError as e:
        errors.append(f"config.yaml: {_format_pydantic_error(e)}")
        return None


def _load_interactions(root: Path, errors: list[str]) -> Optional[InteractionGraph]:
    path = root / "interactions.yaml"
    if not path.is_file():
        errors.append(f"interactions.yaml not found (expected at {path})")
        return None
    try:
        data = _load_yaml(path)
    except yaml.YAMLError as e:
        errors.append(f"interactions.yaml: invalid YAML: {e}")
        return None
    try:
        return InteractionGraph.model_validate(data)
    except PydanticValidationError as e:
        errors.append(f"interactions.yaml: {_format_pydantic_error(e)}")
        return None


def _discover_agent_folders(root: Path) -> list[Path]:
    return sorted(
        (p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")),
        key=lambda p: p.name,
    )


def _load_agent_config(folder: Path, errors: list[str]) -> Optional[AgentConfig]:
    path = folder / "agent-config.yaml"
    if not path.is_file():
        errors.append(f"{folder.name}/agent-config.yaml not found")
        return None
    try:
        data = _load_yaml(path)
    except yaml.YAMLError as e:
        errors.append(f"{folder.name}/agent-config.yaml: invalid YAML: {e}")
        return None
    try:
        return AgentConfig.model_validate(data)
    except PydanticValidationError as e:
        errors.append(f"{folder.name}/agent-config.yaml: {_format_pydantic_error(e)}")
        return None


def _load_skill(
    folder: Path,
    cfg: AgentConfig,
    errors: list[str],
    warnings_out: list[str],
) -> str:
    """Read `skill.md` and return its body as agent instructions.

    An optional YAML frontmatter block (`---`, YAML, `---`) is recognized
    only when it's the very first thing in the file (`_FRONTMATTER_RE`,
    anchored with `\\A`, applied once) -- a `---`-delimited block anywhere
    else in the file is left untouched, treated as ordinary Markdown. A
    `skill.md` with no frontmatter returns exactly `text.strip()`, same as
    before this feature existed (back-compat contract, asserted directly by
    `tests/test_loader.py::test_skill_md_without_frontmatter_is_unchanged`).

    When frontmatter IS present, it's parsed and reconciled against `cfg`
    (this agent's already-loaded `AgentConfig`) instead of being discarded --
    see docs/file-contracts.md, "skill.md -- frontmatter", for the full
    rules. In short:

    - `agent-config.yaml` is always authoritative. Frontmatter supplies a
      value only for a field `agent-config.yaml` left unset.
    - `name`, if present, must agree with `cfg.name` (the same value the
      folder-vs-name check in validation.py checks against the folder) --
      disagreement is an error, in the same style as that check.
    - `description`, if present and `cfg.description` is unset (`""`), is
      adopted onto `cfg` in place (so every downstream consumer -- adapters
      included -- sees one reconciled value, not two). If both are set and
      differ, `cfg.description` wins and a warning is recorded.
    - Any other frontmatter key is warned about, not errored. This is the
      one deliberate asymmetry with every other `common/` YAML file (all of
      which use `extra="forbid"`, see models.py): skill.md's frontmatter is
      a surface other agent-SDK hosts (e.g. Claude Code, Codex, Cursor --
      see Spotify's `portal-ai-plugins` for a real example) read and extend
      too, so a key commonadk doesn't recognize may still be meaningful to
      one of them and shouldn't block a commonadk load.
    - Malformed frontmatter YAML, or frontmatter that doesn't parse to a
      mapping, is an error naming the file.
    """
    agent_name = cfg.name
    path = folder / "skill.md"
    if not path.is_file():
        errors.append(f"{agent_name}/skill.md not found")
        return ""

    text = path.read_text()
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return text.strip()

    body = text[match.end() :].strip()

    try:
        frontmatter = yaml.safe_load(match.group("yaml"))
    except yaml.YAMLError as e:
        errors.append(f"{agent_name}/skill.md: invalid frontmatter YAML: {e}")
        return body

    if frontmatter is None:
        frontmatter = {}
    if not isinstance(frontmatter, dict):
        errors.append(
            f"{agent_name}/skill.md: frontmatter must be a YAML mapping "
            f"(got {type(frontmatter).__name__})"
        )
        return body

    fm_name = frontmatter.pop("name", None)
    fm_description = frontmatter.pop("description", None)

    if fm_name is not None and fm_name != agent_name:
        errors.append(
            f"{agent_name}/skill.md: frontmatter declares name '{fm_name}', "
            f"which does not match agent-config.yaml's name '{agent_name}' "
            f"(also the agent's folder name) -- skill.md frontmatter's "
            f"`name` and agent-config.yaml's `name` must agree"
        )

    if fm_description:
        if not cfg.description:
            cfg.description = fm_description
        elif cfg.description != fm_description:
            warnings_out.append(
                f"{agent_name}/skill.md: frontmatter `description` differs "
                f"from agent-config.yaml's `description` -- agent-config.yaml "
                f"wins ({cfg.description!r} kept over {fm_description!r})"
            )

    for key in sorted(frontmatter):
        warnings_out.append(
            f"{agent_name}/skill.md: unrecognized frontmatter key '{key}' -- "
            f"ignored (skill.md frontmatter is a shared surface other "
            f"agent-SDK hosts may add their own keys to; unlike commonadk's "
            f"own YAML files, unknown frontmatter keys are warnings, not "
            f"errors)"
        )

    return body


def _load_tools(folder: Path, agent_name: str, errors: list[str]) -> dict[str, ToolSpec]:
    path = folder / "tools.py"
    if not path.is_file():
        errors.append(f"{agent_name}/tools.py not found")
        return {}

    module_name = f"commonadk._loaded_tools.{folder.name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        errors.append(f"{agent_name}/tools.py: could not create module spec")
        return {}

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:  # noqa: BLE001 - surfaced as a validation error
        errors.append(f"{agent_name}/tools.py: error while importing: {e!r}")
        return {}

    tools: dict[str, ToolSpec] = {}
    for attr_name, obj in vars(module).items():
        if (
            inspect.isfunction(obj)
            and obj.__module__ == module_name
            and not attr_name.startswith("_")
        ):
            tools[attr_name] = ToolSpec.from_function(obj)
    return tools
