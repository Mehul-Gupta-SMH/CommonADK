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

_FRONTMATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*\n?", re.DOTALL)


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
        agent_instructions[cfg.name] = _load_skill(folder, cfg.name, errors)
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

    for message in val_warnings:
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


def _load_skill(folder: Path, agent_name: str, errors: list[str]) -> str:
    path = folder / "skill.md"
    if not path.is_file():
        errors.append(f"{agent_name}/skill.md not found")
        return ""
    text = path.read_text()
    text = _FRONTMATTER_RE.sub("", text, count=1)
    return text.strip()


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
