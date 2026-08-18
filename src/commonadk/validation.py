"""Cross-cutting checks over an assembled (but not-yet-wrapped) project.

All checks run and every problem is collected before anything is raised --
`ValidationError` always lists every error found in one pass, never just the
first. Errors block loading; warnings (e.g. a missing-but-recommended return
type hint) are surfaced via the `warnings` module but do not fail the load.
"""

from __future__ import annotations

from typing import Optional

from .models import AgentConfig, InteractionGraph, ProjectConfig, ToolSpec


class ValidationError(Exception):
    """Raised by `commonadk.load` when one or more validation checks fail.

    `.errors` holds every individual error message so callers/tests can
    inspect them without parsing the exception string.
    """

    def __init__(self, errors: list[str]):
        self.errors = list(errors)
        header = f"commonadk validation failed with {len(self.errors)} error(s):"
        body = "\n".join(f"  - {e}" for e in self.errors)
        super().__init__(f"{header}\n{body}")


def validate(
    *,
    project_config: Optional[ProjectConfig],
    graph: Optional[InteractionGraph],
    agent_configs: dict[str, AgentConfig],
    agent_tools: dict[str, dict[str, ToolSpec]],
    agent_folder_names: dict[str, str],
) -> tuple[list[str], list[str]]:
    """Run every cross-cutting check. Returns (errors, warnings)."""
    errors: list[str] = []
    warnings: list[str] = []

    _check_tools(agent_configs, agent_tools, errors, warnings)
    _check_folder_names(agent_folder_names, errors)
    if graph is not None:
        _check_edges(graph, agent_configs, errors)
    _check_entry(project_config, graph, agent_configs, errors)
    if project_config is not None:
        _check_models(project_config, agent_configs, errors)
    _check_runtime(agent_configs, warnings)

    return errors, warnings


def _check_runtime(
    agent_configs: dict[str, AgentConfig],
    warnings: list[str],
) -> None:
    # `runtime:` is reserved for future mixed-target spawning (plan.md,
    # "Deferred / roadmap"). It is not honored in v1 -- every agent builds
    # under the single target passed to project.build() -- so setting it is
    # a warning, not an error.
    for agent_name, cfg in agent_configs.items():
        if cfg.runtime is not None:
            warnings.append(
                f"{agent_name}: `runtime:` is reserved for future mixed-target "
                f"spawning and is not yet honored -- the agent will build "
                f"under the target passed to project.build()"
            )


def _check_tools(
    agent_configs: dict[str, AgentConfig],
    agent_tools: dict[str, dict[str, ToolSpec]],
    errors: list[str],
    warnings: list[str],
) -> None:
    for agent_name, cfg in agent_configs.items():
        tools_in_module = agent_tools.get(agent_name, {})
        for tool_name in cfg.tools:
            spec = tools_in_module.get(tool_name)
            if spec is None:
                available = sorted(tools_in_module) or ["<none>"]
                errors.append(
                    f"{agent_name}: tool '{tool_name}' is listed in "
                    f"agent-config.yaml but is not defined as a function in "
                    f"{agent_name}/tools.py (available: {available})"
                )
                continue
            if not spec.has_docstring:
                errors.append(
                    f"{agent_name}/tools.py: tool '{tool_name}' is missing a "
                    f"docstring (required)"
                )
            if not spec.fully_typed:
                errors.append(
                    f"{agent_name}/tools.py: tool '{tool_name}' has one or "
                    f"more parameters without type hints (required)"
                )
            if spec.return_type is None:
                warnings.append(
                    f"{agent_name}/tools.py: tool '{tool_name}' has no return "
                    f"type hint (recommended)"
                )


def _check_folder_names(
    agent_folder_names: dict[str, str], errors: list[str]
) -> None:
    for config_name, folder_name in agent_folder_names.items():
        if config_name != folder_name:
            errors.append(
                f"agent folder '{folder_name}' declares name '{config_name}' "
                f"in agent-config.yaml; the folder name and the `name` field "
                f"must match"
            )


def _check_edges(
    graph: InteractionGraph,
    agent_configs: dict[str, AgentConfig],
    errors: list[str],
) -> None:
    for i, edge in enumerate(graph.edges):
        if edge.from_ not in agent_configs:
            errors.append(
                f"interactions.yaml: edge #{i} has unknown 'from' agent "
                f"'{edge.from_}' (known agents: {sorted(agent_configs)})"
            )
        if edge.to not in agent_configs:
            errors.append(
                f"interactions.yaml: edge #{i} has unknown 'to' agent "
                f"'{edge.to}' (known agents: {sorted(agent_configs)})"
            )
        # edge.type is a Literal["delegate", "handoff"] enforced by pydantic
        # at parse time, so an invalid type never reaches this point -- it is
        # reported as a YAML/model error where interactions.yaml is parsed.


def _check_entry(
    project_config: Optional[ProjectConfig],
    graph: Optional[InteractionGraph],
    agent_configs: dict[str, AgentConfig],
    errors: list[str],
) -> None:
    config_entry = project_config.entry if project_config else None
    graph_entry = graph.entry if graph else None

    if config_entry and config_entry not in agent_configs:
        errors.append(
            f"config.yaml: entry agent '{config_entry}' does not exist "
            f"(known agents: {sorted(agent_configs)})"
        )
    if graph_entry and graph_entry not in agent_configs:
        errors.append(
            f"interactions.yaml: entry agent '{graph_entry}' does not exist "
            f"(known agents: {sorted(agent_configs)})"
        )
    if config_entry and graph_entry and config_entry != graph_entry:
        errors.append(
            f"entry agent mismatch: config.yaml says '{config_entry}' but "
            f"interactions.yaml says '{graph_entry}'"
        )
    if not config_entry and not graph_entry:
        errors.append(
            "no entry agent specified: set `entry` in config.yaml and/or "
            "interactions.yaml"
        )


def _check_models(
    project_config: ProjectConfig,
    agent_configs: dict[str, AgentConfig],
    errors: list[str],
) -> None:
    def resolvable(raw: str) -> bool:
        return "/" in raw or raw in project_config.model_aliases

    if not resolvable(project_config.default_model):
        errors.append(
            f"config.yaml: default_model '{project_config.default_model}' is "
            f"not a LiteLLM-format string and is not defined in "
            f"model_aliases (known aliases: {sorted(project_config.model_aliases)})"
        )

    for agent_name, cfg in agent_configs.items():
        raw = cfg.model or project_config.default_model
        if not resolvable(raw):
            errors.append(
                f"{agent_name}: model alias '{raw}' is not defined in "
                f"config.yaml model_aliases (known aliases: "
                f"{sorted(project_config.model_aliases)})"
            )
