"""Framework-neutral Pydantic models mirroring the `common/` file contracts.

See plan.md ("File contracts") for the authoritative shape of each YAML file.
These models hold the *parsed and resolved* representation that loader.py
builds and validation.py checks -- adapters (M2+) consume them directly and
never touch the raw YAML.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class EnvRequirement(BaseModel):
    """A single runtime environment variable an agent needs (name only, never a value)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    required: bool = True


class Requires(BaseModel):
    """The `requires:` block of an agent-config.yaml."""

    model_config = ConfigDict(extra="forbid")

    env: list[EnvRequirement] = Field(default_factory=list)


class ProjectConfig(BaseModel):
    """Parsed `common/config.yaml`."""

    model_config = ConfigDict(extra="forbid")

    name: str
    entry: Optional[str] = None
    targets: list[str] = Field(default_factory=list)
    default_model: str
    model_aliases: dict[str, str] = Field(default_factory=dict)


class AgentConfig(BaseModel):
    """Parsed `common/<agent>/agent-config.yaml`."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    model: Optional[str] = None
    model_params: dict[str, Any] = Field(default_factory=dict)
    tools: list[str] = Field(default_factory=list)
    requires: Requires = Field(default_factory=Requires)
    targets: dict[str, dict[str, Any]] = Field(default_factory=dict)
    runtime: Optional[str] = None
    """Reserved for future mixed-target spawning (plan.md "Deferred / roadmap").

    Unset in v1. If set, validation emits a warning that it is not yet
    honored -- `project.build(..., target=...)` still builds every agent
    under the single target passed to it.
    """


class ToolParameter(BaseModel):
    """One parameter of a tool function, derived from its signature."""

    name: str
    type: str
    required: bool
    default: Optional[str] = None


class ToolSpec(BaseModel):
    """A single tool: the live callable plus schema metadata derived from it.

    Type hints and a docstring are required on every tool function -- that is
    what every downstream SDK adapter turns into a tool schema. Validation of
    that requirement lives in validation.py; this model just stores what
    inspect/typing can determine.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str = ""
    func: Callable[..., Any]
    parameters: list[ToolParameter] = Field(default_factory=list)
    return_type: Optional[str] = None
    has_docstring: bool = False
    fully_typed: bool = False

    @classmethod
    def from_function(cls, func: Callable[..., Any]) -> "ToolSpec":
        """Build a ToolSpec by introspecting a plain Python function."""
        name = func.__name__
        doc = inspect.getdoc(func) or ""
        sig = inspect.signature(func)

        try:
            hints = inspect.get_annotations(func, eval_str=True)
        except Exception:
            hints = dict(getattr(func, "__annotations__", {}))

        parameters: list[ToolParameter] = []
        fully_typed = True
        for pname, param in sig.parameters.items():
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            annotation = hints.get(pname, inspect.Parameter.empty)
            if annotation is inspect.Parameter.empty:
                fully_typed = False
                type_str = "Any"
            else:
                type_str = getattr(annotation, "__name__", str(annotation))
            has_default = param.default is not inspect.Parameter.empty
            parameters.append(
                ToolParameter(
                    name=pname,
                    type=type_str,
                    required=not has_default,
                    default=repr(param.default) if has_default else None,
                )
            )

        return_annotation = hints.get("return", inspect.Signature.empty)
        return_type = (
            None
            if return_annotation is inspect.Signature.empty
            else getattr(return_annotation, "__name__", str(return_annotation))
        )

        return cls(
            name=name,
            description=doc,
            func=func,
            parameters=parameters,
            return_type=return_type,
            has_docstring=bool(doc.strip()),
            fully_typed=fully_typed,
        )


class InteractionEdge(BaseModel):
    """One edge in `common/interactions.yaml`."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    from_: str = Field(alias="from")
    to: str
    type: Literal["delegate", "handoff"]


class InteractionGraph(BaseModel):
    """Parsed `common/interactions.yaml`."""

    model_config = ConfigDict(extra="forbid")

    entry: Optional[str] = None
    edges: list[InteractionEdge] = Field(default_factory=list)


class AgentSpec(BaseModel):
    """A fully-resolved agent: its config, instructions text, and live tools."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: AgentConfig
    instructions: str = ""
    tools: list[ToolSpec] = Field(default_factory=list)

    @property
    def name(self) -> str:
        return self.config.name


class Project(BaseModel):
    """The fully loaded and validated project: config + agents + interaction graph."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: ProjectConfig
    agents: dict[str, AgentSpec] = Field(default_factory=dict)
    graph: InteractionGraph = Field(default_factory=InteractionGraph)

    def resolve_model(self, agent_name: str) -> str:
        """Resolve the LiteLLM-format model string an agent should use.

        Resolution order: the agent's own `model` (alias or literal string) if
        set, else the project's `default_model`. A raw value already in
        LiteLLM format (contains a `/`, e.g. `gemini/gemini-2.5-pro`) passes
        through unchanged. Anything else is looked up in
        `config.model_aliases`; an alias that isn't defined there raises
        ValueError.
        """
        agent = self._require_agent(agent_name)
        raw = agent.config.model or self.config.default_model
        return self.resolve_model_string(raw)

    def resolve_model_string(self, raw: str) -> str:
        """Resolve a raw model/alias string against `config.model_aliases`."""
        if "/" in raw:
            return raw
        if raw in self.config.model_aliases:
            return self.config.model_aliases[raw]
        raise ValueError(
            f"Unknown model alias {raw!r}. Defined aliases: "
            f"{sorted(self.config.model_aliases)}"
        )

    def check_env(self, agent_name: str) -> list[str]:
        """Return the names of required env vars that are missing for this agent."""
        import os

        agent = self._require_agent(agent_name)
        missing = []
        for req in agent.config.requires.env:
            if req.required and not os.environ.get(req.name):
                missing.append(req.name)
        return missing

    def _require_agent(self, agent_name: str) -> AgentSpec:
        if agent_name not in self.agents:
            raise KeyError(
                f"Unknown agent {agent_name!r}. Known agents: "
                f"{sorted(self.agents)}"
            )
        return self.agents[agent_name]
