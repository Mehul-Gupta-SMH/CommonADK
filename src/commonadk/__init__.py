"""commonadk -- define an agent system once, run it on any agent SDK.

See plan.md at the repo root for the full hypothesis and architecture. This
package (M1) is the framework-neutral core: models, loader, validation, and
the mermaid renderer. It has no dependency on any agent SDK.
"""

from .adapters import BaseAdapter, get_adapter
from .loader import load
from .mermaid import render_mermaid, write_interaction_layer
from .models import (
    AgentConfig,
    AgentSpec,
    EnvRequirement,
    InteractionEdge,
    InteractionGraph,
    Project,
    ProjectConfig,
    Requires,
    ToolParameter,
    ToolSpec,
)
from .validation import ValidationError

__all__ = [
    "BaseAdapter",
    "get_adapter",
    "load",
    "render_mermaid",
    "write_interaction_layer",
    "AgentConfig",
    "AgentSpec",
    "EnvRequirement",
    "InteractionEdge",
    "InteractionGraph",
    "Project",
    "ProjectConfig",
    "Requires",
    "ToolParameter",
    "ToolSpec",
    "ValidationError",
]
