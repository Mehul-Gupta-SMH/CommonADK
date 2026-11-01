"""commonadk -- define an agent system once, run it on any agent SDK.

See plan.md at the repo root for the full hypothesis and architecture. This
package (M1) is the framework-neutral core: models, loader, validation, and
the mermaid renderer. It has no dependency on any agent SDK.
"""

from .adapters import BaseAdapter, get_adapter
from .loader import load
from .mermaid import render_mermaid, write_interaction_layer
from .mixed import MixedSystem, RuntimeUnit, build_mixed
from .runners import BaseRunner, get_runner
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
from .decisions import (
    AnswerValue,
    DecisionAnswer,
    DecisionCase,
    DecisionKind,
    DecisionOutcome,
    DecisionQuestion,
    DecisionSpec,
    accuracy_by_question,
    load_decisions,
    score_case,
    unanswered,
)
from .validation import ValidationError

__all__ = [
    "AnswerValue",
    "BaseAdapter",
    "get_adapter",
    "load",
    "render_mermaid",
    "write_interaction_layer",
    "MixedSystem",
    "RuntimeUnit",
    "build_mixed",
    "BaseRunner",
    "get_runner",
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
    # Declared answer contracts, scored by equality (no judge).
    "DecisionAnswer",
    "DecisionCase",
    "DecisionKind",
    "DecisionOutcome",
    "DecisionQuestion",
    "DecisionSpec",
    "accuracy_by_question",
    "load_decisions",
    "score_case",
    "unanswered",
]
