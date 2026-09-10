"""An ordered, JSON-serializable trace of events plus roll-up totals.

See docs/runner-design.md, "Trace and rollups". The rule that matters: a
sum is only ever computed over the events that actually reported the field
being summed, and `usage_complete`/`cost_complete` say, explicitly, whether
that sum is the whole run's true total or only a subset -- with a `note`
spelling out which, rather than silently presenting a partial sum as
complete. `Trace` itself has no agent-SDK import -- appending events,
rolling them up, and (de)serializing to JSON never requires any SDK.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

from .events import Event, LLMCall, RunFinished, ToolCall, event_from_dict


def _rollup_llm_calls(calls: list[LLMCall]) -> dict[str, Any]:
    count = len(calls)
    reported = [c for c in calls if c.total_tokens is not None]
    usage_complete = count == 0 or len(reported) == count
    costed = [c for c in calls if c.cost_usd is not None]
    cost_complete = count == 0 or len(costed) == count

    note: Optional[str] = None
    if not usage_complete:
        note = (
            f"{count - len(reported)} of {count} LLM call(s) did not report "
            "token usage; the token totals below sum only the calls that "
            "did -- they are NOT the true total for this run."
        )
    elif not cost_complete:
        note = (
            f"{count - len(costed)} of {count} LLM call(s) used a model not "
            "in the static pricing table (runners/pricing.py); cost_usd "
            "below sums only the calls that were priced -- it is NOT the "
            "true total cost for this run."
        )

    return {
        "count": count,
        "reported_count": len(reported),
        "usage_complete": usage_complete,
        "prompt_tokens": (
            sum(c.prompt_tokens for c in reported if c.prompt_tokens is not None)
            if reported
            else None
        ),
        "completion_tokens": (
            sum(c.completion_tokens for c in reported if c.completion_tokens is not None)
            if reported
            else None
        ),
        "total_tokens": (sum(c.total_tokens for c in reported) if reported else None),
        "priced_count": len(costed),
        "cost_complete": cost_complete,
        "cost_usd": (round(sum(c.cost_usd for c in costed), 6) if costed else None),
        "note": note,
    }


def _rollup_tool_calls(calls: list[ToolCall]) -> dict[str, Any]:
    return {
        "count": len(calls),
        "errors": sum(1 for c in calls if c.error),
    }


class Trace:
    """An ordered log of one `run()` call's events, plus roll-up totals."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def append(self, event: Event) -> None:
        self.events.append(event)

    def rollup(self) -> dict[str, Any]:
        """Roll-up totals: overall, plus a breakdown per agent name.

        `LLMCall`/`ToolCall` events with `agent_name=None` (a documented
        per-SDK attribution gap -- see docs/runner-design.md) count toward
        the overall totals but are excluded from every `per_agent` bucket,
        rather than being guessed into one.
        """
        llm_calls = [e for e in self.events if isinstance(e, LLMCall)]
        tool_calls = [e for e in self.events if isinstance(e, ToolCall)]

        agent_names = sorted(
            {c.agent_name for c in llm_calls if c.agent_name}
            | {c.agent_name for c in tool_calls if c.agent_name}
        )
        per_agent = {
            name: {
                "llm_calls": _rollup_llm_calls(
                    [c for c in llm_calls if c.agent_name == name]
                ),
                "tool_calls": _rollup_tool_calls(
                    [c for c in tool_calls if c.agent_name == name]
                ),
            }
            for name in agent_names
        }

        return {
            "event_count": len(self.events),
            "llm_calls": _rollup_llm_calls(llm_calls),
            "tool_calls": _rollup_tool_calls(tool_calls),
            "per_agent": per_agent,
        }

    def final_event(self) -> Optional[Union[RunFinished, Event]]:
        """The last `RunFinished` or `RunError` event, if any -- `None`
        for a trace that has neither yet (still in progress, or empty)."""
        from .events import RunError

        for event in reversed(self.events):
            if isinstance(event, (RunFinished, RunError)):
                return event
        return None

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        payload = {
            "events": [e.to_dict() for e in self.events],
            "totals": self.rollup(),
        }
        return json.dumps(payload, indent=indent, default=str)

    def write(self, path: Union[str, Path]) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_json(cls, text: str) -> "Trace":
        data = json.loads(text)
        trace = cls()
        for event_dict in data["events"]:
            trace.append(event_from_dict(event_dict))
        return trace

    @classmethod
    def read(cls, path: Union[str, Path]) -> "Trace":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


__all__ = ["Trace"]
