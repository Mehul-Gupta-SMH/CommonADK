"""Tools available to the coordinator agent.

Plain, typed, documented Python functions -- commonadk turns these into
native tool schemas for whichever SDK an agent is built against (M2+). No
network calls; everything here is pure Python so the example runs offline.
"""

from __future__ import annotations


def split_into_subtopics(topic: str, max_subtopics: int = 3) -> list[str]:
    """Split a broad research topic into a short list of narrower subtopics.

    Args:
        topic: The user's original research topic, e.g. "electric vehicles".
        max_subtopics: The maximum number of subtopics to return.

    Returns:
        A list of subtopic strings, each derived from `topic`, capped at
        `max_subtopics` entries.
    """
    angles = ["background and history", "current state", "open challenges", "future outlook"]
    return [f"{topic}: {angle}" for angle in angles[:max_subtopics]]


def format_handoff_note(from_agent: str, to_agent: str, summary: str) -> str:
    """Format a short note describing why work is being routed between agents.

    Args:
        from_agent: Name of the agent handing off or delegating work.
        to_agent: Name of the agent receiving the work.
        summary: One-line description of what needs to happen next.

    Returns:
        A single human-readable line summarizing the routing decision.
    """
    return f"[{from_agent} -> {to_agent}] {summary}"
