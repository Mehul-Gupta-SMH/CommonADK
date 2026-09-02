"""Tools available to the coordinator agent. Pure Python, no network calls."""

from __future__ import annotations


def greet(name: str) -> str:
    """Produce a short greeting for the user.

    Args:
        name: The user's name, or a generic placeholder if unknown.

    Returns:
        A one-line greeting.
    """
    return f"Hi {name}, let me route that for you."
