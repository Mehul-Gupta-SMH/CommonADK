"""Tools available to the assistant agent. Pure Python, no network calls."""

from __future__ import annotations


def count_words(text: str) -> int:
    """Count the number of whitespace-separated words in a piece of text.

    Args:
        text: The text to measure.

    Returns:
        The number of words in `text`.
    """
    return len(text.split())
