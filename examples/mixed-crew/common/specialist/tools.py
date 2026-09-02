"""Tools available to the specialist agent. Pure Python, no network calls."""

from __future__ import annotations


def summarize(text: str, max_words: int = 30) -> str:
    """Trim text down to a short summary, word-capped.

    Args:
        text: The text to summarize.
        max_words: The maximum number of words to keep.

    Returns:
        The first `max_words` words of `text`, joined back into a string.
    """
    return " ".join(text.split()[:max_words])
