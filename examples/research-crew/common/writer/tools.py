"""Tools available to the writer agent. Pure Python, no network calls."""

from __future__ import annotations


def count_words(text: str) -> int:
    """Count the number of whitespace-separated words in a piece of text.

    Args:
        text: The text to measure.

    Returns:
        The number of words in `text`.
    """
    return len(text.split())


def format_as_markdown(title: str, bullet_points: list[str]) -> str:
    """Format a title and a list of bullet points as a Markdown snippet.

    Args:
        title: The heading for the summary.
        bullet_points: The individual points to render as a bulleted list.

    Returns:
        A Markdown string with an `##` heading followed by a bullet list.
    """
    lines = [f"## {title}", ""]
    lines.extend(f"- {point}" for point in bullet_points)
    return "\n".join(lines)
