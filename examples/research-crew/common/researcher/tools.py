"""Tools available to the researcher agent.

These are stubs: they return canned, deterministic data instead of making
real network calls, so the example project loads, validates, and runs its
tests entirely offline. A real deployment would swap the bodies for actual
API calls while keeping the same typed signature and docstring contract.
"""

from __future__ import annotations


def search_web(query: str) -> str:
    """Search the web for a query and return a short summary of top results.

    Requires the TAVILY_API_KEY environment variable in a real deployment
    (declared in this agent's `requires.env`). This stub returns canned
    text so the example works offline.

    Args:
        query: The search query.

    Returns:
        A short plain-text summary of (stubbed) search results.
    """
    return (
        f"Stub search results for '{query}': three articles found covering "
        f"background, recent developments, and expert commentary."
    )


def fetch_page(url: str) -> str:
    """Fetch a web page and return its plain-text content.

    Args:
        url: The URL of the page to fetch.

    Returns:
        Stubbed plain-text page content.
    """
    return f"Stub page content fetched from {url}."
