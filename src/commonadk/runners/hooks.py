"""Observe-only hook registry. See docs/runner-design.md, "Hook contract".

v1 is observe-only: a registered callback receives an `Event` strictly
*after* it happened. It cannot block a tool call, rewrite a reported token
count, veto a transfer, or otherwise change the run in progress -- see the
design doc for why that's a deliberate v1 scope decision and exactly how
this shape stays extensible to intervention later without a breaking
change to `register`/`fire`.

Exception policy -- isolate and report, never fail-fast: a raising hook's
exception is caught, recorded in `HookRegistry.errors`, and surfaced via
`warnings.warn`; every remaining hook for that event still runs, and the
run being observed is never aborted by a broken observer. See the design
doc, "Hook contract" for the full justification.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Callable, Optional

from .events import Event

HookCallback = Callable[[Event], None]


class HookRegistry:
    """Register callbacks per event type (or catch-all) and fire them in order."""

    def __init__(self) -> None:
        self._by_type: dict[Optional[type], list[HookCallback]] = defaultdict(list)
        self.errors: list[tuple[Event, HookCallback, BaseException]] = []
        """Every (event, callback, exception) a raising hook produced, in
        the order it happened -- inspect this after a run to see whether
        any observer misbehaved; it is never consulted to change control
        flow, only for visibility."""

    def register(
        self, callback: HookCallback, event_type: Optional[type[Event]] = None
    ) -> None:
        """Register `callback` for `event_type`, or every event if `event_type` is `None`.

        Multiple callbacks (for the same or different types) may be
        registered; they fire in registration order, type-specific hooks
        before catch-all hooks for the same event.
        """
        self._by_type[event_type].append(callback)

    def fire(self, event: Event) -> None:
        """Call every hook registered for `event`'s exact type plus every
        catch-all hook, in that order. Never raises -- see the module
        docstring's exception policy.
        """
        callbacks = list(self._by_type.get(type(event), ())) + list(
            self._by_type.get(None, ())
        )
        for callback in callbacks:
            try:
                callback(event)
            except Exception as exc:  # noqa: BLE001 -- isolate-and-report policy
                self.errors.append((event, callback, exc))
                name = getattr(callback, "__name__", repr(callback))
                warnings.warn(
                    f"commonadk: hook {name!r} raised {exc!r} while handling "
                    f"{event.kind!r} (seq={event.seq}) -- isolated, the run "
                    "continues (see docs/runner-design.md, 'Hook contract')",
                    stacklevel=2,
                )


__all__ = ["HookRegistry", "HookCallback"]
