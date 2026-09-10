"""A single, obvious, static token-cost table. See docs/runner-design.md,
"Cost estimation".

**This is a snapshot, not a live price feed.** USD per 1M tokens, input and
output priced separately, for a small set of model ids this repo's shipped
example and adapters route to natively (see README.md, "Supported
targets"). It WILL drift out of date -- provider pricing changes more
often than this file will be updated -- and is not fetched from anywhere at
runtime. Update it here, in this one place, when it drifts; nowhere else
in `runners/` hardcodes a price.

`estimate_cost_usd` returns `None` -- never `0`, never a guess -- whenever
the model isn't in this table or either token count is itself `None` (i.e.
the SDK didn't report usage for that call in the first place). It is never
called with a token count a runner fabricated; every caller in this
codebase passes through exactly what the SDK reported, unchanged.

The Claude Agent SDK is the one target where this table should NOT be
used at all once its runner is built: `ResultMessage.total_cost_usd` and
`ModelUsage.costUSD` (see docs/runner-design.md's per-SDK mapping table)
are cost figures the SDK computes itself, more authoritative than a static
table here could ever be.
"""

from __future__ import annotations

from typing import Optional

# model id (the bare id after resolving "provider/model", or an SDK-native
# id string -- see each runner's `_resolved_model`) -> (usd_per_1m_input,
# usd_per_1m_output). Indicative rates only -- see module docstring.
_PRICING_USD_PER_1M_TOKENS: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


def estimate_cost_usd(
    model: Optional[str],
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
) -> Optional[float]:
    """USD cost for `prompt_tokens` + `completion_tokens` on `model`, or
    `None` if `model` isn't priced here or either token count is `None`.

    `model` may be a full `"provider/model-id"` LiteLLM-format string (as
    `Project.resolve_model` returns) or a bare SDK-native id -- only the
    part after the last `/` is looked up, so both forms work.
    """
    if not model or prompt_tokens is None or completion_tokens is None:
        return None
    bare_id = model.rsplit("/", 1)[-1]
    rates = _PRICING_USD_PER_1M_TOKENS.get(bare_id)
    if rates is None:
        return None
    input_rate, output_rate = rates
    cost = (prompt_tokens / 1_000_000) * input_rate + (
        completion_tokens / 1_000_000
    ) * output_rate
    return round(cost, 6)


__all__ = ["estimate_cost_usd"]
