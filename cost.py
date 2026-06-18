"""Token cost estimation for observability and batch reporting."""

from __future__ import annotations

import os

DEFAULT_INPUT_COST_PER_1M = float(os.getenv("OPENAI_INPUT_COST_PER_1M", "0.15"))
DEFAULT_OUTPUT_COST_PER_1M = float(os.getenv("OPENAI_OUTPUT_COST_PER_1M", "0.60"))


def estimate_cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    *,
    input_cost_per_1m: float = DEFAULT_INPUT_COST_PER_1M,
    output_cost_per_1m: float = DEFAULT_OUTPUT_COST_PER_1M,
) -> float:
    return round(
        (prompt_tokens * input_cost_per_1m + completion_tokens * output_cost_per_1m) / 1_000_000,
        6,
    )
