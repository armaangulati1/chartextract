"""Per-run latency/cost reporting and Langfuse trace enrichment."""

from __future__ import annotations

import os
from typing import Optional

from langfuse import get_client

from cost import estimate_cost_usd
from schema import ExtractionOutput, RunMetrics


def langfuse_trace_url(trace_id: Optional[str]) -> Optional[str]:
    if not trace_id:
        return None
    host = os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com").rstrip("/")
    return f"{host}/trace/{trace_id}"


def publish_run_metrics(
    output: ExtractionOutput,
    *,
    latency_ms: float,
) -> tuple[float, Optional[str]]:
    """Push latency/token/cost metadata to Langfuse; return cost and trace id."""
    cost = estimate_cost_usd(output.usage.prompt_tokens, output.usage.completion_tokens)
    trace_id: Optional[str] = None

    try:
        client = get_client()
        client.update_current_trace(
            metadata={
                "latency_ms": round(latency_ms, 2),
                "prompt_tokens": output.usage.prompt_tokens,
                "completion_tokens": output.usage.completion_tokens,
                "total_tokens": output.usage.total_tokens,
                "estimated_cost_usd": cost,
                "needs_review_fields": output.needs_review,
                "needs_review_count": len(output.needs_review),
                "review_threshold": output.review_threshold,
            },
            output={
                "needs_review": output.needs_review,
                "usage": output.usage.model_dump(),
            },
        )
        trace_id = client.get_current_trace_id()
    except Exception:
        trace_id = None

    return cost, trace_id


def attach_run_metrics(output: ExtractionOutput, *, latency_ms: float) -> ExtractionOutput:
    cost, trace_id = publish_run_metrics(output, latency_ms=latency_ms)
    return output.model_copy(
        update={
            "run_metrics": RunMetrics(
                latency_ms=round(latency_ms, 2),
                estimated_cost_usd=cost,
                trace_id=trace_id,
            )
        }
    )
