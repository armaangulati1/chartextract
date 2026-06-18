from schema import ExtractionOutput, OncologyExtract, TokenUsage

from cost import estimate_cost_usd
from observability import attach_run_metrics, langfuse_trace_url


def test_estimate_cost_usd():
    assert estimate_cost_usd(1_000_000, 500_000, input_cost_per_1m=0.15, output_cost_per_1m=0.60) == 0.45


def test_attach_run_metrics_adds_latency_and_cost():
    output = ExtractionOutput(
        extract=OncologyExtract(primary_site="lung"),
        usage=TokenUsage(prompt_tokens=1000, completion_tokens=200, total_tokens=1200),
    )
    enriched = attach_run_metrics(output, latency_ms=1234.5)
    assert enriched.run_metrics.latency_ms == 1234.5
    assert enriched.run_metrics.estimated_cost_usd > 0


def test_langfuse_trace_url():
    url = langfuse_trace_url("abc123")
    assert url.endswith("/trace/abc123")
    assert langfuse_trace_url(None) is None
