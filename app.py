from typing import Optional

import json
import os
from datetime import date

import requests
import streamlit as st

from eval_dashboard import format_metric_pct, load_eval_metrics
from fhir import to_fhir, validate_fhir_bundle
from observability import langfuse_trace_url
from schema import Biomarker, BiomarkerStatus, CancerStage, EcogPerformanceStatus, ExtractionOutput

API_URL = os.environ.get("API_URL", "http://localhost:8000")

FIELD_LABELS = {
    "primary_site": "Primary site",
    "histology": "Histology",
    "stage": "Stage",
    "ecog_performance_status": "ECOG performance status",
    "line_of_therapy": "Line of therapy",
    "date_of_diagnosis": "Date of diagnosis",
    "biomarkers": "Biomarkers",
    "treatment_regimen": "Treatment regimen",
}

STAGE_OPTIONS = [""] + [s.value for s in CancerStage]
ECOG_OPTIONS = list(EcogPerformanceStatus)
BIOMARKER_STATUS_OPTIONS = [s.value for s in BiomarkerStatus]


def _call_extract(text: str, review_threshold: float) -> ExtractionOutput:
    resp = requests.post(
        f"{API_URL}/extract",
        json={"text": text, "review_threshold": review_threshold},
        timeout=120,
    )
    resp.raise_for_status()
    return ExtractionOutput.model_validate(resp.json())


def _parse_biomarkers(raw: str) -> list[Biomarker]:
    if not raw.strip():
        return []
    items = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        name, status = line.split(":", 1)
        items.append(Biomarker(name=name.strip(), status=BiomarkerStatus(status.strip().lower())))
    return items


def _format_biomarkers(items: list[Biomarker]) -> str:
    return "\n".join(f"{b.name}: {b.status.value}" for b in items)


def _render_scalar_editor(field_name: str, value):
    label = FIELD_LABELS.get(field_name, field_name)
    if field_name == "stage":
        current = value.value if isinstance(value, CancerStage) else (value or "")
        choice = st.selectbox(
            label,
            STAGE_OPTIONS,
            index=STAGE_OPTIONS.index(current) if current in STAGE_OPTIONS else 0,
            key=f"review_{field_name}",
        )
        return CancerStage(choice) if choice else None
    if field_name == "ecog_performance_status":
        current = int(value) if value is not None else 0
        return st.selectbox(
            label,
            ECOG_OPTIONS,
            format_func=lambda x: str(int(x)),
            index=ECOG_OPTIONS.index(EcogPerformanceStatus(current)) if current in ECOG_OPTIONS else 0,
            key=f"review_{field_name}",
        )
    if field_name == "line_of_therapy":
        return st.number_input(
            label,
            min_value=1,
            max_value=10,
            value=int(value) if value is not None else 1,
            key=f"review_{field_name}",
        )
    if field_name == "date_of_diagnosis":
        if isinstance(value, str):
            try:
                value = date.fromisoformat(value)
            except ValueError:
                value = None
        picked = st.date_input(label, value=value, key=f"review_{field_name}")
        return picked
    if field_name == "treatment_regimen":
        text = ", ".join(value) if isinstance(value, list) else (value or "")
        edited = st.text_area(label, value=text, key=f"review_{field_name}")
        return [part.strip() for part in edited.split(",") if part.strip()]
    if field_name == "biomarkers":
        if isinstance(value, list) and value and isinstance(value[0], dict):
            value = [Biomarker(**item) for item in value]
        text = _format_biomarkers(value) if isinstance(value, list) else (value or "")
        edited = st.text_area(
            f"{label} (name: status per line)",
            value=text,
            key=f"review_{field_name}",
        )
        return _parse_biomarkers(edited)
    return st.text_input(label, value=value or "", key=f"review_{field_name}") or None


def _apply_review(output: ExtractionOutput, accepted: dict[str, object]) -> ExtractionOutput:
    payload = output.extract.model_dump(mode="json")
    for field_name, value in accepted.items():
        if field_name == "date_of_diagnosis" and isinstance(value, date):
            payload[field_name] = value.isoformat()
        elif field_name == "stage" and isinstance(value, CancerStage):
            payload[field_name] = value.value
        elif field_name == "ecog_performance_status" and value is not None:
            payload[field_name] = int(value)
        elif field_name == "biomarkers":
            payload[field_name] = [
                {"name": b.name, "status": b.status.value} for b in value
            ]
        else:
            payload[field_name] = value

    updated_fields = dict(output.fields)
    remaining_review = list(output.needs_review)
    for field_name in accepted:
        meta = updated_fields.get(field_name)
        if meta is None:
            continue
        updated_fields[field_name] = meta.model_copy(
            update={"needs_review": False, "confidence": 1.0, "source": "human_review"}
        )
        if field_name in remaining_review:
            remaining_review.remove(field_name)

    from schema import OncologyExtract

    return ExtractionOutput(
        extract=OncologyExtract.model_validate(payload),
        fields=updated_fields,
        needs_review=remaining_review,
        review_threshold=output.review_threshold,
        usage=output.usage,
        run_metrics=output.run_metrics,
    )


st.title("Clinical Text Extractor")
st.caption("Structured oncology variables from clinical notes — with human-in-the-loop review.")

tab_extract, tab_eval = st.tabs(["Extract", "Eval metrics"])

if "output" not in st.session_state:
    st.session_state.output = None
if "approved" not in st.session_state:
    st.session_state.approved = None

with st.sidebar:
    st.header("Review settings")
    review_threshold = st.slider(
        "Confidence threshold",
        min_value=0.5,
        max_value=0.95,
        value=0.75,
        step=0.05,
        help="Fields below this confidence are routed to human review instead of auto-shipping.",
    )

with tab_extract:
    text = st.text_area("Paste clinical text:", height=160)

    if st.button("Extract", type="primary") and text.strip():
        with st.spinner("Extracting..."):
            st.session_state.output = _call_extract(text.strip(), review_threshold)
            st.session_state.approved = None

    output: Optional[ExtractionOutput] = st.session_state.output

    if output is not None:
        metrics = output.run_metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Latency", f"{metrics.latency_ms:.0f} ms")
        m2.metric("Tokens", f"{output.usage.total_tokens:,}")
        m3.metric("Est. cost", f"${metrics.estimated_cost_usd:.4f}")
        m4.metric("Needs review", len(output.needs_review))

        trace_url = langfuse_trace_url(metrics.trace_id)
        if trace_url:
            st.markdown(f"[View trace in Langfuse]({trace_url})")

        flagged = output.needs_review
        if flagged:
            st.warning(
                f"{len(flagged)} field(s) need review before shipping "
                f"(confidence < {output.review_threshold:.2f}): "
                + ", ".join(FIELD_LABELS.get(f, f) for f in flagged)
            )
        else:
            st.success("All extracted fields meet the confidence threshold — ready to ship.")

        with st.expander("Field confidence", expanded=bool(flagged)):
            rows = []
            for name, meta in output.fields.items():
                rows.append(
                    {
                        "field": FIELD_LABELS.get(name, name),
                        "confidence": meta.confidence,
                        "needs_review": meta.needs_review,
                        "source": meta.source,
                        "flags": ", ".join(meta.flags),
                    }
                )
            st.dataframe(rows, use_container_width=True, hide_index=True)

        if flagged:
            st.subheader("Review panel")
            st.caption("Confirm or correct low-confidence fields, then accept to clear the review flag.")
            extract_data = output.extract.model_dump(mode="json")
            accepted: dict[str, object] = {}

            for field_name in flagged:
                meta = output.fields[field_name]
                with st.container(border=True):
                    st.markdown(f"**{FIELD_LABELS.get(field_name, field_name)}**")
                    st.caption(
                        f"Confidence: {meta.confidence:.2f} · Source: {meta.source or 'unknown'}"
                        + (f" · Flags: {', '.join(meta.flags)}" if meta.flags else "")
                    )
                    if meta.evidence:
                        st.info(f"Evidence: {meta.evidence}")
                    accepted[field_name] = _render_scalar_editor(field_name, extract_data.get(field_name))

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Accept reviewed fields", type="primary"):
                    st.session_state.output = _apply_review(output, accepted)
                    st.session_state.approved = st.session_state.output.extract.model_dump(mode="json")
                    st.rerun()
            with col2:
                if st.button("Reset to model output"):
                    st.session_state.approved = None
                    st.rerun()

        display = st.session_state.approved or output.extract.model_dump(mode="json")
        st.subheader("Extracted record")
        st.json(display)

        fhir_bundle = to_fhir(output.extract)
        with st.expander("FHIR export"):
            st.caption("FHIR R4 Bundle mapped from the extracted oncology record.")
            st.json(fhir_bundle)
            if validate_fhir_bundle(fhir_bundle):
                st.success("Bundle passes structural FHIR validation.")
            st.download_button(
                "Download FHIR Bundle",
                data=json.dumps(fhir_bundle, indent=2),
                file_name="oncology.fhir.json",
                mime="application/json",
            )

        if st.session_state.approved:
            st.download_button(
                "Download approved JSON",
                data=json.dumps(st.session_state.approved, indent=2),
                file_name="oncology_extract.json",
                mime="application/json",
            )

with tab_eval:
    st.subheader("Latest eval metrics")
    eval_payload = load_eval_metrics()
    if eval_payload is None:
        st.info("No eval metrics found. Run `python eval.py --data-dir data/eval/ci_gold` to generate them.")
    else:
        source = eval_payload.get("source", "unknown")
        updated = eval_payload.get("updated_at") or "unknown"
        st.caption(f"Source: `{source}` · Updated: {updated}")

        macro_f1 = eval_payload.get("macro_f1")
        if macro_f1 is not None:
            st.metric("Macro F1", format_metric_pct(macro_f1))
        if eval_payload.get("n_examples") is not None:
            st.metric("Examples evaluated", eval_payload["n_examples"])

        table_rows = []
        for row in eval_payload.get("rows", []):
            table_rows.append(
                {
                    "field": row["field"],
                    "precision": format_metric_pct(row["precision"]),
                    "recall": format_metric_pct(row["recall"]),
                    "F1": format_metric_pct(row["f1"]),
                    "TP": row.get("tp", ""),
                    "FP": row.get("fp", ""),
                    "FN": row.get("fn", ""),
                }
            )
        st.dataframe(table_rows, use_container_width=True, hide_index=True)

        errors = eval_payload.get("error_distribution") or {}
        if errors:
            st.markdown("**Error taxonomy**")
            st.dataframe(
                [{"error_type": k, "count": v} for k, v in sorted(errors.items())],
                use_container_width=True,
                hide_index=True,
            )
