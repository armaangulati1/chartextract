"""Map OncologyExtract records to FHIR R4 resources."""

from __future__ import annotations

import uuid
from datetime import date
from typing import Any, Optional

from schema import Biomarker, BiomarkerStatus, OncologyExtract

FHIR_VERSION = "4.0.1"
DEFAULT_PATIENT_REF = "Patient/oncology-extract-subject"

BIOMARKER_LOINC: dict[str, str] = {
    "EGFR": "81252-9",
    "PD-L1": "85307-0",
    "HER2": "48676-1",
    "ALK": "81479-3",
    "ROS1": "81479-3",
    "BRAF": "5098-2",
    "KRAS": "5095-8",
    "NRAS": "5096-6",
    "PSA": "2857-1",
    "CA 19-9": "24108-3",
    "CA-19-9": "24108-3",
}

BIOMARKER_INTERPRETATION = {
    BiomarkerStatus.POSITIVE: {
        "coding": [{"system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation", "code": "POS", "display": "Positive"}]
    },
    BiomarkerStatus.NEGATIVE: {
        "coding": [{"system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation", "code": "NEG", "display": "Negative"}]
    },
    BiomarkerStatus.EQUIVOCAL: {
        "coding": [{"system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation", "code": "IND", "display": "Indeterminate"}]
    },
    BiomarkerStatus.UNKNOWN: {
        "coding": [{"system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation", "code": "UNK", "display": "Unknown"}]
    },
}


def _id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _condition(record: OncologyExtract) -> dict[str, Any]:
    code_text_parts = []
    if record.histology:
        code_text_parts.append(record.histology)
    if record.stage:
        code_text_parts.append(f"stage {record.stage.value}")

    condition: dict[str, Any] = {
        "resourceType": "Condition",
        "id": _id("condition"),
        "clinicalStatus": {
            "coding": [
                {
                    "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                    "code": "active",
                    "display": "Active",
                }
            ]
        },
        "verificationStatus": {
            "coding": [
                {
                    "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                    "code": "confirmed",
                    "display": "Confirmed",
                }
            ]
        },
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/condition-category",
                        "code": "problem-list-item",
                        "display": "Problem List Item",
                    }
                ]
            }
        ],
        "code": {
            "coding": [
                {
                    "system": "http://snomed.info/sct",
                    "code": "363346000",
                    "display": "Malignant neoplastic disease",
                }
            ],
            "text": " ".join(code_text_parts) or "Malignant neoplasm",
        },
        "subject": {"reference": DEFAULT_PATIENT_REF},
    }

    if record.date_of_diagnosis:
        condition["onsetDateTime"] = _iso_date(record.date_of_diagnosis)

    if record.primary_site:
        condition["bodySite"] = [
            {
                "coding": [
                    {
                        "system": "http://snomed.info/sct",
                        "code": "123037004",
                        "display": record.primary_site,
                    }
                ],
                "text": record.primary_site,
            }
        ]

    if record.stage:
        condition["stage"] = [
            {
                "summary": {
                    "coding": [
                        {
                            "system": "http://cancerstaging.org",
                            "code": record.stage.value,
                            "display": f"AJCC stage {record.stage.value}",
                        }
                    ],
                    "text": f"Stage {record.stage.value}",
                }
            }
        ]

    return condition


def _iso_date(value: date | str) -> str:
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _biomarker_observation(biomarker: Biomarker) -> dict[str, Any]:
    loinc = BIOMARKER_LOINC.get(biomarker.name.upper()) or BIOMARKER_LOINC.get(biomarker.name)
    coding = []
    if loinc:
        coding.append(
            {
                "system": "http://loinc.org",
                "code": loinc,
                "display": biomarker.name,
            }
        )

    return {
        "resourceType": "Observation",
        "id": _id("observation"),
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": "laboratory",
                        "display": "Laboratory",
                    }
                ]
            }
        ],
        "code": {
            "coding": coding,
            "text": biomarker.name,
        },
        "subject": {"reference": DEFAULT_PATIENT_REF},
        "interpretation": [BIOMARKER_INTERPRETATION[biomarker.status]],
        "valueCodeableConcept": {
            "text": biomarker.status.value,
        },
    }


def _ecog_observation(ecog: int) -> dict[str, Any]:
    return {
        "resourceType": "Observation",
        "id": _id("observation"),
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": "survey",
                        "display": "Survey",
                    }
                ]
            }
        ],
        "code": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": "89262-0",
                    "display": "ECOG Performance Status score",
                }
            ],
            "text": "ECOG performance status",
        },
        "subject": {"reference": DEFAULT_PATIENT_REF},
        "valueInteger": ecog,
    }


def _line_of_therapy_observation(line: int) -> dict[str, Any]:
    return {
        "resourceType": "Observation",
        "id": _id("observation"),
        "status": "final",
        "code": {"text": "Line of therapy"},
        "subject": {"reference": DEFAULT_PATIENT_REF},
        "valueInteger": line,
    }


def _medication_statement(drug: str) -> dict[str, Any]:
    return {
        "resourceType": "MedicationStatement",
        "id": _id("medication"),
        "status": "active",
        "medicationCodeableConcept": {"text": drug},
        "subject": {"reference": DEFAULT_PATIENT_REF},
    }


def to_fhir(
    record: OncologyExtract,
    *,
    patient_reference: str = DEFAULT_PATIENT_REF,
) -> dict[str, Any]:
    """Return a FHIR R4 Bundle containing mapped oncology resources."""
    entries: list[dict[str, Any]] = []

    if any(
        (
            record.primary_site,
            record.histology,
            record.stage,
            record.date_of_diagnosis,
        )
    ):
        condition = _condition(record)
        condition["subject"] = {"reference": patient_reference}
        entries.append({"fullUrl": f"urn:uuid:{condition['id']}", "resource": condition})

    if record.ecog_performance_status is not None:
        ecog = _ecog_observation(int(record.ecog_performance_status))
        ecog["subject"] = {"reference": patient_reference}
        entries.append({"fullUrl": f"urn:uuid:{ecog['id']}", "resource": ecog})

    if record.line_of_therapy is not None:
        lot = _line_of_therapy_observation(record.line_of_therapy)
        lot["subject"] = {"reference": patient_reference}
        entries.append({"fullUrl": f"urn:uuid:{lot['id']}", "resource": lot})

    for biomarker in record.biomarkers:
        obs = _biomarker_observation(biomarker)
        obs["subject"] = {"reference": patient_reference}
        entries.append({"fullUrl": f"urn:uuid:{obs['id']}", "resource": obs})

    for drug in record.treatment_regimen:
        med = _medication_statement(drug)
        med["subject"] = {"reference": patient_reference}
        entries.append({"fullUrl": f"urn:uuid:{med['id']}", "resource": med})

    return {
        "resourceType": "Bundle",
        "type": "collection",
        "timestamp": date.today().isoformat(),
        "entry": entries,
    }


def validate_fhir_bundle(bundle: dict[str, Any]) -> bool:
    """Lightweight structural validation for mapped FHIR bundles."""
    if bundle.get("resourceType") != "Bundle":
        return False
    if bundle.get("type") != "collection":
        return False
    if "entry" not in bundle or not isinstance(bundle["entry"], list):
        return False
    for item in bundle["entry"]:
        resource = item.get("resource", {})
        if "resourceType" not in resource:
            return False
    return True
