from fastapi.testclient import TestClient

from api import app
from extractor import ClinicalExtract, Medication


def test_clinical_extract_schema():
    data = ClinicalExtract(
        medications=[Medication(name="metoprolol", dose="50 mg", route="oral")],
        adverse_reactions=["dizziness"],
        patient_age=67,
    )
    assert data.patient_age == 67
    assert data.medications[0].name == "metoprolol"


def test_health():
    assert TestClient(app).get("/health").json() == {"status": "ok"}
