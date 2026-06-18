from fastapi.testclient import TestClient

from api import app
from schema import Biomarker, BiomarkerStatus, CancerStage, OncologyExtract


def test_oncology_extract_import_from_schema():
    data = OncologyExtract(
        primary_site="lung",
        histology="adenocarcinoma",
        stage=CancerStage.IIIA,
        biomarkers=[Biomarker(name="EGFR", status=BiomarkerStatus.NEGATIVE)],
        line_of_therapy=1,
        treatment_regimen=["pembrolizumab"],
    )
    assert data.primary_site == "lung"
    assert data.stage == CancerStage.IIIA


def test_health():
    assert TestClient(app).get("/health").json() == {"status": "ok"}
