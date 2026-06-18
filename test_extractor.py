from fastapi.testclient import TestClient

from api import app
from schema import Biomarker, BiomarkerStatus, CancerStage, ExtractionOutput, OncologyExtract


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


def test_extraction_output_schema():
    extract = OncologyExtract(primary_site="lung")
    output = ExtractionOutput(
        extract=extract,
        fields={"primary_site": {"confidence": 0.9, "needs_review": False}},
        needs_review=[],
        review_threshold=0.75,
    )
    assert output.extract.primary_site == "lung"
    assert output.fields["primary_site"].confidence == 0.9


def test_health():
    assert TestClient(app).get("/health").json() == {"status": "ok"}
