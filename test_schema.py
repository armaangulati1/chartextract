from datetime import date

from schema import (
    Biomarker,
    BiomarkerStatus,
    CancerStage,
    EcogPerformanceStatus,
    OncologyExtract,
    export_json_schema,
)


def test_oncology_extract_validates():
    record = OncologyExtract(
        primary_site="lung",
        histology="adenocarcinoma",
        stage=CancerStage.IIIA,
        biomarkers=[Biomarker(name="EGFR", status=BiomarkerStatus.POSITIVE)],
        ecog_performance_status=EcogPerformanceStatus.RESTRICTED_STRENUOUS,
        line_of_therapy=2,
        date_of_diagnosis=date(2024, 3, 15),
        treatment_regimen=["osimertinib", "carboplatin"],
    )
    assert record.stage == CancerStage.IIIA
    assert record.ecog_performance_status == 1


def test_export_json_schema_has_contract_fields():
    schema = export_json_schema()
    props = schema["properties"]
    for field in (
        "primary_site",
        "histology",
        "stage",
        "biomarkers",
        "ecog_performance_status",
        "line_of_therapy",
        "date_of_diagnosis",
        "treatment_regimen",
    ):
        assert field in props
