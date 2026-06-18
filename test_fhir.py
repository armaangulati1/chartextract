from datetime import date

from schema import Biomarker, BiomarkerStatus, CancerStage, EcogPerformanceStatus, OncologyExtract

from fhir import to_fhir, validate_fhir_bundle


def test_to_fhir_returns_valid_bundle():
    record = OncologyExtract(
        primary_site="lung",
        histology="adenocarcinoma",
        stage=CancerStage.IIIA,
        biomarkers=[Biomarker(name="EGFR", status=BiomarkerStatus.NEGATIVE)],
        ecog_performance_status=EcogPerformanceStatus.RESTRICTED_STRENUOUS,
        line_of_therapy=1,
        date_of_diagnosis=date(2023, 10, 9),
        treatment_regimen=["pembrolizumab", "carboplatin"],
    )

    bundle = to_fhir(record)

    assert bundle["resourceType"] == "Bundle"
    assert validate_fhir_bundle(bundle)
    assert len(bundle["entry"]) >= 5

    resource_types = {item["resource"]["resourceType"] for item in bundle["entry"]}
    assert "Condition" in resource_types
    assert "Observation" in resource_types
    assert "MedicationStatement" in resource_types


def test_to_fhir_empty_record_still_valid():
    bundle = to_fhir(OncologyExtract())
    assert bundle["resourceType"] == "Bundle"
    assert bundle["entry"] == []
    assert validate_fhir_bundle(bundle)
