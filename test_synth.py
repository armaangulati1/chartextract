import random

from schema import CancerStage, OncologyExtract
from synth import sample_record, spot_check_pair


def test_sample_record_is_valid_oncology_extract():
    record = sample_record(random.Random(0))
    assert isinstance(record, OncologyExtract)
    assert record.primary_site in {
        "lung", "breast", "colorectal", "prostate", "melanoma",
        "pancreas", "ovary", "kidney", "head and neck",
    }
    assert record.stage in CancerStage
    assert record.line_of_therapy >= 1
    assert record.treatment_regimen


def test_sample_record_reproducible_with_seed():
    a = sample_record(random.Random(99))
    b = sample_record(random.Random(99))
    assert a.model_dump() == b.model_dump()


def test_spot_check_accepts_embedded_values():
    gold = sample_record(random.Random(1))
    note = (
        f"Oncology follow-up: {gold.primary_site} {gold.histology}, "
        f"stage {gold.stage.value if gold.stage else ''}, "
        f"ECOG {gold.ecog_performance_status.value if gold.ecog_performance_status is not None else ''}, "
        f"{gold.line_of_therapy}-line {', '.join(gold.treatment_regimen)}."
    )
    if gold.date_of_diagnosis:
        note += f" Diagnosed {gold.date_of_diagnosis.isoformat()}."
    for bm in gold.biomarkers:
        note += f" {bm.name} {bm.status.value}."
    misses = spot_check_pair({"note": note, "gold": gold.model_dump(mode="json")})
    assert misses == []
