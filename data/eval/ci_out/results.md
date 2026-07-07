# Extraction evaluation results

Examples evaluated: **6**
Total errors logged: **8**

## Per-field metrics

| field | TP | FP | FN | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| primary_site | 5 | 1 | 1 | 83.3% | 83.3% | 83.3% |
| histology | 6 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| stage | 5 | 0 | 1 | 100.0% | 83.3% | 90.9% |
| ecog_performance_status | 5 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| line_of_therapy | 6 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| date_of_diagnosis | 6 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| biomarkers | 9 | 0 | 1 | 100.0% | 90.0% | 94.7% |
| treatment_regimen | 10 | 4 | 1 | 71.4% | 90.9% | 80.0% |
| macro_avg |  |  |  | 94.3% | 93.4% | 93.6% |
| micro_avg | 52 | 5 | 4 | 91.2% | 92.9% | 92.0% |

## Error taxonomy

| error_type | count | share |
|---|---:|---:|
| hallucinated | 4 | 50.0% |
| wrong_value | 1 | 12.5% |
| missed | 3 | 37.5% |
