# Extraction evaluation results

Examples evaluated: **6**
Total errors logged: **8**

## Per-field metrics

| field | TP | FP | FN | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| primary_site | 5 | 1 | 1 | 83.3% | 83.3% | 83.3% |
| histology | 6 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| stage | 6 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| ecog_performance_status | 5 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| line_of_therapy | 6 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| date_of_diagnosis | 6 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| biomarkers | 9 | 0 | 1 | 100.0% | 90.0% | 94.7% |
| treatment_regimen | 10 | 4 | 1 | 71.4% | 90.9% | 80.0% |
| macro_avg |  |  |  | 94.3% | 95.5% | 94.8% |
| micro_avg | 53 | 5 | 3 | 91.4% | 94.6% | 93.0% |

## Error taxonomy

| error_type | count | share |
|---|---:|---:|
| hallucinated | 4 | 50.0% |
| wrong_value | 1 | 12.5% |
| normalization | 1 | 12.5% |
| missed | 2 | 25.0% |
