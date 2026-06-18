# Extraction experiment results

Gold set: `data/eval/ci_gold` (6 notes)

## Configuration comparison (per-field F1)

| field | single_pass_mini | pipeline_verifier_mini | Δ F1 (pp) |
|---|---:|---:|---:|
| primary_site | 100.0% | 83.3% | -16.7 |
| histology | 100.0% | 100.0% | +0.0 |
| stage | 66.7% | 100.0% | +33.3 |
| ecog_performance_status | 100.0% | 100.0% | +0.0 |
| line_of_therapy | 100.0% | 100.0% | +0.0 |
| date_of_diagnosis | 100.0% | 100.0% | +0.0 |
| biomarkers | 100.0% | 94.7% | -5.3 |
| treatment_regimen | 100.0% | 80.0% | -20.0 |
| **macro_avg** | **95.8%** | **94.8%** | **-1.1** |

## All configurations (macro-F1)

| config | mode | model | verifier | macro-F1 |
|---|---|---|---:|---:|
| single_pass_mini | single_pass | gpt-4o-mini | n/a | 95.8% |
| pipeline_no_verifier_mini | pipeline | gpt-4o-mini | no | 92.7% |
| pipeline_verifier_mini | pipeline | gpt-4o-mini | yes | 94.8% |
| single_pass_4o | single_pass | gpt-4o | n/a | 100.0% |

## Verifier impact: `single_pass_mini` → `pipeline_verifier_mini`

- **Errors fixed** (3):
  - `0028` **stage** (wrong_value): gold='III' pred='IIIB'
  - `0114` **primary_site** (normalization): gold='colorectal' pred='colon'
  - `0150` **stage** (wrong_value): gold='III' pred='IIIB'
- **Errors introduced** (7):
  - `0000` **biomarkers** (missed): gold="{'name': 'PSA', 'status': 'negative'}" pred=''
  - `0006` **treatment_regimen** (hallucinated): gold='' pred='irinotecan'
  - `0006` **treatment_regimen** (hallucinated): gold='' pred='fluorouracil'
  - `0006` **treatment_regimen** (hallucinated): gold='' pred='leucovorin'
  - `0006` **treatment_regimen** (missed): gold='folfiri' pred=''
  - `0114` **primary_site** (wrong_value): gold='colorectal' pred='proximal colon'
  - `0150` **treatment_regimen** (hallucinated): gold='' pred='lisinopril'

## Takeaway

On the 6-note CI gold set, **pipeline_verifier_mini** macro-F1 is 94.8% vs **95.8%** for **single_pass_mini** (-1.1 pp), so the agentic+verifier stack hurts aggregate accuracy—not vibes.

Per-field gains were strongest on **stage**; regressions appeared on **primary_site, biomarkers, treatment_regimen**. The verifier fixed 3 error(s) (notably primary_site, stage) and introduced 7 (biomarkers, primary_site, treatment_regimen).

Net: targeted extractors recover **stage** (+33 pp vs single-pass) but split regimens into component drugs (FOLFIRI → fluorouracil/irinotecan/leucovorin) and the verifier can drop low-signal biomarkers (PSA)—worth keeping the router/extractors, tightening regimen normalization, and raising the verifier threshold before production.

## Dataset evaluations

Production extract path: `extract_record()` → pipeline with verifier (`gpt-4o-mini`).

### Synthetic — CI gold (6 notes)

Examples: **6** · Errors logged: **8**

#### Per-field metrics

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

#### Error taxonomy

| error_type | count | share |
|---|---:|---:|
| hallucinated | 4 | 50.0% |
| wrong_value | 1 | 12.5% |
| missed | 3 | 37.5% |

### Real — MTSamples (50 notes)

Examples: **50** · Errors logged: **224**

#### Per-field metrics

| field | TP | FP | FN | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| primary_site | 17 | 21 | 29 | 44.7% | 37.0% | 40.5% |
| histology | 26 | 9 | 13 | 74.3% | 66.7% | 70.3% |
| stage | 3 | 1 | 2 | 75.0% | 60.0% | 66.7% |
| ecog_performance_status | 7 | 3 | 0 | 70.0% | 100.0% | 82.4% |
| line_of_therapy | 2 | 14 | 0 | 12.5% | 100.0% | 22.2% |
| date_of_diagnosis | 9 | 10 | 2 | 47.4% | 81.8% | 60.0% |
| biomarkers | 2 | 4 | 2 | 33.3% | 50.0% | 40.0% |
| treatment_regimen | 69 | 109 | 36 | 38.8% | 65.7% | 48.8% |
| macro_avg |  |  |  | 49.5% | 70.1% | 53.8% |
| micro_avg | 135 | 171 | 84 | 44.1% | 61.6% | 51.4% |

#### Error taxonomy

| error_type | count | share |
|---|---:|---:|
| hallucinated | 136 | 60.7% |
| wrong_value | 32 | 14.3% |
| wrong_span | 6 | 2.7% |
| normalization | 1 | 0.4% |
| missed | 49 | 21.9% |

### Synthetic vs real (summary)

| dataset | notes | macro-F1 | Δ vs synthetic |
|---|---:|---:|---:|
| synthetic (CI gold) | 6 | 93.6% | — |
| real (MTSamples) | 50 | 53.8% | -39.8 pp |

#### Per-field F1 gap (real − synthetic)

| field | synthetic | real | gap (pp) |
|---|---:|---:|---:|
| primary_site | 83.3% | 40.5% | -42.9 |
| histology | 100.0% | 70.3% | -29.7 |
| stage | 90.9% | 66.7% | -24.2 |
| ecog_performance_status | 100.0% | 82.4% | -17.6 |
| line_of_therapy | 100.0% | 22.2% | -77.8 |
| date_of_diagnosis | 100.0% | 60.0% | -40.0 |
| biomarkers | 94.7% | 40.0% | -54.7 |
| treatment_regimen | 80.0% | 48.8% | -31.2 |

#### Takeaway

On **50** real MTSamples oncology notes, macro-F1 is **53.8%** vs **93.6%** on synthetic CI gold (**-39.8 pp gap**)—expected degradation on messy real text.

Weakest real fields: **line_of_therapy, biomarkers, primary_site** (many notes lack explicit stage/line/biomarkers, so null-vs-extract mismatches dominate). Largest synthetic advantage: **primary_site, histology, stage, ecog_performance_status, line_of_therapy, date_of_diagnosis, biomarkers, treatment_regimen**.

Real notes are hematology-heavy consults with sparse structured oncology variables; improve primary_site/histology recall before trusting stage/regimen metrics on production charts.
