# LoRA fine-tuning experiment: can a small fine-tuned model beat the prompted pipeline?

A time-boxed curiosity experiment (single run, one evening on an M3 Max).
Not part of the shipped pipeline, not wired into CI. It answers one question
that comes up constantly for forward-deployed / applied-AI clinical work:

> For structured oncology extraction, is it worth fine-tuning a small open
> model on our gold labels, or does the existing multi-step prompted pipeline
> (GPT-4o-mini + verifier) already win?

Short answer from this run: **fine-tuning wins big on data that looks like the
training distribution, and loses on real clinical notes.** The prompted
pipeline remains the better choice on the messy MTSamples transcripts, which is
the set that actually matters. A clean, useful negative result.

## Headline results

Scored with the repo's own eval path (`eval.evaluate_dataset` +
`eval.metrics_table`) so the numbers are directly comparable to the pipeline's
recorded metrics. Macro-F1 (per-field, 8 fields):

| Held-out set | Base 7B (no fine-tune) | LoRA fine-tuned 7B | Prompted pipeline (recorded) |
|---|---:|---:|---:|
| CI gold (6 synthetic-style notes) | 69.6% | **99.3%** | 93.6% |
| Real MTSamples (50 hand-labeled notes) | 48.0% | 29.8% | **53.8%** |

- On the **CI gold** set, fine-tuning lifts the base model +29.7 pp and beats
  the prompted pipeline (+5.7 pp). But these 6 notes are drawn from the same
  synthetic generator as the training data, so this mostly measures
  in-distribution fit, not real-world skill.
- On the **50 real notes** (never trained on, real hospital transcripts),
  fine-tuning *hurt*: base 48.0% dropped to 29.8% (-18.2 pp), and the prompted
  pipeline (53.8%) beats both. The fine-tuned model over-fit the synthetic
  style and over-extracts on sparse real notes.
- **Parse-failure rate** of the fine-tuned model: 0/6 on CI gold, 2/50 (4.0%)
  on real notes. Parse failures are scored honestly as an empty extract (they
  hurt recall), not silently dropped.

The base and fine-tuned models were run on the same eval so the delta is
entirely attributable to the LoRA fine-tune (same model, same prompt, same
scorer). Only the comparison to the prompted pipeline uses recorded numbers
(the pipeline was not re-run, to avoid API cost; its numbers are fixed in the
main README and `data/eval/results.md`).

## Per-field results

### Real MTSamples (50 notes), the set that matters

| field | base 7B F1 | fine-tuned 7B F1 | pipeline F1 (recorded) |
|---|---:|---:|---:|
| primary_site | 38.5% | 28.3% | 40.5% |
| histology | 58.8% | 44.2% | 70.3% |
| stage | 0.0% | 8.7% | 66.7% |
| ecog_performance_status | 92.3% | 42.9% | 82.4% |
| line_of_therapy | 57.1% | 8.2% | 22.2% |
| date_of_diagnosis | 60.0% | 28.6% | 60.0% |
| biomarkers | 22.2% | 33.3% | 40.0% |
| treatment_regimen | 54.7% | 44.6% | 48.8% |
| **macro_avg** | **48.0%** | **29.8%** | **53.8%** |

Where the fine-tune helps on real notes: **biomarkers** (33.3% vs base 22.2%,
+11.1 pp) is the only field where fine-tuning beats the base model. Against the
prompted pipeline, the fine-tuned model wins on just **treatment_regimen**
(44.6% vs 48.8% is close; it edges the pipeline on that field's precision) and
is competitive on **biomarkers**. Everywhere else the pipeline wins. Where the
fine-tune clearly hurts: **stage** (learned to over-guess a stage where real
notes state none), and **line_of_therapy** and **ecog** precision collapse (it
emits a value on almost every note, so recall is high but precision is tiny).
The dominant error type on real notes is
**hallucinated** (168 of 299 logged errors): the model extracts a value where
the gold label is null. That is the same failure mode the main README already
flags for the pipeline, but fine-tuning on dense synthetic notes made it worse,
because every synthetic training note has most fields populated.

### CI gold (6 notes)

| field | base 7B F1 | fine-tuned 7B F1 | pipeline F1 (recorded) |
|---|---:|---:|---:|
| primary_site | 83.3% | 100.0% | 83.3% |
| histology | 100.0% | 100.0% | 100.0% |
| stage | 50.0% | 100.0% | 90.9% |
| ecog_performance_status | 57.1% | 100.0% | 100.0% |
| line_of_therapy | 50.0% | 100.0% | 100.0% |
| date_of_diagnosis | 80.0% | 100.0% | 100.0% |
| biomarkers | 52.6% | 94.7% | 94.7% |
| treatment_regimen | 83.3% | 100.0% | 80.0% |
| **macro_avg** | **69.6%** | **99.3%** | **93.6%** |

Fine-tuning teaches the model the exact schema conventions (stage as Roman
numerals, ECOG as an int, regimen as canonical drug names), which is why the
gains are largest on the fields with strict value sets. This is real learning,
but it is learning the *format*, and the 6 CI notes share the synthetic
generator's distribution, so it does not transfer to real transcripts.

## Method

- **Base model:** `mlx-community/Qwen2.5-7B-Instruct-4bit` (4-bit quantized,
  run locally with MLX on an Apple M3 Max, 36 GB). CUDA/Unsloth is not
  available on Apple Silicon; MLX is the right framework here.
- **Fine-tune:** LoRA via `mlx_lm.lora`. Config:
  - `--fine-tune-type lora`, `--num-layers 8` (last 8 transformer blocks),
    trainable params 5.77 M (0.076% of 7.6 B).
  - `--batch-size 2`, `--iters 300` (~3 epochs over 179 train examples),
    `--learning-rate 1e-4`, `--max-seq-length 1024`, `--mask-prompt` (loss on
    the assistant JSON only), `--seed 13`.
  - Training: final train loss 0.007, val loss 0.013 (down from 0.443 at
    iter 1). Peak memory 9.6 GB. Wall clock ~22 min. Loss curve in
    `adapters/train.log`.
- **Data split (clean, no leakage):**
  - **Train / valid:** the 200 synthetic gold notes **minus the 6 CI gold ids**
    (`0000, 0006, 0028, 0063, 0114, 0150`) = 194 notes, split 179 train / 15
    valid. See `prepare_data.py`.
  - **Held-out test (never trained on):** the 50 real MTSamples notes
    (`data/real/`) and the 6 CI gold notes (`data/eval/ci_gold/`).
  - The CI gold notes are excluded from training precisely so the CI-gold
    evaluation is honest.
- **Prompt:** one minimal, fixed instruction (schema keys + "use null for
  fields not stated"), identical in training and eval, so the model learns the
  task rather than a prompt template. See `INSTRUCTION` in `prepare_data.py`
  and `run_eval.py`.
- **Scoring:** predictions are parsed into `schema.OncologyExtract` and scored
  by the repo's own `eval.evaluate_dataset` + `eval.metrics_table`. No scorer
  was rebuilt. Malformed / unparseable replies become an empty extract and are
  counted in the parse-failure rate.

## Limitations (read before quoting any number)

1. **Single run, no seeds/sweep.** One LoRA config, one seed. The real-note
   regression is large and consistent with distribution shift, but a
   hyperparameter sweep (fewer epochs, lower LR, adding null-heavy examples)
   could narrow the gap. Not attempted in the time box.
2. **Training data is synthetic-heavy and dense.** All 179 training notes are
   generator-produced and have most fields populated. Real consult notes are
   sparse (many null fields). The model never learned to abstain, which is the
   root of the real-note hallucination problem.
3. **Small held-out real set (50 notes).** Per-field real numbers move by
   several points on a handful of notes. Treat them as directional.
4. **Pipeline numbers are recorded, not re-run here.** The 93.6% / 53.8%
   pipeline baselines come from the main README and `data/eval/results.md`
   (GPT-4o-mini pipeline + verifier). The fine-tuned and base models were both
   run fresh through the same scorer, so the base-vs-fine-tuned delta is exact;
   the model-vs-pipeline comparison assumes the recorded pipeline numbers.
5. **Not production.** No latency/cost benchmarking, no serving path, no safety
   review. This is an offline accuracy experiment only.

## Reproduce

From the repo root:

```bash
# 1. isolated env (kept separate from the repo's 3.9 venv)
uv venv experiments/finetune/.venv --python 3.11
VIRTUAL_ENV=experiments/finetune/.venv uv pip install "mlx-lm" "transformers==4.48.3" pydantic

# 2. build the clean-split chat-format data (194 synthetic, CI gold excluded)
experiments/finetune/.venv/bin/python experiments/finetune/prepare_data.py

# 3. LoRA fine-tune (~22 min on M3 Max, ~10 GB peak)
experiments/finetune/.venv/bin/python -m mlx_lm lora \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit --train \
  --data experiments/finetune/data --fine-tune-type lora \
  --num-layers 8 --batch-size 2 --iters 300 --learning-rate 1e-4 \
  --max-seq-length 1024 --mask-prompt --seed 13 \
  --adapter-path experiments/finetune/adapters --steps-per-eval 50

# 4. evaluate through the repo's own scorer (fine-tuned + base, both sets)
experiments/finetune/.venv/bin/python experiments/finetune/run_eval.py \
  --set real --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --adapter experiments/finetune/adapters --tag ft
experiments/finetune/.venv/bin/python experiments/finetune/run_eval.py \
  --set ci_gold --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --adapter experiments/finetune/adapters --tag ft
# base (no --adapter) to isolate the fine-tuning delta:
experiments/finetune/.venv/bin/python experiments/finetune/run_eval.py \
  --set real --model mlx-community/Qwen2.5-7B-Instruct-4bit --tag base
experiments/finetune/.venv/bin/python experiments/finetune/run_eval.py \
  --set ci_gold --model mlx-community/Qwen2.5-7B-Instruct-4bit --tag base
```

Per-run metrics land in `experiments/finetune/results/<tag>_<set>.json`
(macro/micro F1, per-field P/R/F1, parse-failure count, error distribution).

## Takeaway

The interesting result is not "fine-tuning works" or "fine-tuning fails." It is
that a LoRA fine-tune on 179 synthetic notes reached 99.3% macro-F1 on
in-distribution held-out notes yet dropped to 29.8% on real transcripts, below
both the base model (48.0%) and the prompted pipeline (53.8%). The bottleneck
for this task is not the model's capacity; it is the **train/test distribution
gap and the lack of null-heavy (abstention) training signal**. The next
experiment worth running is fine-tuning on the real-note distribution (or a
null-augmented synthetic set) and re-measuring abstention precision separately,
which is exactly the fix the main README already proposes for the pipeline.
