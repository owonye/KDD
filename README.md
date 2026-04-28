# STAR: Structure-Aware Adaptive Retrieval for RAG

This repository contains the experimental code for **STAR**, a structure-aware adaptive retrieval controller for retrieval-augmented generation (RAG). STAR decides whether to answer from the initially retrieved evidence or perform one additional retrieval step by diagnosing the current evidence set with four lightweight signals:

- retrieval relevance
- redundancy
- coverage
- supportiveness

The main experiments compare STAR against Vanilla RAG, Fixed larger-k RAG, and a confidence-based adaptive retrieval baseline on HotpotQA and Natural Questions.

## Repository Layout

```text
src/
  calibrate.py                 # Calibrates STAR and confidence thresholds
  evaluate.py                  # Runs RAG baselines and computes EM/F1
  extract_case_analysis.py     # Extracts representative disagreement cases
  summarize_results.py         # Prints baseline summaries from evaluation CSVs
  summarize_failure_modes.py   # Aggregates STAR-vs-confidence failure modes
  run_experiments.py           # End-to-end experiment runner
  rag/                         # Retrieval, signal, and generation utilities

scripts/
  plot_tradeoff.py             # Regenerates the quality-cost figure
  run_hotpotqa_smoke.sh        # Small smoke-test runner

assets/
  rag_tradeoff.png             # Quality-cost figure used in the paper
  rag_tradeoff.pdf             # Vector version of the same figure
```

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For paper-grade generation, set an OpenAI API key:

```bash
export OPENAI_API_KEY=your_api_key_here
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

You can also place the key in a local `.env` file. `.env` is ignored by git.

## Main Experimental Protocol

The paper uses:

- generator: `gpt-4.1-mini`
- prompt version: `short_answer_v2`
- dense encoder: `BAAI/bge-small-en-v1.5`
- corpus split: `validation`
- query split: `validation`
- document slice: `doc_start=0`, `doc_limit=20000`
- initial retrieval budget: `k_init=3`
- label strategy: `evidence`
- seed: `42`

For a size of 1,000, `run_experiments.py` uses disjoint query slices by default:

- calibration: query indices `0--999`
- evaluation: query indices `1000--1999`

Overlapping calibration/evaluation slices are rejected unless `--allow-overlap-splits` is explicitly set.

## Reproducing the Main Runs

### HotpotQA 1000 with Full Ablation

```bash
python src/run_experiments.py \
  --mode hotpotqa \
  --sizes 1000 \
  --doc-limit 20000 \
  --corpus-split validation \
  --query-split validation \
  --initial-k 3 \
  --expanded-k 8 \
  --label-strategy evidence \
  --use-openai \
  --run-ablation \
  --use-run-subdir \
  --run-name hotpot-v2-conf-target-final-ablation-1000-clean \
  --retrieval-cache-dir results/cache_shared \
  --openai-cache-path results/openai_cache_shared.jsonl
```

### Natural Questions 1000 with Full Ablation

```bash
python src/run_experiments.py \
  --mode nq \
  --sizes 1000 \
  --doc-limit 20000 \
  --corpus-split validation \
  --query-split validation \
  --initial-k 3 \
  --expanded-k 5 \
  --nq-max-tokens 180 \
  --nq-stride 90 \
  --label-strategy evidence \
  --use-openai \
  --run-ablation \
  --use-run-subdir \
  --run-name nq-v2-k5-chunk180-conf-target-final-ablation-1000-clean \
  --retrieval-cache-dir results/cache_shared \
  --openai-cache-path results/openai_cache_shared.jsonl
```

The shared retrieval and OpenAI caches are optional but useful for repeated runs.

## Summarizing Results

After a run finishes, summarize the latest matching output directory.

### HotpotQA

```bash
HOTPOT_DIR=$(ls -td results/*hotpot-v2-conf-target-final-ablation-1000-clean* | head -1)

python src/summarize_results.py \
  --input "$HOTPOT_DIR"/eval_hotpotqa_1000.csv

cat "$HOTPOT_DIR"/ablation_summary_hotpotqa_1000.csv
head -5 "$HOTPOT_DIR"/case_analysis_hotpotqa_1000.csv
```

### Natural Questions

```bash
NQ_DIR=$(ls -td results/*nq-v2-k5-chunk180-conf-target-final-ablation-1000-clean* | head -1)

python src/summarize_results.py \
  --input "$NQ_DIR"/eval_nq_1000.csv

cat "$NQ_DIR"/ablation_summary_nq_1000.csv
head -5 "$NQ_DIR"/case_analysis_nq_1000.csv
```

## Failure-Mode Analysis

The paper includes an aggregate analysis of cases where the confidence baseline stops but STAR expands.

```bash
python src/summarize_failure_modes.py \
  --input "$HOTPOT_DIR"/eval_hotpotqa_1000.csv \
  --output "$HOTPOT_DIR"/failure_modes_hotpotqa_1000.csv

python src/summarize_failure_modes.py \
  --input "$NQ_DIR"/eval_nq_1000.csv \
  --output "$NQ_DIR"/failure_modes_nq_1000.csv
```

The script reports:

- confidence premature stops
- STAR premature stops
- premature stops corrected by STAR
- confidence STOP / STAR EXPAND cases
- breakdowns by STAR reason, such as `high_redundancy` and `mixed_insufficiency`

## Regenerating the Quality-Cost Figure

The figure in `assets/rag_tradeoff.png` is generated from the paper table values:

```bash
python scripts/plot_tradeoff.py
```

This creates:

- `assets/rag_tradeoff.png`
- `assets/rag_tradeoff.pdf`

## Output Files

Each run writes artifacts under `results/`, typically including:

- `manifest_<dataset>_<size>.json`
- `calib_<dataset>_<size>.json`
- `confidence_calib_<dataset>_<size>.json`
- `eval_<dataset>_<size>.csv`
- `ablation_summary_<dataset>_<size>.csv`
- `case_analysis_<dataset>_<size>.csv`
- `failure_modes_<dataset>_<size>.csv` when generated
- sidecar `*.meta.json` files

The metadata files record the generator, prompt version, embedding model, calibration source, retrieval cache path, OpenAI cache path, and related run configuration.

## Notes

- Paper-grade runs require `--use-openai`.
- `--allow-simple-generator` is available only for lightweight debugging.
- `label_strategy=hybrid_generation` is supported as an optional sensitivity mode, but the default paper setting is `label_strategy=evidence`.
- The experiments are intentionally controlled: retrieval uses slice-based benchmark corpora and a one-step STOP/EXPAND policy.
