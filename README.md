# KDD
# KDD RAG Prototype

Minimal prototype for a structure-aware adaptive retrieval paper idea:

- retrieve an initial set of documents
- analyze evidence structure
- estimate evidence sufficiency
- retrieve more only when needed

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/main.py
```

## Run modes

### 1. Demo mode

```powershell
python src/main.py --mode demo
```

### 2. HotpotQA + FAISS mode

```powershell
python src/main.py --mode hotpotqa
```

### 3. Natural Questions + FAISS mode

```powershell
python src/main.py --mode nq
```

### 4. Dataset mode + OpenAI generator

Set your API key first:

```powershell
$env:OPENAI_API_KEY="your_api_key"
```

Then run:

```powershell
python src/main.py --mode hotpotqa --use-openai
```

## Baseline comparison

Run all baselines and save a CSV:

```powershell
python src/evaluate.py --mode demo
```

or

```powershell
python src/evaluate.py --mode hotpotqa
```

or

```powershell
python src/evaluate.py --mode nq
```

You can also change retrieval budget:

```powershell
python src/evaluate.py --mode hotpotqa --initial-k 3 --expanded-k 5
```

This produces:

- `vanilla_rag`
- `fixed_large_k_rag`
- `confidence_adaptive_rag`
- `structure_aware_adaptive_rag`

Results are saved to `results/baseline_results.csv`.

## Calibration

You can calibrate the sufficiency estimator with lightweight silver labels:

```powershell
python src/calibrate.py --mode hotpotqa --output results/calibration_hotpotqa.json
```

Then reuse the calibrated weights:

```powershell
python src/evaluate.py --mode hotpotqa --calibration-file .\results\calibration_hotpotqa.json
```

## Result summary

```powershell
python src/summarize_results.py
```

This prints:

- average EM and F1
- average retrieval calls
- average document count
- reason counts for each baseline

## Current scope

- toy retriever
- evidence features: relevance, redundancy, diversity, supportiveness
- rule-based sufficiency estimator
- two actions: `answer_now` or `retrieve_more`

## Implemented next steps

- FAISS-based dense retriever
- OpenAI API based generator
- HotpotQA loader using the `datasets` library

## Notes

- `demo` mode runs without external downloads.
- `hotpotqa` and `nq` modes download the dataset and embedding model on first run.
- `--use-openai` requires `OPENAI_API_KEY`.
