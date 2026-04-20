# KDD RAG Prototype

Structure-aware adaptive retrieval prototype for RAG.

## 1) Setup (once)

```powershell
# create venv
python -m venv .venv

# activate venv
.venv\Scripts\Activate.ps1

# install deps
pip install -r requirements.txt
```

## 2) API key

Create `.env` at project root:

```dotenv
OPENAI_API_KEY=your_api_key_here
```

`.env` is ignored by git.

## 3) Core flow (dataset-level)

HotpotQA:

```powershell
python src/calibrate.py --mode hotpotqa --output results/calibration_hotpotqa.json
python src/evaluate.py --mode hotpotqa --calibration-file .\results\calibration_hotpotqa.json --use-openai
python src/summarize_results.py
```

Natural Questions:

```powershell
python src/calibrate.py --mode nq --output results/calibration_nq.json
python src/evaluate.py --mode nq --calibration-file .\results\calibration_nq.json --use-openai
python src/summarize_results.py
```

## 4) Split commands by size (100 / 300 / 1000)

### HotpotQA - 100

```powershell
python src/calibrate.py --mode hotpotqa --query-limit 100 --doc-limit 100 --output results/calib_hotpot_100.json
python src/evaluate.py --mode hotpotqa --query-limit 100 --doc-limit 100 --calibration-file .\results\calib_hotpot_100.json --use-openai --output results/eval_hotpot_100.csv
```

### HotpotQA - 300

```powershell
python src/calibrate.py --mode hotpotqa --query-limit 300 --doc-limit 300 --output results/calib_hotpot_300.json
python src/evaluate.py --mode hotpotqa --query-limit 300 --doc-limit 300 --calibration-file .\results\calib_hotpot_300.json --use-openai --output results/eval_hotpot_300.csv
```

### HotpotQA - 1000

```powershell
python src/calibrate.py --mode hotpotqa --query-limit 1000 --doc-limit 1000 --output results/calib_hotpot_1000.json
python src/evaluate.py --mode hotpotqa --query-limit 1000 --doc-limit 1000 --calibration-file .\results\calib_hotpot_1000.json --use-openai --output results/eval_hotpot_1000.csv
```

### NQ - 100

```powershell
python src/calibrate.py --mode nq --query-limit 100 --doc-limit 100 --output results/calib_nq_100.json
python src/evaluate.py --mode nq --query-limit 100 --doc-limit 100 --calibration-file .\results\calib_nq_100.json --use-openai --output results/eval_nq_100.csv
```

### NQ - 300

```powershell
python src/calibrate.py --mode nq --query-limit 300 --doc-limit 300 --output results/calib_nq_300.json
python src/evaluate.py --mode nq --query-limit 300 --doc-limit 300 --calibration-file .\results\calib_nq_300.json --use-openai --output results/eval_nq_300.csv
```

### NQ - 1000

```powershell
python src/calibrate.py --mode nq --query-limit 1000 --doc-limit 1000 --output results/calib_nq_1000.json
python src/evaluate.py --mode nq --query-limit 1000 --doc-limit 1000 --calibration-file .\results\calib_nq_1000.json --use-openai --output results/eval_nq_1000.csv
```

## 5) Auto runner

```powershell
python src/run_experiments.py --mode hotpotqa --sizes 100,300,1000 --use-openai
python src/run_experiments.py --mode nq --sizes 100,300,1000 --use-openai
```

## 6) Quick check

```powershell
python src/main.py --mode demo
```

## Notes

- Default baselines:
  - `vanilla_rag`
  - `fixed_large_k_rag`
  - `confidence_adaptive_rag`
  - `structure_aware_adaptive_rag`
- First run of `hotpotqa` / `nq` may download datasets and embedding model.
