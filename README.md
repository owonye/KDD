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
# 1) 캘리브레이션(개발 구간에서 가중치/임계값 탐색)
python src/calibrate.py --mode hotpotqa --output results/calibration_hotpotqa.json
# 2) 평가(테스트 구간에서 베이스라인 비교)
python src/evaluate.py --mode hotpotqa --calibration-file .\results\calibration_hotpotqa.json --use-openai
# 3) 결과 요약(EM/F1/호출수/문서수/확장비율)
python src/summarize_results.py
```

Natural Questions:

```powershell
# 1) 캘리브레이션(개발 구간에서 가중치/임계값 탐색)
python src/calibrate.py --mode nq --output results/calibration_nq.json
# 2) 평가(테스트 구간에서 베이스라인 비교)
python src/evaluate.py --mode nq --calibration-file .\results\calibration_nq.json --use-openai
# 3) 결과 요약(EM/F1/호출수/문서수/확장비율)
python src/summarize_results.py
```

## 4) Split commands by size (100 / 300 / 1000)

### HotpotQA - 100

```powershell
# 캘리브레이션(질의/문서 100개)
python src/calibrate.py --mode hotpotqa --query-limit 100 --doc-limit 100 --output results/calib_hotpot_100.json
# 평가(위 캘리브레이션 결과 사용)
python src/evaluate.py --mode hotpotqa --query-limit 100 --doc-limit 100 --calibration-file .\results\calib_hotpot_100.json --use-openai --output results/eval_hotpot_100.csv
```

### HotpotQA - 300

```powershell
# 캘리브레이션(질의/문서 300개)
python src/calibrate.py --mode hotpotqa --query-limit 300 --doc-limit 300 --output results/calib_hotpot_300.json
# 평가(위 캘리브레이션 결과 사용)
python src/evaluate.py --mode hotpotqa --query-limit 300 --doc-limit 300 --calibration-file .\results\calib_hotpot_300.json --use-openai --output results/eval_hotpot_300.csv
```

### HotpotQA - 1000

```powershell
# 캘리브레이션(질의/문서 1000개)
python src/calibrate.py --mode hotpotqa --query-limit 1000 --doc-limit 1000 --output results/calib_hotpot_1000.json
# 평가(위 캘리브레이션 결과 사용)
python src/evaluate.py --mode hotpotqa --query-limit 1000 --doc-limit 1000 --calibration-file .\results\calib_hotpot_1000.json --use-openai --output results/eval_hotpot_1000.csv
```

### NQ - 100

```powershell
# 캘리브레이션(질의/문서 100개)
python src/calibrate.py --mode nq --query-limit 100 --doc-limit 100 --output results/calib_nq_100.json
# 평가(위 캘리브레이션 결과 사용)
python src/evaluate.py --mode nq --query-limit 100 --doc-limit 100 --calibration-file .\results\calib_nq_100.json --use-openai --output results/eval_nq_100.csv
```

### NQ - 300

```powershell
# 캘리브레이션(질의/문서 300개)
python src/calibrate.py --mode nq --query-limit 300 --doc-limit 300 --output results/calib_nq_300.json
# 평가(위 캘리브레이션 결과 사용)
python src/evaluate.py --mode nq --query-limit 300 --doc-limit 300 --calibration-file .\results\calib_nq_300.json --use-openai --output results/eval_nq_300.csv
```

### NQ - 1000

```powershell
# 캘리브레이션(질의/문서 1000개)
python src/calibrate.py --mode nq --query-limit 1000 --doc-limit 1000 --output results/calib_nq_1000.json
# 평가(위 캘리브레이션 결과 사용)
python src/evaluate.py --mode nq --query-limit 1000 --doc-limit 1000 --calibration-file .\results\calib_nq_1000.json --use-openai --output results/eval_nq_1000.csv
```

## 5) Auto runner

```powershell
# HotpotQA 전체 크기 실험(기본: calib/eval 질의 구간 분리)
python src/run_experiments.py --mode hotpotqa --sizes 100,300,1000 --use-openai
# NQ 전체 크기 실험(기본: calib/eval 질의 구간 분리)
python src/run_experiments.py --mode nq --sizes 100,300,1000 --use-openai
```

By default, `run_experiments.py` uses disjoint query slices for calibration and evaluation:

- calibration: `[query_start : query_start + size]`
- evaluation: `[query_start + size : query_start + 2*size]`

You can override this with:

```powershell
# calib/eval 질의 구간을 명시적으로 지정
python src/run_experiments.py --mode hotpotqa --sizes 300 --calib-query-start 0 --calib-query-limit 300 --eval-query-start 300 --eval-query-limit 300
```

To run ablations with re-calibration (`w/o relevance`, `w/o redundancy`, `w/o coverage`, `w/o supportiveness`):

```powershell
# 구조 신호 제거 ablation(각 신호 제거 후 재캘리브레이션 + 평가)
python src/run_experiments.py --mode hotpotqa --sizes 100,300,1000 --run-ablation
python src/run_experiments.py --mode nq --sizes 100,300,1000 --run-ablation
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
