# KDD RAG Prototype

## 1) 환경 설정

```powershell
# 가상환경 생성
python -m venv .venv

# 가상환경 활성화
.venv\Scripts\Activate.ps1

# 의존성 설치
pip install -r requirements.txt
```

프로젝트 루트 `.env` 파일:

```dotenv
OPENAI_API_KEY=your_api_key_here
```

## 2) 논문용 실행 전제

- 논문용 EM/F1 평가는 `--use-openai`를 사용해야 합니다.
- 기본 split은 `corpus_split=train`, `query_split=validation`입니다.
- `corpus_split != query_split`이고 `doc-slice-policy=query_union`이면 `--doc-limit`을 반드시 지정해야 합니다.
- 결과는 `--output-dir` 아래에 저장됩니다.

## 3) 메인 실험 (권장)

```powershell
# HotpotQA: 메인 + ablation
python src/run_experiments.py --mode hotpotqa --sizes 100,300,1000 --use-openai --run-ablation --output-dir results --doc-limit 20000 --label-strategy evidence

# NQ: 메인 + ablation
python src/run_experiments.py --mode nq --sizes 100,300,1000 --use-openai --run-ablation --output-dir results --doc-limit 20000 --label-strategy evidence
```

## 4) 선택 옵션

```powershell
# hybrid_generation 라벨 전략(비용 증가)
python src/run_experiments.py --mode hotpotqa --sizes 100 --use-openai --run-ablation --label-strategy hybrid_generation --output-dir results --doc-limit 20000
```

```powershell
# split/구간 수동 지정
python src/run_experiments.py --mode hotpotqa --sizes 300 --use-openai --output-dir results --doc-limit 20000 --corpus-split train --query-split validation --calib-query-start 0 --calib-query-limit 300 --eval-query-start 300 --eval-query-limit 300
```

## 5) 결과 파일

- 메인 캘리브레이션: `results/calib_<mode>_<size>.json`
- 메인 평가: `results/eval_<mode>_<size>.csv`
- ablation 캘리브레이션: `results/calib_<mode>_<size>_wo_<signal>.json`
- ablation 평가: `results/eval_<mode>_<size>_wo_<signal>.csv`
- ablation 요약표: `results/ablation_summary_<mode>_<size>.csv`
- case analysis: `results/case_analysis_<mode>_<size>.csv`
- 실행 설정 로그: `results/run_<mode>_<size>.meta.json`
- calibrate/evaluate 설정 로그: 각 출력 파일 옆 `*.meta.json`

## 6) 추가 분석 명령

```powershell
# 여러 CSV 통합 요약 + ablation 표 저장
python src/summarize_results.py --inputs-glob "results/eval_hotpotqa_300*.csv" --ablation-output results/ablation_summary_hotpotqa_300.csv
```

```powershell
# 케이스 분석 재생성
python src/extract_case_analysis.py --input results/eval_hotpotqa_300.csv --output results/case_analysis_hotpotqa_300.csv --top-n 20
```
