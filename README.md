# KDD RAG Prototype

## Quick Start

```powershell
# 1) 가상환경 생성/활성화
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2) 의존성 설치
pip install -r requirements.txt
```

프로젝트 루트에 `.env` 파일 생성:

```dotenv
OPENAI_API_KEY=your_api_key_here
```

## 실험 실행 (이 2줄만 실행)

```powershell
# HotpotQA: 메인 + ablation
python src/run_experiments.py --mode hotpotqa --sizes 100,300,1000 --use-openai --run-ablation

# NQ: 메인 + ablation
python src/run_experiments.py --mode nq --sizes 100,300,1000 --use-openai --run-ablation
```

## 결과 확인

```powershell
# 예시: HotpotQA 300 결과 요약
python src/summarize_results.py --input results/eval_hotpotqa_300.csv
```

## 메모

- 총 실행: 명령 2번
- 생성 평가 파일 수: 30개
  - (데이터셋 2개) x (크기 3개) x (메인 1 + ablation 4)
