# KDD Adaptive RAG Experiments

논문 기본 프로토콜 기준으로 실행하는 실험 코드입니다.

## 1) 환경 변수

프로젝트 루트에 `.env`:

```dotenv
OPENAI_API_KEY=your_api_key_here
```

## 2) 논문 기본 프로토콜

- 고정 코퍼스 슬라이스: `corpus_split=validation`, `doc_start=0`, `doc_limit=20000`
- 질의 분할: `query_split=validation`
- one-step 정책: `k_init=3`, `k_exp=8`, `STOP/EXPAND`
- calibration 기본: `label_strategy=evidence`
- `query_union` 정책은 사용하지 않음

## 3) 메인 실행

```powershell
# HotpotQA: 100/300/1000, 메인+ablation
python src/run_experiments.py --mode hotpotqa --sizes 100,300,1000 --paper-defaults --use-openai --run-ablation

# Natural Questions: 100/300/1000, 메인+ablation
python src/run_experiments.py --mode nq --sizes 100,300,1000 --paper-defaults --use-openai --run-ablation
```

## 4) 캐시 (속도 최적화)

기본값으로 retrieval 캐시가 켜져 있습니다.

- 캐시 루트: `results/cache`
- 임베딩 캐시: `results/cache/embeddings/*.npz`
- FAISS 인덱스 캐시: `results/cache/faiss/*.index`
- 키 기준: `dataset + corpus_split + doc_range + embedding_model`

동일한 코퍼스 범위로 반복 실행하면 임베딩/인덱스 재계산을 대부분 건너뜁니다.

캐시 경로 변경:

```powershell
python src/run_experiments.py --mode hotpotqa --sizes 300 --paper-defaults --use-openai --retrieval-cache-dir results/cache
```

## 5) 출력 파일

- manifest: `results/manifest_<mode>_<size>.json`
- 구조형 calibration: `results/calib_<mode>_<size>.json`
- confidence calibration: `results/confidence_calib_<mode>_<size>.json`
- 평가 CSV: `results/eval_<mode>_<size>.csv`
- ablation calibration/eval:
  - `results/calib_<mode>_<size>_wo_<signal>.json`
  - `results/eval_<mode>_<size>_wo_<signal>.csv`
- ablation 요약: `results/ablation_summary_<mode>_<size>.csv`
- case analysis: `results/case_analysis_<mode>_<size>.csv`
- 실행 메타: `results/*.meta.json`

## 6) 선택 옵션

`hybrid_generation`은 sensitivity check 용도(비용/시간 증가):

```powershell
python src/run_experiments.py --mode hotpotqa --sizes 100 --paper-defaults --use-openai --label-strategy hybrid_generation
```

## 7) 참고

비용/시간을 줄이려면:

- 먼저 `--sizes 100`으로 calibration/동작 확인
- 이후 `300`, `1000` 순서로 확대
- 같은 `doc_start/doc_limit/split/model`을 유지해 캐시 적중률 확보
