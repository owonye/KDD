#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if [ ! -f .env ]; then
  echo "Missing .env. Create .env with OPENAI_API_KEY before running." >&2
  exit 1
fi

mkdir -p results

python src/run_experiments.py \
  --mode hotpotqa \
  --sizes 100 \
  --paper-defaults \
  --use-openai \
  --run-ablation \
  --use-run-subdir \
  --run-name hotpot-v2-smoke
