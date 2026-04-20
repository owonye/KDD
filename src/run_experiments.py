import argparse
import subprocess
import sys
from pathlib import Path


def parse_sizes(raw: str) -> list[int]:
    values = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("At least one size is required.")
    return values


def run_command(command: list[str]) -> None:
    printable = " ".join(command)
    print(f"[RUN] {printable}")
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["hotpotqa", "nq"], required=True)
    parser.add_argument("--sizes", default="100,300,1000")
    parser.add_argument("--doc-start", type=int, default=0)
    parser.add_argument("--query-start", type=int, default=0)
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--initial-k", type=int, default=3)
    parser.add_argument("--expanded-k", type=int, default=5)
    parser.add_argument("--confidence-threshold", type=float, default=0.88)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--use-openai", action="store_true")
    args = parser.parse_args()

    sizes = parse_sizes(args.sizes)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for size in sizes:
        calib_path = output_dir / f"calib_{args.mode}_{size}.json"
        eval_path = output_dir / f"eval_{args.mode}_{size}.csv"

        calibrate_cmd = [
            sys.executable,
            "src/calibrate.py",
            "--mode",
            args.mode,
            "--doc-start",
            str(args.doc_start),
            "--doc-limit",
            str(size),
            "--query-start",
            str(args.query_start),
            "--query-limit",
            str(size),
            "--embedding-model",
            args.embedding_model,
            "--initial-k",
            str(args.initial_k),
            "--expanded-k",
            str(args.expanded_k),
            "--output",
            str(calib_path),
        ]
        run_command(calibrate_cmd)

        evaluate_cmd = [
            sys.executable,
            "src/evaluate.py",
            "--mode",
            args.mode,
            "--doc-start",
            str(args.doc_start),
            "--doc-limit",
            str(size),
            "--query-start",
            str(args.query_start),
            "--query-limit",
            str(size),
            "--embedding-model",
            args.embedding_model,
            "--initial-k",
            str(args.initial_k),
            "--expanded-k",
            str(args.expanded_k),
            "--confidence-threshold",
            str(args.confidence_threshold),
            "--calibration-file",
            str(calib_path),
            "--output",
            str(eval_path),
        ]
        if args.use_openai:
            evaluate_cmd.append("--use-openai")
        run_command(evaluate_cmd)

    print("[DONE] Completed all experiment sizes.")


if __name__ == "__main__":
    main()
