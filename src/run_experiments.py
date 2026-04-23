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


def parse_signal_list(raw: str) -> list[str]:
    allowed = {"relevance", "redundancy", "coverage", "supportiveness"}
    values = [token.strip() for token in raw.split(",") if token.strip()]
    if not values:
        return []
    unknown = [value for value in values if value not in allowed]
    if unknown:
        raise ValueError(f"Unknown ablation signal(s): {unknown}")
    return values


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
    parser.add_argument("--weak-support-overlap-threshold", type=float, default=0.2)
    parser.add_argument("--calib-query-start", type=int, default=None)
    parser.add_argument("--calib-query-limit", type=int, default=None)
    parser.add_argument("--eval-query-start", type=int, default=None)
    parser.add_argument("--eval-query-limit", type=int, default=None)
    parser.add_argument("--allow-overlap-splits", action="store_true")
    parser.add_argument("--run-ablation", action="store_true")
    parser.add_argument("--ablation-signals", default="relevance,redundancy,coverage,supportiveness")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--use-openai", action="store_true")
    args = parser.parse_args()

    sizes = parse_sizes(args.sizes)
    ablation_signals = parse_signal_list(args.ablation_signals)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for size in sizes:
        calib_path = output_dir / f"calib_{args.mode}_{size}.json"
        eval_path = output_dir / f"eval_{args.mode}_{size}.csv"
        calib_query_start = args.calib_query_start if args.calib_query_start is not None else args.query_start
        calib_query_limit = args.calib_query_limit if args.calib_query_limit is not None else size
        eval_query_start = args.eval_query_start if args.eval_query_start is not None else (calib_query_start + calib_query_limit)
        eval_query_limit = args.eval_query_limit if args.eval_query_limit is not None else size
        if not args.allow_overlap_splits:
            calib_end = calib_query_start + calib_query_limit
            eval_end = eval_query_start + eval_query_limit
            overlap = not (eval_query_start >= calib_end or calib_query_start >= eval_end)
            if overlap:
                raise ValueError(
                    "Calibration and evaluation query slices overlap. Use --allow-overlap-splits to bypass."
                )

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
            str(calib_query_start),
            "--query-limit",
            str(calib_query_limit),
            "--embedding-model",
            args.embedding_model,
            "--initial-k",
            str(args.initial_k),
            "--expanded-k",
            str(args.expanded_k),
            "--weak-support-overlap-threshold",
            str(args.weak_support_overlap_threshold),
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
            str(eval_query_start),
            "--query-limit",
            str(eval_query_limit),
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

        if args.run_ablation:
            for signal in ablation_signals:
                ablation_calib_path = output_dir / f"calib_{args.mode}_{size}_wo_{signal}.json"
                ablation_eval_path = output_dir / f"eval_{args.mode}_{size}_wo_{signal}.csv"
                ablation_calibrate_cmd = [
                    sys.executable,
                    "src/calibrate.py",
                    "--mode",
                    args.mode,
                    "--doc-start",
                    str(args.doc_start),
                    "--doc-limit",
                    str(size),
                    "--query-start",
                    str(calib_query_start),
                    "--query-limit",
                    str(calib_query_limit),
                    "--embedding-model",
                    args.embedding_model,
                    "--initial-k",
                    str(args.initial_k),
                    "--expanded-k",
                    str(args.expanded_k),
                    "--weak-support-overlap-threshold",
                    str(args.weak_support_overlap_threshold),
                    "--ablate-signal",
                    signal,
                    "--output",
                    str(ablation_calib_path),
                ]
                run_command(ablation_calibrate_cmd)

                ablation_evaluate_cmd = [
                    sys.executable,
                    "src/evaluate.py",
                    "--mode",
                    args.mode,
                    "--doc-start",
                    str(args.doc_start),
                    "--doc-limit",
                    str(size),
                    "--query-start",
                    str(eval_query_start),
                    "--query-limit",
                    str(eval_query_limit),
                    "--embedding-model",
                    args.embedding_model,
                    "--initial-k",
                    str(args.initial_k),
                    "--expanded-k",
                    str(args.expanded_k),
                    "--calibration-file",
                    str(ablation_calib_path),
                    "--baselines",
                    "structure_aware_adaptive_rag",
                    "--output",
                    str(ablation_eval_path),
                ]
                if args.use_openai:
                    ablation_evaluate_cmd.append("--use-openai")
                run_command(ablation_evaluate_cmd)

    print("[DONE] Completed all experiment sizes.")


if __name__ == "__main__":
    main()
