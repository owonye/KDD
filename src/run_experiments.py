import argparse
import subprocess
import sys
from pathlib import Path

from experiment_utils import write_run_config


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


def resolve_doc_slice(
    policy: str,
    fixed_doc_start: int,
    fixed_doc_limit: int | None,
    calib_query_start: int,
    calib_query_limit: int,
    eval_query_start: int,
    eval_query_limit: int,
) -> tuple[int, int]:
    if policy == "fixed":
        if fixed_doc_limit is None:
            raise ValueError("--doc-limit is required when --doc-slice-policy=fixed.")
        return fixed_doc_start, fixed_doc_limit

    # query_union: include all examples touched by calib/eval query slices.
    start = min(calib_query_start, eval_query_start)
    end = max(calib_query_start + calib_query_limit, eval_query_start + eval_query_limit)
    return start, end - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["hotpotqa", "nq"], required=True)
    parser.add_argument("--sizes", default="100,300,1000")
    parser.add_argument("--doc-start", type=int, default=0)
    parser.add_argument("--doc-limit", type=int, default=None)
    parser.add_argument("--doc-slice-policy", choices=["query_union", "fixed"], default="query_union")
    parser.add_argument("--query-start", type=int, default=0)
    parser.add_argument("--corpus-split", default="train")
    parser.add_argument("--query-split", default="validation")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--initial-k", type=int, default=3)
    parser.add_argument("--expanded-k", type=int, default=5)
    parser.add_argument("--confidence-threshold", type=float, default=0.88)
    parser.add_argument("--weak-support-overlap-threshold", type=float, default=0.2)
    parser.add_argument("--label-strategy", choices=["evidence", "hybrid_generation"], default="evidence")
    parser.add_argument("--calib-query-start", type=int, default=None)
    parser.add_argument("--calib-query-limit", type=int, default=None)
    parser.add_argument("--eval-query-start", type=int, default=None)
    parser.add_argument("--eval-query-limit", type=int, default=None)
    parser.add_argument("--allow-overlap-splits", action="store_true")
    parser.add_argument("--run-ablation", action="store_true")
    parser.add_argument("--ablation-signals", default="relevance,redundancy,coverage,supportiveness")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--use-openai", action="store_true")
    parser.add_argument("--openai-model", default="gpt-4.1-mini")
    parser.add_argument("--allow-simple-generator", action="store_true")
    args = parser.parse_args()

    if not args.use_openai and not args.allow_simple_generator:
        raise ValueError("Paper-grade experiments require --use-openai (or --allow-simple-generator explicitly).")
    if args.label_strategy == "hybrid_generation" and not args.use_openai:
        raise ValueError("hybrid_generation label strategy requires --use-openai.")

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
                raise ValueError("Calibration/evaluation query slices overlap. Use --allow-overlap-splits to bypass.")

        if args.corpus_split != args.query_split and args.doc_slice_policy == "query_union":
            current_doc_start = args.doc_start
            current_doc_limit = args.doc_limit if args.doc_limit is not None else max(sizes)
        else:
            current_doc_start, current_doc_limit = resolve_doc_slice(
                policy=args.doc_slice_policy,
                fixed_doc_start=args.doc_start,
                fixed_doc_limit=args.doc_limit,
                calib_query_start=calib_query_start,
                calib_query_limit=calib_query_limit,
                eval_query_start=eval_query_start,
                eval_query_limit=eval_query_limit,
            )

        calibrate_cmd = [
            sys.executable,
            "src/calibrate.py",
            "--mode",
            args.mode,
            "--doc-start",
            str(current_doc_start),
            "--doc-limit",
            str(current_doc_limit),
            "--corpus-split",
            args.corpus_split,
            "--query-start",
            str(calib_query_start),
            "--query-limit",
            str(calib_query_limit),
            "--query-split",
            args.query_split,
            "--embedding-model",
            args.embedding_model,
            "--initial-k",
            str(args.initial_k),
            "--expanded-k",
            str(args.expanded_k),
            "--weak-support-overlap-threshold",
            str(args.weak_support_overlap_threshold),
            "--label-strategy",
            args.label_strategy,
            "--seed",
            str(args.seed),
            "--output",
            str(calib_path),
        ]
        if args.use_openai:
            calibrate_cmd.extend(["--use-openai", "--openai-model", args.openai_model])
        run_command(calibrate_cmd)

        evaluate_cmd = [
            sys.executable,
            "src/evaluate.py",
            "--mode",
            args.mode,
            "--doc-start",
            str(current_doc_start),
            "--doc-limit",
            str(current_doc_limit),
            "--corpus-split",
            args.corpus_split,
            "--query-start",
            str(eval_query_start),
            "--query-limit",
            str(eval_query_limit),
            "--query-split",
            args.query_split,
            "--embedding-model",
            args.embedding_model,
            "--initial-k",
            str(args.initial_k),
            "--expanded-k",
            str(args.expanded_k),
            "--confidence-threshold",
            str(args.confidence_threshold),
            "--seed",
            str(args.seed),
            "--calibration-file",
            str(calib_path),
            "--output",
            str(eval_path),
        ]
        if args.use_openai:
            evaluate_cmd.extend(["--use-openai", "--openai-model", args.openai_model])
        if args.allow_simple_generator:
            evaluate_cmd.append("--allow-simple-generator")
        run_command(evaluate_cmd)

        ablation_outputs: dict[str, dict[str, str]] = {}
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
                    str(current_doc_start),
                    "--doc-limit",
                    str(current_doc_limit),
                    "--corpus-split",
                    args.corpus_split,
                    "--query-start",
                    str(calib_query_start),
                    "--query-limit",
                    str(calib_query_limit),
                    "--query-split",
                    args.query_split,
                    "--embedding-model",
                    args.embedding_model,
                    "--initial-k",
                    str(args.initial_k),
                    "--expanded-k",
                    str(args.expanded_k),
                    "--weak-support-overlap-threshold",
                    str(args.weak_support_overlap_threshold),
                    "--label-strategy",
                    args.label_strategy,
                    "--ablate-signal",
                    signal,
                    "--seed",
                    str(args.seed),
                    "--output",
                    str(ablation_calib_path),
                ]
                if args.use_openai:
                    ablation_calibrate_cmd.extend(["--use-openai", "--openai-model", args.openai_model])
                run_command(ablation_calibrate_cmd)

                ablation_evaluate_cmd = [
                    sys.executable,
                    "src/evaluate.py",
                    "--mode",
                    args.mode,
                    "--doc-start",
                    str(current_doc_start),
                    "--doc-limit",
                    str(current_doc_limit),
                    "--corpus-split",
                    args.corpus_split,
                    "--query-start",
                    str(eval_query_start),
                    "--query-limit",
                    str(eval_query_limit),
                    "--query-split",
                    args.query_split,
                    "--embedding-model",
                    args.embedding_model,
                    "--initial-k",
                    str(args.initial_k),
                    "--expanded-k",
                    str(args.expanded_k),
                    "--seed",
                    str(args.seed),
                    "--calibration-file",
                    str(ablation_calib_path),
                    "--baselines",
                    "structure_aware_adaptive_rag",
                    "--output",
                    str(ablation_eval_path),
                ]
                if args.use_openai:
                    ablation_evaluate_cmd.extend(["--use-openai", "--openai-model", args.openai_model])
                if args.allow_simple_generator:
                    ablation_evaluate_cmd.append("--allow-simple-generator")
                run_command(ablation_evaluate_cmd)
                ablation_outputs[signal] = {
                    "calibration": str(ablation_calib_path),
                    "evaluation": str(ablation_eval_path),
                }

            ablation_summary_cmd = [
                sys.executable,
                "src/summarize_results.py",
                "--inputs-glob",
                f"results/eval_{args.mode}_{size}*.csv",
                "--ablation-output",
                f"results/ablation_summary_{args.mode}_{size}.csv",
            ]
            run_command(ablation_summary_cmd)

        case_analysis_cmd = [
            sys.executable,
            "src/extract_case_analysis.py",
            "--input",
            str(eval_path),
            "--output",
            f"results/case_analysis_{args.mode}_{size}.csv",
            "--top-n",
            "20",
        ]
        run_command(case_analysis_cmd)

        write_run_config(
            output_dir / f"run_{args.mode}_{size}.meta.json",
            {
                "script": "run_experiments.py",
                "args": vars(args),
                "size": size,
                "doc_slice": [current_doc_start, current_doc_start + current_doc_limit],
                "calibration_query_slice": [calib_query_start, calib_query_start + calib_query_limit],
                "evaluation_query_slice": [eval_query_start, eval_query_start + eval_query_limit],
                "outputs": {
                    "calibration": str(calib_path),
                    "evaluation": str(eval_path),
                    "ablation": ablation_outputs,
                },
            },
        )

    print("[DONE] Completed all experiment sizes.")


if __name__ == "__main__":
    main()
