import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import re

from experiment_utils import write_manifest, write_run_config
from rag.pipeline import GENERATOR_PROMPT_VERSION

DEFAULT_OPENAI_CACHE_PATH = "results/openai_cache.jsonl"
DEFAULT_RETRIEVAL_CACHE_DIR = "results/cache"


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


def slugify(value: str, max_len: int = 60) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    if not normalized:
        return "run"
    return normalized[:max_len].strip("-") or "run"


def build_run_subdir_name(args: argparse.Namespace, sizes: list[int]) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    size_token = "x".join(str(size) for size in sizes)
    base_name = f"{timestamp}_{args.mode}_s{size_token}_{args.label_strategy}"
    if args.run_ablation:
        base_name += "_ablation"
    if args.run_name:
        base_name += f"_{slugify(args.run_name)}"
    return slugify(base_name, max_len=120)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["hotpotqa", "nq"], required=True)
    parser.add_argument("--sizes", default="100,300,1000")
    parser.add_argument("--doc-start", type=int, default=0)
    parser.add_argument("--doc-limit", type=int, default=20000)
    parser.add_argument("--doc-slice-policy", choices=["fixed"], default="fixed")
    parser.add_argument("--query-start", type=int, default=0)
    parser.add_argument("--corpus-split", default="train")
    parser.add_argument("--query-split", default="validation")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--nq-max-tokens", type=int, default=220)
    parser.add_argument("--nq-stride", type=int, default=110)
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
    parser.add_argument("--paper-defaults", action="store_true")
    parser.add_argument("--manifest-out", default="")
    parser.add_argument("--confidence-calibration-out", default="")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--use-run-subdir", action="store_true")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--use-openai", action="store_true")
    parser.add_argument("--openai-model", default="gpt-4.1-mini")
    parser.add_argument("--openai-cache-path", default=DEFAULT_OPENAI_CACHE_PATH)
    parser.add_argument("--retrieval-cache-dir", default=DEFAULT_RETRIEVAL_CACHE_DIR)
    parser.add_argument("--allow-simple-generator", action="store_true")
    args = parser.parse_args()

    if not args.use_openai and not args.allow_simple_generator:
        raise ValueError("Paper-grade experiments require --use-openai (or --allow-simple-generator explicitly).")
    if args.label_strategy == "hybrid_generation" and not args.use_openai:
        raise ValueError("hybrid_generation label strategy requires --use-openai.")
    if args.doc_limit <= 0:
        raise ValueError("--doc-limit must be positive.")

    if args.paper_defaults:
        args.doc_limit = 20000
        args.doc_slice_policy = "fixed"
        args.query_split = "validation"
        args.corpus_split = args.query_split
        args.initial_k = 3
        args.expanded_k = 8
        args.label_strategy = "evidence"

    sizes = parse_sizes(args.sizes)
    ablation_signals = parse_signal_list(args.ablation_signals)
    output_base_dir = Path(args.output_dir)
    output_dir = output_base_dir / build_run_subdir_name(args, sizes) if args.use_run_subdir else output_base_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.use_run_subdir and args.openai_cache_path == DEFAULT_OPENAI_CACHE_PATH:
        args.openai_cache_path = str(output_dir / "openai_cache.jsonl")
    if args.use_run_subdir and args.retrieval_cache_dir == DEFAULT_RETRIEVAL_CACHE_DIR:
        args.retrieval_cache_dir = str(output_dir / "cache")

    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] OpenAI cache path: {args.openai_cache_path}")
    print(f"[INFO] Retrieval cache directory: {args.retrieval_cache_dir}")

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

        current_doc_start = args.doc_start
        current_doc_limit = args.doc_limit

        manifest_path = Path(args.manifest_out) if args.manifest_out else output_dir / f"manifest_{args.mode}_{size}.json"
        manifest = write_manifest(
            manifest_path,
            {
                "dataset": args.mode,
                "corpus_split": args.corpus_split,
                "query_split": args.query_split,
                "doc_start": current_doc_start,
                "doc_limit": current_doc_limit,
                "calib_query_start": calib_query_start,
                "calib_query_limit": calib_query_limit,
                "eval_query_start": eval_query_start,
                "eval_query_limit": eval_query_limit,
                "initial_k": args.initial_k,
                "expanded_k": args.expanded_k,
                "label_strategy": args.label_strategy,
                "embedding_model": args.embedding_model,
                "nq_max_tokens": args.nq_max_tokens,
                "nq_stride": args.nq_stride,
                "seed": args.seed,
                "prompt_version": GENERATOR_PROMPT_VERSION,
                "retrieval_cache_dir": args.retrieval_cache_dir,
            },
        )

        confidence_calib_path = (
            Path(args.confidence_calibration_out)
            if args.confidence_calibration_out
            else output_dir / f"confidence_calib_{args.mode}_{size}.json"
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
            "--retrieval-cache-dir",
            args.retrieval_cache_dir,
            "--nq-max-tokens",
            str(args.nq_max_tokens),
            "--nq-stride",
            str(args.nq_stride),
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
            "--manifest-path",
            str(manifest_path),
            "--confidence-calibration-out",
            str(confidence_calib_path),
            "--output",
            str(calib_path),
        ]
        if args.use_openai:
            calibrate_cmd.extend(
                [
                    "--use-openai",
                    "--openai-model",
                    args.openai_model,
                    "--openai-cache-path",
                    args.openai_cache_path,
                ]
            )
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
            "--retrieval-cache-dir",
            args.retrieval_cache_dir,
            "--nq-max-tokens",
            str(args.nq_max_tokens),
            "--nq-stride",
            str(args.nq_stride),
            "--initial-k",
            str(args.initial_k),
            "--expanded-k",
            str(args.expanded_k),
            "--confidence-threshold",
            str(args.confidence_threshold),
            "--weak-support-overlap-threshold",
            str(args.weak_support_overlap_threshold),
            "--seed",
            str(args.seed),
            "--manifest-path",
            str(manifest_path),
            "--calibration-file",
            str(calib_path),
            "--confidence-calibration-file",
            str(confidence_calib_path),
            "--output",
            str(eval_path),
        ]
        if args.use_openai:
            evaluate_cmd.extend(
                [
                    "--use-openai",
                    "--openai-model",
                    args.openai_model,
                    "--openai-cache-path",
                    args.openai_cache_path,
                ]
            )
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
                    "--retrieval-cache-dir",
                    args.retrieval_cache_dir,
                    "--nq-max-tokens",
                    str(args.nq_max_tokens),
                    "--nq-stride",
                    str(args.nq_stride),
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
                    "--manifest-path",
                    str(manifest_path),
                    "--output",
                    str(ablation_calib_path),
                ]
                if args.use_openai:
                    ablation_calibrate_cmd.extend(
                        [
                            "--use-openai",
                            "--openai-model",
                            args.openai_model,
                            "--openai-cache-path",
                            args.openai_cache_path,
                        ]
                    )
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
                    "--retrieval-cache-dir",
                    args.retrieval_cache_dir,
                    "--nq-max-tokens",
                    str(args.nq_max_tokens),
                    "--nq-stride",
                    str(args.nq_stride),
                    "--initial-k",
                    str(args.initial_k),
                    "--expanded-k",
                    str(args.expanded_k),
                    "--weak-support-overlap-threshold",
                    str(args.weak_support_overlap_threshold),
                    "--seed",
                    str(args.seed),
                    "--manifest-path",
                    str(manifest_path),
                    "--calibration-file",
                    str(ablation_calib_path),
                    "--baselines",
                    "structure_aware_adaptive_rag",
                    "--output",
                    str(ablation_eval_path),
                ]
                if args.use_openai:
                    ablation_evaluate_cmd.extend(
                        [
                            "--use-openai",
                            "--openai-model",
                            args.openai_model,
                            "--openai-cache-path",
                            args.openai_cache_path,
                        ]
                    )
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
                str(output_dir / f"eval_{args.mode}_{size}*.csv"),
                "--ablation-output",
                str(output_dir / f"ablation_summary_{args.mode}_{size}.csv"),
            ]
            run_command(ablation_summary_cmd)

        case_analysis_cmd = [
            sys.executable,
            "src/extract_case_analysis.py",
            "--input",
            str(eval_path),
            "--output",
            str(output_dir / f"case_analysis_{args.mode}_{size}.csv"),
            "--top-n",
            "20",
            "--balance-by-reason",
            "--max-per-reason",
            "5",
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
                    "confidence_calibration": str(confidence_calib_path),
                    "ablation": ablation_outputs,
                },
                "generator_type": "openai" if args.use_openai else "simple_placeholder",
                "model_version": args.openai_model if args.use_openai else "simple_placeholder",
                "label_strategy": args.label_strategy,
                "ablate_signal": ablation_signals if args.run_ablation else [],
                "initial_k": args.initial_k,
                "expanded_k": args.expanded_k,
                "doc_limit": args.doc_limit,
                "corpus_split": args.corpus_split,
                "query_split": args.query_split,
                "retrieval_cache_dir": args.retrieval_cache_dir,
                "nq_max_tokens": args.nq_max_tokens,
                "nq_stride": args.nq_stride,
                "manifest_path": str(manifest_path),
                "manifest_id": manifest["manifest_id"],
                "prompt_version": manifest["prompt_version"],
            },
        )

    print("[DONE] Completed all experiment sizes.")


if __name__ == "__main__":
    main()
