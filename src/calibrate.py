import argparse
import json
import re
import time
from itertools import product
from pathlib import Path

from dotenv import load_dotenv

from experiment_utils import load_manifest, set_global_seed, write_run_config
from rag.pipeline import (
    FaissRetriever,
    OpenAIGenerator,
    Query,
    SimpleRetriever,
    SufficiencyEstimator,
    build_demo_corpus,
    build_silver_label,
    compute_query_overlap,
    embed_corpus_texts,
    extract_evidence_features,
    GENERATOR_PROMPT_VERSION,
    load_hotpotqa_queries,
    load_hotpotqa_sample,
    min_max_normalize,
    load_nq_queries,
    load_nq_sample,
)


def build_resources(
    mode: str,
    doc_start: int,
    doc_limit: int,
    corpus_split: str,
    query_start: int,
    query_limit: int,
    query_split: str,
    embedding_model: str,
    retrieval_cache_dir: str,
    nq_max_tokens: int,
    nq_stride: int,
):
    if mode == "demo":
        corpus = build_demo_corpus()
        queries = [Query("When is the birthday of Michael Phelps?", answer="June 30, 1985", answers=["June 30, 1985"])]
        retriever = SimpleRetriever(corpus)
        return queries, retriever

    if mode == "hotpotqa":
        raw_docs = load_hotpotqa_sample(start=doc_start, limit=doc_limit, split=corpus_split)
        cache_namespace = f"hotpotqa::{corpus_split}::{doc_start}:{doc_start + doc_limit}"
        corpus = embed_corpus_texts(
            raw_docs,
            model_name=embedding_model,
            cache_dir=retrieval_cache_dir,
            cache_namespace=cache_namespace,
        )
        queries = load_hotpotqa_queries(start=query_start, limit=query_limit, split=query_split)
        retriever = FaissRetriever(
            corpus,
            model_name=embedding_model,
            cache_dir=retrieval_cache_dir,
            cache_namespace=cache_namespace,
        )
        return queries, retriever

    raw_docs = load_nq_sample(
        start=doc_start,
        limit=doc_limit,
        split=corpus_split,
        max_tokens=nq_max_tokens,
        stride=nq_stride,
    )
    cache_namespace = f"nq::{corpus_split}::{doc_start}:{doc_start + doc_limit}::chunk_{nq_max_tokens}_{nq_stride}"
    corpus = embed_corpus_texts(
        raw_docs,
        model_name=embedding_model,
        cache_dir=retrieval_cache_dir,
        cache_namespace=cache_namespace,
    )
    queries = load_nq_queries(start=query_start, limit=query_limit, split=query_split)
    retriever = FaissRetriever(
        corpus,
        model_name=embedding_model,
        cache_dir=retrieval_cache_dir,
        cache_namespace=cache_namespace,
    )
    return queries, retriever


def resolve_manifest_overrides(args: argparse.Namespace) -> argparse.Namespace:
    manifest = load_manifest(args.manifest_path)
    if not manifest:
        args.manifest_id = None
        return args

    args.manifest_id = manifest.get("manifest_id")
    args.doc_start = int(manifest["doc_start"])
    args.doc_limit = int(manifest["doc_limit"])
    args.corpus_split = str(manifest["corpus_split"])
    args.query_start = int(manifest["calib_query_start"])
    args.query_limit = int(manifest["calib_query_limit"])
    args.query_split = str(manifest["query_split"])
    args.initial_k = int(manifest["initial_k"])
    args.expanded_k = int(manifest["expanded_k"])
    args.embedding_model = str(manifest["embedding_model"])
    args.seed = int(manifest["seed"])
    args.retrieval_cache_dir = str(manifest.get("retrieval_cache_dir", args.retrieval_cache_dir))
    args.nq_max_tokens = int(manifest.get("nq_max_tokens", args.nq_max_tokens))
    args.nq_stride = int(manifest.get("nq_stride", args.nq_stride))
    args.expansion_selection_tolerance = float(
        manifest.get("expansion_selection_tolerance", args.expansion_selection_tolerance)
    )
    if args.label_strategy == "evidence":
        args.label_strategy = str(manifest.get("label_strategy", args.label_strategy))
    return args


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def f1_score(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = {}
    for token in pred_tokens:
        common[token] = min(pred_tokens.count(token), gold_tokens.count(token))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def evidence_has_weak_support(query: Query, docs, overlap_threshold: float) -> bool:
    gold_answers = query.answers if query.answers else ([query.answer] if query.answer else [])
    if not gold_answers:
        return False
    normalized_gold = [normalize_answer(answer) for answer in gold_answers if answer]
    normalized_gold = [answer for answer in normalized_gold if answer]
    if not normalized_gold:
        return False
    for doc in docs:
        doc_text = normalize_answer(doc.text)
        has_match = any(gold in doc_text for gold in normalized_gold)
        if has_match and compute_query_overlap(query, doc) >= overlap_threshold:
            return True
    return False


def is_monotonic_configuration(
    relevance_weight: float,
    coverage_weight: float,
    supportiveness_weight: float,
    redundancy_weight: float,
) -> bool:
    # F(D,q)=wr*R - wu*U + wc*C + ws*S follows monotonicity
    # when all weights are non-negative.
    return all(weight >= 0.0 for weight in [relevance_weight, coverage_weight, supportiveness_weight, redundancy_weight])


def get_weight_grid(ablate_signal: str) -> tuple[list[float], list[float], list[float], list[float]]:
    default_grid = [0.0, 0.1, 0.2, 0.35, 0.5, 0.75]
    wr_grid = [0.0] if ablate_signal == "relevance" else default_grid
    wc_grid = [0.0] if ablate_signal == "coverage" else default_grid
    ws_grid = [0.0] if ablate_signal == "supportiveness" else default_grid
    wu_grid = [0.0] if ablate_signal == "redundancy" else default_grid
    return wr_grid, wc_grid, ws_grid, wu_grid


def label_from_evidence(
    query: Query,
    retriever,
    initial_k: int,
    expanded_k: int,
    overlap_threshold: float,
    embedding_model: str,
):
    initial_docs = retriever.retrieve(query, initial_k)
    expanded_docs = retriever.retrieve(query, expanded_k)
    initial_features = extract_evidence_features(query, initial_docs, aspect_model=embedding_model)
    initial_correct = evidence_has_weak_support(query, initial_docs, overlap_threshold=overlap_threshold)
    expanded_correct = evidence_has_weak_support(query, expanded_docs, overlap_threshold=overlap_threshold)
    return initial_features, build_silver_label(initial_correct, expanded_correct)


def label_from_hybrid_generation(
    query: Query,
    retriever,
    generator,
    initial_k: int,
    expanded_k: int,
    overlap_threshold: float,
    embedding_model: str,
):
    initial_docs = retriever.retrieve(query, initial_k)
    expanded_docs = retriever.retrieve(query, expanded_k)
    initial_features = extract_evidence_features(query, initial_docs, aspect_model=embedding_model)
    initial_support = evidence_has_weak_support(query, initial_docs, overlap_threshold=overlap_threshold)
    expanded_support = evidence_has_weak_support(query, expanded_docs, overlap_threshold=overlap_threshold)

    gold_answers = query.answers if query.answers else ([query.answer] if query.answer else [])
    if gold_answers:
        initial_answer = generator.generate(query, initial_docs)
        expanded_answer = generator.generate(query, expanded_docs)
        normalized_gold = [normalize_answer(answer) for answer in gold_answers if answer]
        initial_em = 1 if any(normalize_answer(initial_answer) == gold for gold in normalized_gold) else 0
        expanded_em = 1 if any(normalize_answer(expanded_answer) == gold for gold in normalized_gold) else 0
        initial_f1 = max(f1_score(initial_answer, gold) for gold in gold_answers)
        expanded_f1 = max(f1_score(expanded_answer, gold) for gold in gold_answers)
        initial_good = initial_support and (initial_em == 1 or initial_f1 >= 0.8)
        expanded_good = expanded_support and (expanded_em == 1 or expanded_f1 >= 0.8)
        return initial_features, build_silver_label(initial_good, expanded_good)

    return initial_features, build_silver_label(initial_support, expanded_support)


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["demo", "hotpotqa", "nq"], default="demo")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--doc-start", type=int, default=0)
    parser.add_argument("--doc-limit", type=int, default=20000)
    parser.add_argument("--corpus-split", default="train")
    parser.add_argument("--query-start", type=int, default=0)
    parser.add_argument("--query-limit", type=int, default=100)
    parser.add_argument("--query-split", default="validation")
    parser.add_argument("--initial-k", type=int, default=3)
    parser.add_argument("--expanded-k", type=int, default=5)
    parser.add_argument("--weak-support-overlap-threshold", type=float, default=0.2)
    parser.add_argument("--label-strategy", choices=["evidence", "hybrid_generation"], default="evidence")
    parser.add_argument("--use-openai", action="store_true")
    parser.add_argument("--openai-model", default="gpt-4.1-mini")
    parser.add_argument("--openai-cache-path", default="results/openai_cache.jsonl")
    parser.add_argument("--retrieval-cache-dir", default="results/cache")
    parser.add_argument("--nq-max-tokens", type=int, default=220)
    parser.add_argument("--nq-stride", type=int, default=110)
    parser.add_argument(
        "--expansion-selection-tolerance",
        type=float,
        default=0.0,
        help=(
            "Allow a lower-expansion controller when balanced accuracy is within "
            "this absolute tolerance of the current best calibration score."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--confidence-calibration-out", default="")
    parser.add_argument(
        "--ablate-signal",
        choices=["none", "relevance", "redundancy", "coverage", "supportiveness"],
        default="none",
    )
    parser.add_argument("--output", default="results/calibration.json")
    args = parser.parse_args()
    args = resolve_manifest_overrides(args)
    set_global_seed(args.seed)

    def _format_eta(seconds: float) -> str:
        if seconds < 0:
            seconds = 0
        total = int(seconds)
        hours, rem = divmod(total, 3600)
        minutes, secs = divmod(rem, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def _maybe_log_progress(stage: str, completed: int, total: int, started_at: float, force: bool = False) -> None:
        if total <= 0:
            return
        interval = max(1, total // 20)  # ~5% step
        if not force and completed % interval != 0 and completed != total:
            return
        elapsed = max(time.time() - started_at, 1e-9)
        rate = completed / elapsed
        remaining = max(total - completed, 0)
        eta = remaining / rate if rate > 0 else 0.0
        progress = (completed / total) * 100
        print(
            f"[PROGRESS] {stage}: {completed}/{total} ({progress:.1f}%) "
            f"elapsed={_format_eta(elapsed)} eta={_format_eta(eta)}"
        )

    queries, retriever = build_resources(
        args.mode,
        args.doc_start,
        args.doc_limit,
        args.corpus_split,
        args.query_start,
        args.query_limit,
        args.query_split,
        args.embedding_model,
        args.retrieval_cache_dir,
        args.nq_max_tokens,
        args.nq_stride,
    )
    if args.label_strategy == "hybrid_generation" and not args.use_openai:
        raise ValueError("hybrid_generation label strategy requires --use-openai.")

    examples = []
    excluded_no_support = 0
    generator = OpenAIGenerator(model=args.openai_model, cache_path=args.openai_cache_path) if args.use_openai else None
    feature_aspect_model = "" if args.mode == "demo" else args.embedding_model
    total_queries = len(queries)
    stage_started = time.time()
    for idx, query in enumerate(queries, start=1):
        if args.label_strategy == "hybrid_generation":
            initial_features, silver_label = label_from_hybrid_generation(
                query,
                retriever,
                generator,
                initial_k=args.initial_k,
                expanded_k=args.expanded_k,
                overlap_threshold=args.weak_support_overlap_threshold,
                embedding_model=feature_aspect_model,
            )
        else:
            initial_features, silver_label = label_from_evidence(
                query,
                retriever,
                initial_k=args.initial_k,
                expanded_k=args.expanded_k,
                overlap_threshold=args.weak_support_overlap_threshold,
                embedding_model=feature_aspect_model,
            )
        if silver_label is None:
            excluded_no_support += 1
        else:
            examples.append((initial_features, silver_label))
        _maybe_log_progress("calibration_labels", idx, total_queries, stage_started)

    best = None
    best_acc = -1.0
    best_balanced_acc = -1.0
    strict_best_acc = -1.0
    strict_best_balanced_acc = -1.0
    wr_grid, wc_grid, ws_grid, wu_grid = get_weight_grid(args.ablate_signal)
    threshold_grid = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8]

    for wr, wc, ws, wu, threshold in product(wr_grid, wc_grid, ws_grid, wu_grid, threshold_grid):
        if not is_monotonic_configuration(wr, wc, ws, wu):
            continue
        estimator = SufficiencyEstimator(
            relevance_weight=wr,
            coverage_weight=wc,
            supportiveness_weight=ws,
            redundancy_weight=wu,
            threshold=threshold,
        )
        correct = 0
        predicted_expansions = 0
        sufficient_total = 0
        insufficient_total = 0
        sufficient_correct = 0
        insufficient_correct = 0
        for features, label in examples:
            prediction = 1 if estimator.predict(features).sufficient else 0
            if prediction == label:
                correct += 1
            if label == 1:
                sufficient_total += 1
                if prediction == label:
                    sufficient_correct += 1
            else:
                insufficient_total += 1
                if prediction == label:
                    insufficient_correct += 1
            if prediction == 0:
                predicted_expansions += 1
        acc = correct / max(len(examples), 1)
        sufficient_recall = sufficient_correct / max(sufficient_total, 1)
        insufficient_recall = insufficient_correct / max(insufficient_total, 1)
        balanced_acc = 0.5 * (sufficient_recall + insufficient_recall)
        expansion_rate = predicted_expansions / max(len(examples), 1)
        if (
            balanced_acc > strict_best_balanced_acc
            or (balanced_acc == strict_best_balanced_acc and acc > strict_best_acc)
        ):
            strict_best_acc = acc
            strict_best_balanced_acc = balanced_acc
        best_expansion_rate = best.get("predicted_expansion_rate", float("inf")) if best else float("inf")
        better_balanced = balanced_acc > best_balanced_acc
        better_accuracy_tie = balanced_acc == best_balanced_acc and acc > best_acc
        lower_expansion_near_tie = (
            best is not None
            and args.expansion_selection_tolerance > 0
            and balanced_acc >= strict_best_balanced_acc - args.expansion_selection_tolerance
            and acc >= strict_best_acc - args.expansion_selection_tolerance
            and expansion_rate < best_expansion_rate
        )
        exact_lower_expansion_tie = (
            balanced_acc == best_balanced_acc
            and acc == best_acc
            and expansion_rate < best_expansion_rate
        )
        if better_balanced or better_accuracy_tie or exact_lower_expansion_tie or lower_expansion_near_tie:
            best_acc = acc
            best_balanced_acc = balanced_acc
            best = {
                "relevance_weight": wr,
                "coverage_weight": wc,
                "supportiveness_weight": ws,
                "redundancy_weight": wu,
                "threshold": threshold,
                "silver_accuracy": acc,
                "silver_balanced_accuracy": balanced_acc,
                "silver_sufficient_recall": sufficient_recall,
                "silver_insufficient_recall": insufficient_recall,
                "silver_sufficient_total": sufficient_total,
                "silver_insufficient_total": insufficient_total,
                "predicted_expansion_rate": expansion_rate,
                "expansion_selection_tolerance": args.expansion_selection_tolerance,
                "num_examples": len(examples),
                "excluded_no_support": excluded_no_support,
                "weak_support_overlap_threshold": args.weak_support_overlap_threshold,
                "ablate_signal": args.ablate_signal,
                "label_strategy": args.label_strategy,
                "initial_k": args.initial_k,
                "expanded_k": args.expanded_k,
                "embedding_model": args.embedding_model,
                "manifest_path": args.manifest_path or None,
                "manifest_id": args.manifest_id,
                "corpus_split": args.corpus_split,
                "query_split": args.query_split,
                "doc_start": args.doc_start,
                "doc_limit": args.doc_limit,
                "query_start": args.query_start,
                "query_limit": args.query_limit,
                "seed": args.seed,
                "query_ids": [query.query_id for query in queries if query.query_id],
                "generator_prompt_version": GENERATOR_PROMPT_VERSION,
                "retrieval_cache_dir": args.retrieval_cache_dir,
                "nq_max_tokens": args.nq_max_tokens,
                "nq_stride": args.nq_stride,
            }

    # Confidence baseline calibration from the same silver labels.
    confidence_best = None
    confidence_best_acc = -1.0
    confidence_best_balanced_acc = -1.0
    confidence_threshold_grid = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    confidence_examples = []
    stage_started = time.time()
    for idx, query in enumerate(queries, start=1):
        initial_docs = retriever.retrieve(query, args.initial_k)
        expanded_docs = retriever.retrieve(query, args.expanded_k)
        initial_correct = evidence_has_weak_support(query, initial_docs, overlap_threshold=args.weak_support_overlap_threshold)
        expanded_correct = evidence_has_weak_support(query, expanded_docs, overlap_threshold=args.weak_support_overlap_threshold)
        label = build_silver_label(initial_correct, expanded_correct)
        if label is None:
            continue
        raw_scores = [doc.retrieval_score for doc in initial_docs]
        norm_scores = min_max_normalize(raw_scores) if raw_scores else []
        avg_score = sum(norm_scores) / max(len(norm_scores), 1)
        top1_score = norm_scores[0] if norm_scores else 0.0
        top2_score = norm_scores[1] if len(norm_scores) > 1 else 0.0
        score_gap = top1_score - top2_score
        confidence_score = 0.5 * top1_score + 0.4 * avg_score + 0.1 * score_gap
        confidence_examples.append((confidence_score, label))
        _maybe_log_progress("confidence_labels", idx, total_queries, stage_started)

    for threshold in confidence_threshold_grid:
        correct = 0
        sufficient_total = 0
        insufficient_total = 0
        sufficient_correct = 0
        insufficient_correct = 0
        for score, label in confidence_examples:
            prediction = 1 if score >= threshold else 0
            if prediction == label:
                correct += 1
            if label == 1:
                sufficient_total += 1
                if prediction == label:
                    sufficient_correct += 1
            else:
                insufficient_total += 1
                if prediction == label:
                    insufficient_correct += 1
        acc = correct / max(len(confidence_examples), 1)
        sufficient_recall = sufficient_correct / max(sufficient_total, 1)
        insufficient_recall = insufficient_correct / max(insufficient_total, 1)
        balanced_acc = 0.5 * (sufficient_recall + insufficient_recall)
        if balanced_acc > confidence_best_balanced_acc or (
            balanced_acc == confidence_best_balanced_acc and acc > confidence_best_acc
        ):
            confidence_best_acc = acc
            confidence_best_balanced_acc = balanced_acc
            confidence_best = {
                "threshold": threshold,
                "silver_accuracy": acc,
                "silver_balanced_accuracy": balanced_acc,
                "silver_sufficient_recall": sufficient_recall,
                "silver_insufficient_recall": insufficient_recall,
                "silver_sufficient_total": sufficient_total,
                "silver_insufficient_total": insufficient_total,
                "num_examples": len(confidence_examples),
                "label_strategy": args.label_strategy,
                "manifest_path": args.manifest_path or None,
                "manifest_id": args.manifest_id,
                "initial_k": args.initial_k,
                "expanded_k": args.expanded_k,
                "retrieval_cache_dir": args.retrieval_cache_dir,
            }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(best, indent=2), encoding="utf-8")
    confidence_output = Path(args.confidence_calibration_out) if args.confidence_calibration_out else output_path.with_name(output_path.stem.replace("calib_", "confidence_calib_") + output_path.suffix)
    confidence_output.parent.mkdir(parents=True, exist_ok=True)
    confidence_output.write_text(json.dumps(confidence_best, indent=2), encoding="utf-8")
    write_run_config(
        output_path.with_suffix(".meta.json"),
        {
            "script": "calibrate.py",
            "args": vars(args),
            "num_queries": len(queries),
            "num_examples": len(examples),
            "excluded_no_support": excluded_no_support,
            "generator_type": "openai" if args.use_openai else "simple_placeholder",
            "model_version": args.openai_model if args.use_openai else "simple_placeholder",
            "openai_cache_stats": generator.get_cache_stats() if generator else None,
            "manifest_path": args.manifest_path or None,
            "manifest_id": args.manifest_id,
            "label_strategy": args.label_strategy,
            "excluded_no_support": excluded_no_support,
            "generator_prompt_version": GENERATOR_PROMPT_VERSION,
            "confidence_calibration_output": str(confidence_output),
            "output": str(output_path),
        },
    )
    print(f"Saved calibration to {output_path}")
    print(json.dumps(best, indent=2))
    print(f"Saved confidence calibration to {confidence_output}")
    print(
        f"[DONE] calibration_labels={len(examples)}/{total_queries} "
        f"excluded_no_support={excluded_no_support}"
    )


if __name__ == "__main__":
    main()
