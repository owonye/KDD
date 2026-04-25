import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from experiment_utils import load_manifest, set_global_seed, write_run_config
from rag.pipeline import (
    FaissRetriever,
    OpenAIGenerator,
    Query,
    SimpleGenerator,
    SimpleRetriever,
    StructureAwareAdaptiveRAG,
    SufficiencyEstimator,
    build_demo_corpus,
    embed_corpus_texts,
    load_hotpotqa_queries,
    load_hotpotqa_sample,
    min_max_normalize,
    load_nq_queries,
    load_nq_sample,
)


VALID_BASELINES = {
    "vanilla_rag",
    "fixed_large_k_rag",
    "confidence_adaptive_rag",
    "structure_aware_adaptive_rag",
}


def parse_baselines(raw: str) -> list[str]:
    baselines = [token.strip() for token in raw.split(",") if token.strip()]
    if not baselines:
        raise ValueError("At least one baseline must be provided.")
    unknown = [name for name in baselines if name not in VALID_BASELINES]
    if unknown:
        raise ValueError(f"Unknown baseline(s): {unknown}")
    return baselines


def build_resources(args: argparse.Namespace):
    if args.mode == "demo":
        corpus = build_demo_corpus()
        queries = [Query("When is the birthday of Michael Phelps?", answer="June 30, 1985", answers=["June 30, 1985"])]
        simple_retriever = SimpleRetriever(corpus)
        faiss_retriever = simple_retriever
    elif args.mode == "hotpotqa":
        raw_docs = load_hotpotqa_sample(start=args.doc_start, limit=args.doc_limit, split=args.corpus_split)
        corpus = embed_corpus_texts(raw_docs, model_name=args.embedding_model)
        queries = load_hotpotqa_queries(start=args.query_start, limit=args.query_limit, split=args.query_split)
        simple_retriever = FaissRetriever(corpus, model_name=args.embedding_model)
        faiss_retriever = simple_retriever
    else:
        raw_docs = load_nq_sample(start=args.doc_start, limit=args.doc_limit, split=args.corpus_split)
        corpus = embed_corpus_texts(raw_docs, model_name=args.embedding_model)
        queries = load_nq_queries(start=args.query_start, limit=args.query_limit, split=args.query_split)
        simple_retriever = FaissRetriever(corpus, model_name=args.embedding_model)
        faiss_retriever = simple_retriever

    if args.mode != "demo" and not args.use_openai and not args.allow_simple_generator:
        raise ValueError(
            "Non-demo evaluation requires real QA generation. Use --use-openai or pass --allow-simple-generator explicitly."
        )

    generator = (
        OpenAIGenerator(model=args.openai_model, cache_path=args.openai_cache_path)
        if args.use_openai
        else SimpleGenerator()
    )

    return corpus, queries, simple_retriever, faiss_retriever, generator


def resolve_manifest_overrides(args: argparse.Namespace) -> argparse.Namespace:
    manifest = load_manifest(args.manifest_path)
    if not manifest:
        args.manifest_id = None
        return args
    args.manifest_id = manifest.get("manifest_id")
    args.doc_start = int(manifest["doc_start"])
    args.doc_limit = int(manifest["doc_limit"])
    args.corpus_split = str(manifest["corpus_split"])
    args.query_start = int(manifest["eval_query_start"])
    args.query_limit = int(manifest["eval_query_limit"])
    args.query_split = str(manifest["query_split"])
    args.initial_k = int(manifest["initial_k"])
    args.expanded_k = int(manifest["expanded_k"])
    args.embedding_model = str(manifest["embedding_model"])
    args.seed = int(manifest["seed"])
    return args


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def exact_match_score(prediction: str, gold: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


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


def add_metrics(row: dict[str, Any], query: Query, generator_type: str) -> dict[str, Any]:
    row = row.copy()
    gold_answers = query.answers if query.answers else ([query.answer] if query.answer else [])
    row["gold_answer"] = " || ".join(gold_answers) if gold_answers else None
    if generator_type == "simple_placeholder":
        row["exact_match"] = None
        row["f1"] = None
    elif gold_answers:
        row["exact_match"] = max(exact_match_score(row["answer"], gold) for gold in gold_answers)
        row["f1"] = max(f1_score(row["answer"], gold) for gold in gold_answers)
    else:
        row["exact_match"] = None
        row["f1"] = None
    return row


def load_estimator(args: argparse.Namespace) -> tuple[SufficiencyEstimator, str, dict[str, Any]]:
    estimator = SufficiencyEstimator()
    if not args.calibration_file:
        return estimator, args.structure_aware_label or "structure_aware_adaptive_rag", {}

    calibration_path = Path(args.calibration_file).resolve()
    if not calibration_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calibration_path}")

    config = json.loads(calibration_path.read_text(encoding="utf-8"))
    estimator.update_parameters(
        relevance_weight=config["relevance_weight"],
        coverage_weight=config["coverage_weight"],
        supportiveness_weight=config["supportiveness_weight"],
        redundancy_weight=config["redundancy_weight"],
        threshold=config["threshold"],
    )
    if args.structure_aware_label:
        return estimator, args.structure_aware_label, config

    ablate_signal = config.get("ablate_signal", "none")
    if ablate_signal != "none":
        return estimator, f"structure_aware_wo_{ablate_signal}", config
    return estimator, "structure_aware_adaptive_rag", config


def load_confidence_threshold(args: argparse.Namespace) -> float:
    if not args.confidence_calibration_file:
        return args.confidence_threshold

    confidence_path = Path(args.confidence_calibration_file).resolve()
    if not confidence_path.exists():
        raise FileNotFoundError(f"Confidence calibration file not found: {confidence_path}")
    config = json.loads(confidence_path.read_text(encoding="utf-8"))
    threshold = config.get("threshold")
    if threshold is None:
        raise ValueError("Confidence calibration file missing 'threshold'.")
    return float(threshold)


def run_vanilla(query: Query, retriever, generator, top_k: int = 3) -> dict[str, Any]:
    docs = retriever.retrieve(query, top_k=top_k)
    answer = generator.generate(query, docs)
    return {
        "baseline": "vanilla_rag",
        "query": query.text,
        "decision": "fixed_retrieve",
        "reason": "fixed_policy",
        "used_docs": [doc.doc_id for doc in docs],
        "initial_doc_ids": [doc.doc_id for doc in docs],
        "final_doc_ids": [doc.doc_id for doc in docs],
        "expanded": False,
        "retrieval_calls": 1,
        "initial_doc_count": len(docs),
        "doc_count": len(docs),
        "final_k": len(docs),
        "sufficiency_score": None,
        "relevance": None,
        "redundancy": None,
        "coverage": None,
        "supportiveness": None,
        "answer": answer,
    }


def run_fixed_large_k(query: Query, retriever, generator, initial_k: int = 3, expanded_k: int = 5) -> dict[str, Any]:
    initial_docs = retriever.retrieve(query, top_k=initial_k)
    final_docs = retriever.retrieve(query, top_k=expanded_k)
    answer = generator.generate(query, final_docs)
    return {
        "baseline": "fixed_large_k_rag",
        "query": query.text,
        "decision": "fixed_retrieve_more",
        "reason": "fixed_policy",
        "used_docs": [doc.doc_id for doc in final_docs],
        "initial_doc_ids": [doc.doc_id for doc in initial_docs],
        "final_doc_ids": [doc.doc_id for doc in final_docs],
        "expanded": True,
        "retrieval_calls": 2,
        "initial_doc_count": len(initial_docs),
        "doc_count": len(final_docs),
        "final_k": len(final_docs),
        "sufficiency_score": None,
        "relevance": None,
        "redundancy": None,
        "coverage": None,
        "supportiveness": None,
        "answer": answer,
    }


def run_confidence_baseline(
    query: Query,
    retriever,
    generator,
    initial_k: int = 3,
    expanded_k: int = 5,
    threshold: float = 0.88,
) -> dict[str, Any]:
    initial_docs = retriever.retrieve(query, top_k=initial_k)
    raw_scores = [doc.retrieval_score for doc in initial_docs]
    norm_scores = min_max_normalize(raw_scores) if raw_scores else []
    avg_score = sum(norm_scores) / max(len(norm_scores), 1)
    top1_score = norm_scores[0] if norm_scores else 0.0
    top2_score = norm_scores[1] if len(norm_scores) > 1 else 0.0
    score_gap = top1_score - top2_score

    confidence_score = 0.5 * top1_score + 0.4 * avg_score + 0.1 * score_gap

    if confidence_score >= threshold:
        answer = generator.generate(query, initial_docs)
        return {
            "baseline": "confidence_adaptive_rag",
            "query": query.text,
            "decision": "answer_now",
            "reason": "high_confidence",
            "used_docs": [doc.doc_id for doc in initial_docs],
            "initial_doc_ids": [doc.doc_id for doc in initial_docs],
            "final_doc_ids": [doc.doc_id for doc in initial_docs],
            "expanded": False,
            "retrieval_calls": 1,
            "initial_doc_count": len(initial_docs),
            "doc_count": len(initial_docs),
            "final_k": len(initial_docs),
            "sufficiency_score": confidence_score,
            "relevance": avg_score,
            "redundancy": None,
            "coverage": None,
            "supportiveness": None,
            "answer": answer,
        }

    expanded_docs = retriever.retrieve(query, top_k=expanded_k)
    answer = generator.generate(query, expanded_docs)
    return {
        "baseline": "confidence_adaptive_rag",
        "query": query.text,
        "decision": "retrieve_more",
        "reason": "low_confidence",
        "used_docs": [doc.doc_id for doc in expanded_docs],
        "initial_doc_ids": [doc.doc_id for doc in initial_docs],
        "final_doc_ids": [doc.doc_id for doc in expanded_docs],
        "expanded": True,
        "retrieval_calls": 2,
        "initial_doc_count": len(initial_docs),
        "doc_count": len(expanded_docs),
        "final_k": len(expanded_docs),
        "sufficiency_score": confidence_score,
        "relevance": avg_score,
        "redundancy": None,
        "coverage": None,
        "supportiveness": None,
        "answer": answer,
    }


def run_structure_aware(
    query: Query,
    retriever,
    generator,
    estimator,
    initial_k: int = 3,
    expanded_k: int = 5,
    aspect_model: str = "BAAI/bge-small-en-v1.5",
    baseline_name: str = "structure_aware_adaptive_rag",
) -> dict[str, Any]:
    pipeline = StructureAwareAdaptiveRAG(
        retriever=retriever,
        generator=generator,
        estimator=estimator,
        initial_k=initial_k,
        expanded_k=expanded_k,
        aspect_model=aspect_model,
    )
    result = pipeline.answer(query)
    retrieval_calls = 1 if result["decision"] == "answer_now" else 2
    return {
        "baseline": baseline_name,
        "query": query.text,
        "decision": result["decision"],
        "reason": result["reason"],
        "used_docs": result["used_docs"],
        "initial_doc_ids": result["initial_doc_ids"],
        "final_doc_ids": result["final_doc_ids"],
        "expanded": result["expansion_triggered"],
        "retrieval_calls": retrieval_calls,
        "initial_doc_count": len(result["initial_doc_ids"]),
        "doc_count": len(result["used_docs"]),
        "final_k": result["final_doc_count"],
        "sufficiency_score": result["sufficiency_score"],
        "relevance": result["features"].relevance,
        "redundancy": result["features"].redundancy,
        "coverage": result["features"].coverage,
        "supportiveness": result["features"].supportiveness,
        "answer": result["answer"],
    }


def write_results(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "query_id",
                "query_uid",
                "generator_type",
                "model_version",
                "baseline",
                "query",
                "decision",
                "reason",
                "expanded",
                "used_docs",
                "initial_doc_ids",
                "final_doc_ids",
                "retrieval_calls",
                "initial_doc_count",
                "doc_count",
                "final_k",
                "sufficiency_score",
                "relevance",
                "redundancy",
                "coverage",
                "supportiveness",
                "label_strategy",
                "calibration_source",
                "gold_answer",
                "exact_match",
                "f1",
                "answer",
            ],
        )
        writer.writeheader()
        for row in rows:
            row = row.copy()
            row["used_docs"] = "|".join(row["used_docs"])
            row["initial_doc_ids"] = "|".join(row["initial_doc_ids"])
            row["final_doc_ids"] = "|".join(row["final_doc_ids"])
            writer.writerow(row)


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["demo", "hotpotqa", "nq"], default="demo")
    parser.add_argument("--use-openai", action="store_true")
    parser.add_argument("--allow-simple-generator", action="store_true")
    parser.add_argument("--openai-model", default="gpt-4.1-mini")
    parser.add_argument("--openai-cache-path", default="results/openai_cache.jsonl")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--doc-start", type=int, default=0)
    parser.add_argument("--doc-limit", type=int, default=20000)
    parser.add_argument("--corpus-split", default="train")
    parser.add_argument("--query-start", type=int, default=0)
    parser.add_argument("--query-limit", type=int, default=100)
    parser.add_argument("--query-split", default="validation")
    parser.add_argument("--initial-k", type=int, default=3)
    parser.add_argument("--expanded-k", type=int, default=5)
    parser.add_argument("--confidence-threshold", type=float, default=0.88)
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--confidence-calibration-file", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-file", default="")
    parser.add_argument(
        "--baselines",
        default="vanilla_rag,fixed_large_k_rag,confidence_adaptive_rag,structure_aware_adaptive_rag",
    )
    parser.add_argument("--structure-aware-label", default="")
    parser.add_argument("--output", default="results/baseline_results.csv")
    args = parser.parse_args()
    args = resolve_manifest_overrides(args)
    if not args.use_openai and args.mode != "demo" and not args.allow_simple_generator:
        raise ValueError("For paper-grade EM/F1, run evaluate.py with --use-openai.")
    set_global_seed(args.seed)

    _, queries, simple_retriever, _, generator = build_resources(args)
    generator_type = "openai" if args.use_openai else "simple_placeholder"
    model_version = args.openai_model if args.use_openai else "simple_placeholder"
    estimator, structure_aware_name, calibration_config = load_estimator(args)
    confidence_threshold = load_confidence_threshold(args)
    selected_baselines = set(parse_baselines(args.baselines))
    calibration_source = Path(args.calibration_file).name if args.calibration_file else "default"
    label_strategy = calibration_config.get("label_strategy", "default")

    rows: list[dict[str, Any]] = []
    for query_id, query in enumerate(queries):
        if "vanilla_rag" in selected_baselines:
            row = add_metrics(
                run_vanilla(query, simple_retriever, generator, top_k=args.initial_k),
                query,
                generator_type=generator_type,
            )
            row["query_id"] = query_id
            row["query_uid"] = query.query_id or f"{args.mode}::{args.query_split}::{args.query_start + query_id}"
            row["generator_type"] = generator_type
            row["model_version"] = model_version
            row["label_strategy"] = label_strategy
            row["calibration_source"] = calibration_source
            rows.append(row)
        if "fixed_large_k_rag" in selected_baselines:
            row = add_metrics(
                run_fixed_large_k(
                    query,
                    simple_retriever,
                    generator,
                    initial_k=args.initial_k,
                    expanded_k=args.expanded_k,
                ),
                query,
                generator_type=generator_type,
            )
            row["query_id"] = query_id
            row["query_uid"] = query.query_id or f"{args.mode}::{args.query_split}::{args.query_start + query_id}"
            row["generator_type"] = generator_type
            row["model_version"] = model_version
            row["label_strategy"] = label_strategy
            row["calibration_source"] = calibration_source
            rows.append(row)
        if "confidence_adaptive_rag" in selected_baselines:
            row = add_metrics(
                run_confidence_baseline(
                    query,
                    simple_retriever,
                    generator,
                    initial_k=args.initial_k,
                    expanded_k=args.expanded_k,
                    threshold=confidence_threshold,
                ),
                query,
                generator_type=generator_type,
            )
            row["query_id"] = query_id
            row["query_uid"] = query.query_id or f"{args.mode}::{args.query_split}::{args.query_start + query_id}"
            row["generator_type"] = generator_type
            row["model_version"] = model_version
            row["label_strategy"] = label_strategy
            row["calibration_source"] = (
                Path(args.confidence_calibration_file).name if args.confidence_calibration_file else "default"
            )
            rows.append(row)
        if "structure_aware_adaptive_rag" in selected_baselines:
            row = add_metrics(
                run_structure_aware(
                    query,
                    simple_retriever,
                    generator,
                    estimator,
                    initial_k=args.initial_k,
                    expanded_k=args.expanded_k,
                    aspect_model=args.embedding_model,
                    baseline_name=structure_aware_name,
                ),
                query,
                generator_type=generator_type,
            )
            row["query_id"] = query_id
            row["query_uid"] = query.query_id or f"{args.mode}::{args.query_split}::{args.query_start + query_id}"
            row["generator_type"] = generator_type
            row["model_version"] = model_version
            row["label_strategy"] = label_strategy
            row["calibration_source"] = calibration_source
            rows.append(row)

    output_path = Path(args.output)
    write_results(rows, output_path)
    write_run_config(
        output_path.with_suffix(".meta.json"),
        {
            "script": "evaluate.py",
            "args": vars(args),
            "num_queries": len(queries),
            "num_rows": len(rows),
            "generator_type": generator_type,
            "model_version": model_version,
            "openai_cache_stats": generator.get_cache_stats() if args.use_openai else None,
            "manifest_path": args.manifest_path or None,
            "manifest_id": args.manifest_id,
            "prompt_template_version": "evidence_only_v1",
            "calibration_source": calibration_source,
            "confidence_calibration_source": (
                Path(args.confidence_calibration_file).name if args.confidence_calibration_file else "default"
            ),
            "output": str(output_path),
        },
    )

    print(f"Saved results to {output_path}")
    for row in rows[:4]:
        print(row["baseline"], row["decision"], row["doc_count"], row["retrieval_calls"])


if __name__ == "__main__":
    main()
