import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

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
        queries = [Query("When is the birthday of Michael Phelps?", answer="June 30, 1985")]
        simple_retriever = SimpleRetriever(corpus)
        faiss_retriever = simple_retriever
    elif args.mode == "hotpotqa":
        raw_docs = load_hotpotqa_sample(start=args.doc_start, limit=args.doc_limit)
        corpus = embed_corpus_texts(raw_docs, model_name=args.embedding_model)
        queries = load_hotpotqa_queries(start=args.query_start, limit=args.query_limit)
        simple_retriever = FaissRetriever(corpus, model_name=args.embedding_model)
        faiss_retriever = simple_retriever
    else:
        raw_docs = load_nq_sample(start=args.doc_start, limit=args.doc_limit)
        corpus = embed_corpus_texts(raw_docs, model_name=args.embedding_model)
        queries = load_nq_queries(start=args.query_start, limit=args.query_limit)
        simple_retriever = FaissRetriever(corpus, model_name=args.embedding_model)
        faiss_retriever = simple_retriever

    generator = SimpleGenerator()
    if args.use_openai:
        generator = OpenAIGenerator(model=args.openai_model)

    return corpus, queries, simple_retriever, faiss_retriever, generator


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


def add_metrics(row: dict[str, Any], query: Query) -> dict[str, Any]:
    row = row.copy()
    row["gold_answer"] = query.answer
    if query.answer:
        row["exact_match"] = exact_match_score(row["answer"], query.answer)
        row["f1"] = f1_score(row["answer"], query.answer)
    else:
        row["exact_match"] = None
        row["f1"] = None
    return row


def load_estimator(args: argparse.Namespace) -> tuple[SufficiencyEstimator, str]:
    estimator = SufficiencyEstimator()
    if not args.calibration_file:
        return estimator, args.structure_aware_label or "structure_aware_adaptive_rag"

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
        return estimator, args.structure_aware_label

    ablate_signal = config.get("ablate_signal", "none")
    if ablate_signal != "none":
        return estimator, f"structure_aware_wo_{ablate_signal}"
    return estimator, "structure_aware_adaptive_rag"


def run_vanilla(query: Query, retriever, generator, top_k: int = 3) -> dict[str, Any]:
    docs = retriever.retrieve(query, top_k=top_k)
    answer = generator.generate(query, docs)
    return {
        "baseline": "vanilla_rag",
        "query": query.text,
        "decision": "fixed_retrieve",
        "reason": "fixed_policy",
        "used_docs": [doc.doc_id for doc in docs],
        "retrieval_calls": 1,
        "doc_count": len(docs),
        "sufficiency_score": None,
        "relevance": None,
        "redundancy": None,
        "coverage": None,
        "supportiveness": None,
        "answer": answer,
    }


def run_fixed_large_k(query: Query, retriever, generator, top_k: int = 5) -> dict[str, Any]:
    docs = retriever.retrieve(query, top_k=top_k)
    answer = generator.generate(query, docs)
    return {
        "baseline": "fixed_large_k_rag",
        "query": query.text,
        "decision": "fixed_retrieve_more",
        "reason": "fixed_policy",
        "used_docs": [doc.doc_id for doc in docs],
        "retrieval_calls": 1,
        "doc_count": len(docs),
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
            "retrieval_calls": 1,
            "doc_count": len(initial_docs),
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
        "retrieval_calls": 2,
        "doc_count": len(expanded_docs),
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
        "retrieval_calls": retrieval_calls,
        "doc_count": len(result["used_docs"]),
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
                "baseline",
                "query",
                "decision",
                "reason",
                "used_docs",
                "retrieval_calls",
                "doc_count",
                "sufficiency_score",
                "relevance",
                "redundancy",
                "coverage",
                "supportiveness",
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
            writer.writerow(row)


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["demo", "hotpotqa", "nq"], default="demo")
    parser.add_argument("--use-openai", action="store_true")
    parser.add_argument("--openai-model", default="gpt-4.1-mini")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--doc-start", type=int, default=0)
    parser.add_argument("--doc-limit", type=int, default=100)
    parser.add_argument("--query-start", type=int, default=0)
    parser.add_argument("--query-limit", type=int, default=100)
    parser.add_argument("--initial-k", type=int, default=3)
    parser.add_argument("--expanded-k", type=int, default=5)
    parser.add_argument("--confidence-threshold", type=float, default=0.88)
    parser.add_argument("--calibration-file", default="")
    parser.add_argument(
        "--baselines",
        default="vanilla_rag,fixed_large_k_rag,confidence_adaptive_rag,structure_aware_adaptive_rag",
    )
    parser.add_argument("--structure-aware-label", default="")
    parser.add_argument("--output", default="results/baseline_results.csv")
    args = parser.parse_args()

    _, queries, simple_retriever, _, generator = build_resources(args)
    estimator, structure_aware_name = load_estimator(args)
    selected_baselines = set(parse_baselines(args.baselines))

    rows: list[dict[str, Any]] = []
    for query in queries:
        if "vanilla_rag" in selected_baselines:
            rows.append(add_metrics(run_vanilla(query, simple_retriever, generator, top_k=args.initial_k), query))
        if "fixed_large_k_rag" in selected_baselines:
            rows.append(add_metrics(run_fixed_large_k(query, simple_retriever, generator, top_k=args.expanded_k), query))
        if "confidence_adaptive_rag" in selected_baselines:
            rows.append(
                add_metrics(
                    run_confidence_baseline(
                        query,
                        simple_retriever,
                        generator,
                        initial_k=args.initial_k,
                        expanded_k=args.expanded_k,
                        threshold=args.confidence_threshold,
                    ),
                    query,
                )
            )
        if "structure_aware_adaptive_rag" in selected_baselines:
            rows.append(
                add_metrics(
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
                )
            )

    output_path = Path(args.output)
    write_results(rows, output_path)

    print(f"Saved results to {output_path}")
    for row in rows[:4]:
        print(row["baseline"], row["decision"], row["doc_count"], row["retrieval_calls"])


if __name__ == "__main__":
    main()
