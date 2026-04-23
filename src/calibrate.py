import argparse
import json
import re
from itertools import product
from pathlib import Path

from dotenv import load_dotenv

from experiment_utils import set_global_seed, write_run_config
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
    load_hotpotqa_queries,
    load_hotpotqa_sample,
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
):
    if mode == "demo":
        corpus = build_demo_corpus()
        queries = [Query("When is the birthday of Michael Phelps?", answer="June 30, 1985")]
        retriever = SimpleRetriever(corpus)
        return queries, retriever

    if mode == "hotpotqa":
        raw_docs = load_hotpotqa_sample(start=doc_start, limit=doc_limit, split=corpus_split)
        corpus = embed_corpus_texts(raw_docs, model_name=embedding_model)
        queries = load_hotpotqa_queries(start=query_start, limit=query_limit, split=query_split)
        retriever = FaissRetriever(corpus, model_name=embedding_model)
        return queries, retriever

    raw_docs = load_nq_sample(start=doc_start, limit=doc_limit, split=corpus_split)
    corpus = embed_corpus_texts(raw_docs, model_name=embedding_model)
    queries = load_nq_queries(start=query_start, limit=query_limit, split=query_split)
    retriever = FaissRetriever(corpus, model_name=embedding_model)
    return queries, retriever


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def evidence_has_weak_support(query: Query, docs, overlap_threshold: float) -> bool:
    if not query.answer:
        return False
    gold = normalize_answer(query.answer)
    if not gold:
        return False
    for doc in docs:
        if gold in normalize_answer(doc.text) and compute_query_overlap(query, doc) >= overlap_threshold:
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
    default_grid = [0.1, 0.2, 0.35, 0.5]
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

    if query.answer:
        initial_answer = generator.generate(query, initial_docs)
        expanded_answer = generator.generate(query, expanded_docs)
        initial_em = 1 if normalize_answer(initial_answer) == normalize_answer(query.answer) else 0
        expanded_em = 1 if normalize_answer(expanded_answer) == normalize_answer(query.answer) else 0
        initial_good = initial_support and initial_em == 1
        expanded_good = expanded_support and expanded_em == 1
        return initial_features, build_silver_label(initial_good, expanded_good)

    return initial_features, build_silver_label(initial_support, expanded_support)


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["demo", "hotpotqa", "nq"], default="demo")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--doc-start", type=int, default=0)
    parser.add_argument("--doc-limit", type=int, default=100)
    parser.add_argument("--corpus-split", default="validation")
    parser.add_argument("--query-start", type=int, default=0)
    parser.add_argument("--query-limit", type=int, default=100)
    parser.add_argument("--query-split", default="validation")
    parser.add_argument("--initial-k", type=int, default=3)
    parser.add_argument("--expanded-k", type=int, default=5)
    parser.add_argument("--weak-support-overlap-threshold", type=float, default=0.2)
    parser.add_argument("--label-strategy", choices=["evidence", "hybrid_generation"], default="evidence")
    parser.add_argument("--use-openai", action="store_true")
    parser.add_argument("--openai-model", default="gpt-4.1-mini")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ablate-signal",
        choices=["none", "relevance", "redundancy", "coverage", "supportiveness"],
        default="none",
    )
    parser.add_argument("--output", default="results/calibration.json")
    args = parser.parse_args()
    set_global_seed(args.seed)

    queries, retriever = build_resources(
        args.mode,
        args.doc_start,
        args.doc_limit,
        args.corpus_split,
        args.query_start,
        args.query_limit,
        args.query_split,
        args.embedding_model,
    )
    if args.label_strategy == "hybrid_generation" and not args.use_openai:
        raise ValueError("hybrid_generation label strategy requires --use-openai.")

    examples = []
    generator = OpenAIGenerator(model=args.openai_model) if args.use_openai else None
    for query in queries:
        if args.label_strategy == "hybrid_generation":
            initial_features, silver_label = label_from_hybrid_generation(
                query,
                retriever,
                generator,
                initial_k=args.initial_k,
                expanded_k=args.expanded_k,
                overlap_threshold=args.weak_support_overlap_threshold,
                embedding_model=args.embedding_model,
            )
        else:
            initial_features, silver_label = label_from_evidence(
                query,
                retriever,
                initial_k=args.initial_k,
                expanded_k=args.expanded_k,
                overlap_threshold=args.weak_support_overlap_threshold,
                embedding_model=args.embedding_model,
            )
        examples.append((initial_features, silver_label))

    best = None
    best_acc = -1.0
    wr_grid, wc_grid, ws_grid, wu_grid = get_weight_grid(args.ablate_signal)
    threshold_grid = [0.3, 0.4, 0.5, 0.6, 0.7]

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
        for features, label in examples:
            prediction = 1 if estimator.predict(features).sufficient else 0
            if prediction == label:
                correct += 1
        acc = correct / max(len(examples), 1)
        if acc > best_acc:
            best_acc = acc
            best = {
                "relevance_weight": wr,
                "coverage_weight": wc,
                "supportiveness_weight": ws,
                "redundancy_weight": wu,
                "threshold": threshold,
                "silver_accuracy": acc,
                "num_examples": len(examples),
                "weak_support_overlap_threshold": args.weak_support_overlap_threshold,
                "ablate_signal": args.ablate_signal,
                "label_strategy": args.label_strategy,
                "corpus_split": args.corpus_split,
                "query_split": args.query_split,
                "doc_start": args.doc_start,
                "doc_limit": args.doc_limit,
                "query_start": args.query_start,
                "query_limit": args.query_limit,
                "seed": args.seed,
            }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(best, indent=2), encoding="utf-8")
    write_run_config(
        output_path.with_suffix(".meta.json"),
        {
            "script": "calibrate.py",
            "args": vars(args),
            "num_queries": len(queries),
            "num_examples": len(examples),
            "output": str(output_path),
        },
    )
    print(f"Saved calibration to {output_path}")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
