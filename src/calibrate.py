import argparse
import json
import re
from itertools import product
from pathlib import Path

from dotenv import load_dotenv

from evaluate import exact_match_score
from rag.pipeline import (
    FaissRetriever,
    Query,
    SimpleGenerator,
    SimpleRetriever,
    StructureAwareAdaptiveRAG,
    SufficiencyEstimator,
    build_demo_corpus,
    build_silver_label,
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
    query_start: int,
    query_limit: int,
    embedding_model: str,
):
    if mode == "demo":
        corpus = build_demo_corpus()
        queries = [Query("When is the birthday of Michael Phelps?", answer="June 30, 1985")]
        retriever = SimpleRetriever(corpus)
        return queries, retriever

    if mode == "hotpotqa":
        raw_docs = load_hotpotqa_sample(start=doc_start, limit=doc_limit)
        corpus = embed_corpus_texts(raw_docs, model_name=embedding_model)
        queries = load_hotpotqa_queries(start=query_start, limit=query_limit)
        retriever = FaissRetriever(corpus, model_name=embedding_model)
        return queries, retriever

    raw_docs = load_nq_sample(start=doc_start, limit=doc_limit)
    corpus = embed_corpus_texts(raw_docs, model_name=embedding_model)
    queries = load_nq_queries(start=query_start, limit=query_limit)
    retriever = FaissRetriever(corpus, model_name=embedding_model)
    return queries, retriever


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def evidence_contains_answer(query: Query, docs) -> bool:
    if not query.answer:
        return False
    gold = normalize_answer(query.answer)
    if not gold:
        return False
    for doc in docs:
        if gold in normalize_answer(doc.text):
            return True
    return False


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["demo", "hotpotqa", "nq"], default="demo")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--doc-start", type=int, default=0)
    parser.add_argument("--doc-limit", type=int, default=50)
    parser.add_argument("--query-start", type=int, default=0)
    parser.add_argument("--query-limit", type=int, default=10)
    parser.add_argument("--initial-k", type=int, default=3)
    parser.add_argument("--expanded-k", type=int, default=5)
    parser.add_argument("--output", default="results/calibration.json")
    args = parser.parse_args()

    queries, retriever = build_resources(
        args.mode,
        args.doc_start,
        args.doc_limit,
        args.query_start,
        args.query_limit,
        args.embedding_model,
    )

    examples = []
    for query in queries:
        initial_docs = retriever.retrieve(query, args.initial_k)
        expanded_docs = retriever.retrieve(query, args.expanded_k)
        initial_features = extract_evidence_features(query, initial_docs, aspect_model=args.embedding_model)
        initial_correct = evidence_contains_answer(query, initial_docs)
        expanded_correct = evidence_contains_answer(query, expanded_docs)
        silver_label = build_silver_label(initial_correct, expanded_correct)
        examples.append((initial_features, silver_label))

    best = None
    best_acc = -1.0
    weight_grid = [0.1, 0.2, 0.35, 0.5]
    threshold_grid = [0.3, 0.4, 0.5, 0.6, 0.7]

    for wr, wc, ws, wu, threshold in product(weight_grid, weight_grid, weight_grid, weight_grid, threshold_grid):
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
            }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(f"Saved calibration to {output_path}")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
