import argparse
import csv
import re
from pathlib import Path
from typing import Any

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
)


def build_resources(args: argparse.Namespace):
    if args.mode == "demo":
        corpus = build_demo_corpus()
        queries = [Query("When is the birthday of Michael Phelps?", answer="June 30, 1985")]
        simple_retriever = SimpleRetriever(corpus)
        faiss_retriever = simple_retriever
    else:
        raw_docs = load_hotpotqa_sample(limit=args.doc_limit)
        corpus = embed_corpus_texts(raw_docs, model_name=args.embedding_model)
        queries = load_hotpotqa_queries(limit=args.query_limit)
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


def run_vanilla(query: Query, retriever, generator, top_k: int = 3) -> dict[str, Any]:
    docs = retriever.retrieve(query, top_k=top_k)
    answer = generator.generate(query, docs)
    return {
        "baseline": "vanilla_rag",
        "query": query.text,
        "decision": "fixed_retrieve",
        "used_docs": [doc.doc_id for doc in docs],
        "retrieval_calls": 1,
        "doc_count": len(docs),
        "sufficiency_score": None,
        "answer": answer,
    }


def run_fixed_large_k(query: Query, retriever, generator, top_k: int = 5) -> dict[str, Any]:
    docs = retriever.retrieve(query, top_k=top_k)
    answer = generator.generate(query, docs)
    return {
        "baseline": "fixed_large_k_rag",
        "query": query.text,
        "decision": "fixed_retrieve_more",
        "used_docs": [doc.doc_id for doc in docs],
        "retrieval_calls": 1,
        "doc_count": len(docs),
        "sufficiency_score": None,
        "answer": answer,
    }


def run_confidence_baseline(query: Query, retriever, generator, initial_k: int = 3, expanded_k: int = 5, threshold: float = 0.88) -> dict[str, Any]:
    initial_docs = retriever.retrieve(query, top_k=initial_k)
    avg_score = sum(doc.retrieval_score for doc in initial_docs) / max(len(initial_docs), 1)

    if avg_score >= threshold:
        answer = generator.generate(query, initial_docs)
        return {
            "baseline": "confidence_adaptive_rag",
            "query": query.text,
            "decision": "answer_now",
            "used_docs": [doc.doc_id for doc in initial_docs],
            "retrieval_calls": 1,
            "doc_count": len(initial_docs),
            "sufficiency_score": avg_score,
            "answer": answer,
        }

    expanded_docs = retriever.retrieve(query, top_k=expanded_k)
    answer = generator.generate(query, expanded_docs)
    return {
        "baseline": "confidence_adaptive_rag",
        "query": query.text,
        "decision": "retrieve_more",
        "used_docs": [doc.doc_id for doc in expanded_docs],
        "retrieval_calls": 2,
        "doc_count": len(expanded_docs),
        "sufficiency_score": avg_score,
        "answer": answer,
    }


def run_structure_aware(query: Query, retriever, generator) -> dict[str, Any]:
    pipeline = StructureAwareAdaptiveRAG(
        retriever=retriever,
        generator=generator,
        estimator=SufficiencyEstimator(),
        initial_k=3,
        expanded_k=5,
    )
    result = pipeline.answer(query)
    retrieval_calls = 1 if result["decision"] == "answer_now" else 2
    return {
        "baseline": "structure_aware_adaptive_rag",
        "query": query.text,
        "decision": result["decision"],
        "used_docs": result["used_docs"],
        "retrieval_calls": retrieval_calls,
        "doc_count": len(result["used_docs"]),
        "sufficiency_score": result["sufficiency_score"],
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
                "used_docs",
                "retrieval_calls",
                "doc_count",
                "sufficiency_score",
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["demo", "hotpotqa"], default="demo")
    parser.add_argument("--use-openai", action="store_true")
    parser.add_argument("--openai-model", default="gpt-4.1-mini")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--doc-limit", type=int, default=50)
    parser.add_argument("--query-limit", type=int, default=5)
    parser.add_argument("--output", default="results/baseline_results.csv")
    args = parser.parse_args()

    _, queries, simple_retriever, _, generator = build_resources(args)

    rows: list[dict[str, Any]] = []
    for query in queries:
        rows.append(add_metrics(run_vanilla(query, simple_retriever, generator), query))
        rows.append(add_metrics(run_fixed_large_k(query, simple_retriever, generator), query))
        rows.append(add_metrics(run_confidence_baseline(query, simple_retriever, generator), query))
        rows.append(add_metrics(run_structure_aware(query, simple_retriever, generator), query))

    output_path = Path(args.output)
    write_results(rows, output_path)

    print(f"Saved results to {output_path}")
    for row in rows[:4]:
        print(row["baseline"], row["decision"], row["doc_count"], row["retrieval_calls"])


if __name__ == "__main__":
    main()
