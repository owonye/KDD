import argparse
import csv
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
        queries = [Query("When is the birthday of Michael Phelps?")]
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
        rows.append(run_vanilla(query, simple_retriever, generator))
        rows.append(run_fixed_large_k(query, simple_retriever, generator))
        rows.append(run_confidence_baseline(query, simple_retriever, generator))
        rows.append(run_structure_aware(query, simple_retriever, generator))

    output_path = Path(args.output)
    write_results(rows, output_path)

    print(f"Saved results to {output_path}")
    for row in rows[:4]:
        print(row["baseline"], row["decision"], row["doc_count"], row["retrieval_calls"])


if __name__ == "__main__":
    main()
