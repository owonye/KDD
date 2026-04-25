import argparse
import json
from pathlib import Path

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
    load_nq_queries,
    load_nq_sample,
)


def load_estimator(args: argparse.Namespace) -> SufficiencyEstimator:
    estimator = SufficiencyEstimator()
    if not args.calibration_file:
        return estimator

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
    return estimator


def build_pipeline(args: argparse.Namespace) -> tuple[StructureAwareAdaptiveRAG, Query]:
    if args.mode == "demo":
        corpus = build_demo_corpus()
        retriever = SimpleRetriever(corpus)
        query = Query("When is the birthday of Michael Phelps?")
    elif args.mode == "hotpotqa":
        raw_docs = load_hotpotqa_sample(start=args.doc_start, limit=args.doc_limit, split=args.corpus_split)
        corpus = embed_corpus_texts(raw_docs, model_name=args.embedding_model)
        retriever = FaissRetriever(corpus, model_name=args.embedding_model)
        queries = load_hotpotqa_queries(start=args.query_start, limit=args.query_limit, split=args.query_split)
        query = queries[0]
    else:
        raw_docs = load_nq_sample(start=args.doc_start, limit=args.doc_limit, split=args.corpus_split)
        corpus = embed_corpus_texts(raw_docs, model_name=args.embedding_model)
        retriever = FaissRetriever(corpus, model_name=args.embedding_model)
        queries = load_nq_queries(start=args.query_start, limit=args.query_limit, split=args.query_split)
        query = queries[0]

    generator = SimpleGenerator()
    if args.use_openai:
        generator = OpenAIGenerator(model=args.openai_model)

    estimator = load_estimator(args)

    pipeline = StructureAwareAdaptiveRAG(
        retriever=retriever,
        generator=generator,
        estimator=estimator,
        initial_k=args.initial_k,
        expanded_k=args.expanded_k,
        aspect_model=args.embedding_model,
    )

    return pipeline, query


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["demo", "hotpotqa", "nq"], default="demo")
    parser.add_argument("--use-openai", action="store_true")
    parser.add_argument("--openai-model", default="gpt-4.1-mini")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--doc-start", type=int, default=0)
    parser.add_argument("--doc-limit", type=int, default=20000)
    parser.add_argument("--corpus-split", default="train")
    parser.add_argument("--query-start", type=int, default=0)
    parser.add_argument("--query-limit", type=int, default=100)
    parser.add_argument("--query-split", default="validation")
    parser.add_argument("--initial-k", type=int, default=3)
    parser.add_argument("--expanded-k", type=int, default=5)
    parser.add_argument("--calibration-file", default="")
    args = parser.parse_args()

    pipeline, query = build_pipeline(args)
    result = pipeline.answer(query)

    print("=== Structure-Aware Adaptive RAG Demo ===")
    print(f"Query: {result['query']}")
    print(f"Decision: {result['decision']}")
    print(f"Used docs: {result['used_docs']}")
    print(f"Sufficiency score: {result['sufficiency_score']:.3f}")
    print(f"Features: {result['features']}")
    print(f"Answer: {result['answer']}")


if __name__ == "__main__":
    main()
