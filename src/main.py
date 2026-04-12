import argparse

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


def build_pipeline(args: argparse.Namespace) -> tuple[StructureAwareAdaptiveRAG, Query]:
    if args.mode == "demo":
        corpus = build_demo_corpus()
        retriever = SimpleRetriever(corpus)
        query = Query("When is the birthday of Michael Phelps?")
    elif args.mode == "hotpotqa":
        raw_docs = load_hotpotqa_sample(limit=args.doc_limit)
        corpus = embed_corpus_texts(raw_docs, model_name=args.embedding_model)
        retriever = FaissRetriever(corpus, model_name=args.embedding_model)
        queries = load_hotpotqa_queries(limit=args.query_limit)
        query = queries[0]
    else:
        raw_docs = load_nq_sample(limit=args.doc_limit)
        corpus = embed_corpus_texts(raw_docs, model_name=args.embedding_model)
        retriever = FaissRetriever(corpus, model_name=args.embedding_model)
        queries = load_nq_queries(limit=args.query_limit)
        query = queries[0]

    generator = SimpleGenerator()
    if args.use_openai:
        generator = OpenAIGenerator(model=args.openai_model)

    estimator = SufficiencyEstimator()

    pipeline = StructureAwareAdaptiveRAG(
        retriever=retriever,
        generator=generator,
        estimator=estimator,
        initial_k=args.initial_k,
        expanded_k=args.expanded_k,
    )

    return pipeline, query


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["demo", "hotpotqa", "nq"], default="demo")
    parser.add_argument("--use-openai", action="store_true")
    parser.add_argument("--openai-model", default="gpt-4.1-mini")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--doc-limit", type=int, default=50)
    parser.add_argument("--query-limit", type=int, default=5)
    parser.add_argument("--initial-k", type=int, default=3)
    parser.add_argument("--expanded-k", type=int, default=5)
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
