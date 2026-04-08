from rag.pipeline import (
    Query,
    SimpleGenerator,
    SimpleRetriever,
    StructureAwareAdaptiveRAG,
    SufficiencyEstimator,
    build_demo_corpus,
)


def main() -> None:
    corpus = build_demo_corpus()
    retriever = SimpleRetriever(corpus)
    generator = SimpleGenerator()
    estimator = SufficiencyEstimator()

    pipeline = StructureAwareAdaptiveRAG(
        retriever=retriever,
        generator=generator,
        estimator=estimator,
        initial_k=3,
        expanded_k=5,
    )

    query = Query("When is the birthday of Michael Phelps?")
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
