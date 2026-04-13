from __future__ import annotations

import os
import re
from dataclasses import dataclass
from statistics import mean
from typing import List, Optional, Protocol

import numpy as np


@dataclass
class Query:
    text: str
    answer: Optional[str] = None


@dataclass
class RetrievedDocument:
    doc_id: str
    text: str
    retrieval_score: float
    embedding: List[float]
    supportiveness_score: float = 0.5


class Retriever(Protocol):
    def retrieve(self, query: Query, top_k: int) -> List[RetrievedDocument]:
        ...


class Generator(Protocol):
    def generate(self, query: Query, evidence: List[RetrievedDocument]) -> str:
        ...


@dataclass
class EvidenceFeatures:
    relevance: float
    redundancy: float
    coverage: float
    supportiveness: float


@dataclass
class DecisionResult:
    features: EvidenceFeatures
    sufficiency_score: float
    sufficient: bool
    reason: str


class SimpleRetriever:
    """
    Minimal prototype retriever.
    Replace this with BM25 / FAISS / Elasticsearch later.
    """

    def __init__(self, corpus: List[RetrievedDocument]) -> None:
        self.corpus = corpus

    def retrieve(self, query: Query, top_k: int) -> List[RetrievedDocument]:
        _ = query
        return sorted(
            self.corpus,
            key=lambda doc: doc.retrieval_score,
            reverse=True,
        )[:top_k]


class SimpleGenerator:
    """
    Placeholder generator.
    Replace this with an actual LLM call later.
    """

    def generate(self, query: Query, evidence: List[RetrievedDocument]) -> str:
        joined_ids = ", ".join(doc.doc_id for doc in evidence)
        return f"Answer to '{query.text}' based on docs: {joined_ids}"


class OpenAIGenerator:
    """
    OpenAI API based generator.
    Requires OPENAI_API_KEY to be set in the environment.
    """

    def __init__(self, model: str = "gpt-4.1-mini") -> None:
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set.")

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, query: Query, evidence: List[RetrievedDocument]) -> str:
        context = "\n\n".join(
            f"[{doc.doc_id}] {doc.text}" for doc in evidence
        )
        prompt = (
            "Answer the question using only the provided evidence.\n\n"
            f"Question: {query.text}\n\n"
            f"Evidence:\n{context}\n\n"
            "Answer:"
        )
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
        )
        return response.output_text.strip()


class FaissRetriever:
    """
    FAISS-based dense retriever.
    Embeddings are produced by a sentence-transformers model.
    """

    def __init__(self, corpus: List[RetrievedDocument], model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        import faiss
        from sentence_transformers import SentenceTransformer

        self.corpus = corpus
        self.encoder = SentenceTransformer(model_name)

        matrix = np.array([doc.embedding for doc in corpus], dtype="float32")
        if len(matrix.shape) != 2:
            raise ValueError("Embeddings must be a 2D matrix.")

        dimension = matrix.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        normalized = self._normalize(matrix)
        self.index.add(normalized)

    def _normalize(self, matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return matrix / norms

    def retrieve(self, query: Query, top_k: int) -> List[RetrievedDocument]:
        query_embedding = self.encoder.encode([query.text], normalize_embeddings=True)
        scores, indices = self.index.search(np.array(query_embedding, dtype="float32"), top_k)

        results: List[RetrievedDocument] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            doc = self.corpus[idx]
            results.append(
                RetrievedDocument(
                    doc_id=doc.doc_id,
                    text=doc.text,
                    retrieval_score=float(score),
                    supportiveness_score=doc.supportiveness_score,
                    embedding=doc.embedding,
                )
            )
        return results


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_pairwise_similarities(docs: List[RetrievedDocument]) -> List[float]:
    similarities: List[float] = []
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            similarities.append(cosine_similarity(docs[i].embedding, docs[j].embedding))
    return similarities


def normalize_text(text: str) -> List[str]:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return [token for token in text.split() if token]


def min_max_normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    min_v = min(values)
    max_v = max(values)
    if max_v == min_v:
        return [1.0 for _ in values]
    return [(value - min_v) / (max_v - min_v) for value in values]


def compute_query_overlap(query: Query, doc: RetrievedDocument) -> float:
    query_tokens = set(normalize_text(query.text))
    doc_tokens = set(normalize_text(doc.text))
    if not query_tokens or not doc_tokens:
        return 0.0
    overlap = len(query_tokens & doc_tokens)
    return overlap / len(query_tokens)


def extract_question_aspects(query: Query) -> List[str]:
    tokens = normalize_text(query.text)
    stopwords = {
        "what", "when", "where", "who", "which", "why", "how", "is", "are", "was",
        "were", "the", "a", "an", "of", "in", "on", "for", "to", "did", "do", "does",
        "and", "or", "with", "by", "from", "as", "at",
    }
    filtered = [token for token in tokens if token not in stopwords]

    # Add simple bigrams to capture lightweight aspect phrases.
    bigrams = [" ".join(pair) for pair in zip(filtered, filtered[1:])]

    # Capture capitalized tokens from the original query (lightweight entity hint).
    capitalized = re.findall(r"\b[A-Z][a-zA-Z]+\b", query.text)

    aspects = list(dict.fromkeys(filtered + bigrams + capitalized))
    if not aspects:
        return tokens[:3]
    return aspects


_ASPECT_ENCODER = None


def get_aspect_encoder(model_name: str):
    global _ASPECT_ENCODER
    if _ASPECT_ENCODER is None:
        try:
            from sentence_transformers import SentenceTransformer

            _ASPECT_ENCODER = SentenceTransformer(model_name)
        except Exception:
            _ASPECT_ENCODER = False
    return _ASPECT_ENCODER


def compute_aspect_document_similarity(aspect: str, doc: RetrievedDocument, encoder=None) -> float:
    if encoder is None:
        doc_tokens = set(normalize_text(doc.text))
        if not doc_tokens:
            return 0.0
        return 1.0 if aspect in doc_tokens else 0.0

    aspect_vec = encoder.encode([aspect], normalize_embeddings=True)[0]
    return float(np.dot(aspect_vec, np.array(doc.embedding)))


def estimate_coverage(query: Query, docs: List[RetrievedDocument], model_name: str = "BAAI/bge-small-en-v1.5") -> float:
    aspects = extract_question_aspects(query)
    if not aspects or not docs:
        return 0.0

    encoder = get_aspect_encoder(model_name)
    if encoder is False:
        encoder = None

    scores: List[float] = []
    for aspect in aspects:
        best = max(compute_aspect_document_similarity(aspect, doc, encoder=encoder) for doc in docs)
        scores.append(best)
    return mean(scores)


def estimate_supportiveness(query: Query, docs: List[RetrievedDocument], normalized_scores: List[float]) -> float:
    if not docs:
        return 0.0

    scores: List[float] = []
    for doc, normalized_score in zip(docs, normalized_scores):
        overlap_score = compute_query_overlap(query, doc)
        blended_score = 0.5 * normalized_score + 0.5 * overlap_score
        if doc.supportiveness_score != 0.5:
            blended_score = 0.5 * blended_score + 0.5 * doc.supportiveness_score
        scores.append(blended_score)
    return max(scores) if scores else 0.0


def extract_evidence_features(query: Query, docs: List[RetrievedDocument], aspect_model: str = "BAAI/bge-small-en-v1.5") -> EvidenceFeatures:
    if not docs:
        return EvidenceFeatures(0.0, 1.0, 0.0, 0.0)

    raw_scores = [doc.retrieval_score for doc in docs]
    normalized_scores = min_max_normalize(raw_scores)
    relevance = mean(normalized_scores)
    supportiveness = estimate_supportiveness(query, docs, normalized_scores)

    pairwise_sims = compute_pairwise_similarities(docs)
    redundancy = mean(pairwise_sims) if pairwise_sims else 0.0
    coverage = estimate_coverage(query, docs, model_name=aspect_model)

    return EvidenceFeatures(
        relevance=relevance,
        redundancy=redundancy,
        coverage=coverage,
        supportiveness=supportiveness,
    )


class SufficiencyEstimator:
    """
    Start with a simple weighted score.
    Later you can replace this with logistic regression or another classifier.
    """

    def __init__(
        self,
        relevance_weight: float = 0.35,
        coverage_weight: float = 0.20,
        supportiveness_weight: float = 0.35,
        redundancy_weight: float = 0.20,
        threshold: float = 0.55,
        redundancy_threshold: float = 0.85,
        coverage_threshold: float = 0.25,
        supportiveness_threshold: float = 0.45,
    ) -> None:
        self.relevance_weight = relevance_weight
        self.coverage_weight = coverage_weight
        self.supportiveness_weight = supportiveness_weight
        self.redundancy_weight = redundancy_weight
        self.threshold = threshold
        self.redundancy_threshold = redundancy_threshold
        self.coverage_threshold = coverage_threshold
        self.supportiveness_threshold = supportiveness_threshold

    def _infer_reason(self, features: EvidenceFeatures, sufficient: bool) -> str:
        if sufficient:
            return "answer_sufficient"
        if features.redundancy >= self.redundancy_threshold:
            return "high_redundancy"
        if features.coverage <= self.coverage_threshold:
            return "low_coverage"
        if features.supportiveness <= self.supportiveness_threshold:
            return "weak_supportiveness"
        return "mixed_insufficiency"

    def predict(self, features: EvidenceFeatures) -> DecisionResult:
        score = (
            self.relevance_weight * features.relevance
            + self.coverage_weight * features.coverage
            + self.supportiveness_weight * features.supportiveness
            - self.redundancy_weight * features.redundancy
        )
        sufficient = score >= self.threshold
        return DecisionResult(
            features=features,
            sufficiency_score=score,
            sufficient=sufficient,
            reason=self._infer_reason(features, sufficient),
        )

    def update_parameters(
        self,
        relevance_weight: float,
        coverage_weight: float,
        supportiveness_weight: float,
        redundancy_weight: float,
        threshold: float,
    ) -> None:
        self.relevance_weight = relevance_weight
        self.coverage_weight = coverage_weight
        self.supportiveness_weight = supportiveness_weight
        self.redundancy_weight = redundancy_weight
        self.threshold = threshold


class StructureAwareAdaptiveRAG:
    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        estimator: SufficiencyEstimator,
        initial_k: int = 3,
        expanded_k: int = 5,
        aspect_model: str = "BAAI/bge-small-en-v1.5",
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.estimator = estimator
        self.initial_k = initial_k
        self.expanded_k = expanded_k
        self.aspect_model = aspect_model

    def answer(self, query: Query) -> dict:
        initial_docs = self.retriever.retrieve(query, top_k=self.initial_k)
        initial_features = extract_evidence_features(query, initial_docs, aspect_model=self.aspect_model)
        decision = self.estimator.predict(initial_features)

        if decision.sufficient:
            answer = self.generator.generate(query, initial_docs)
            return {
                "query": query.text,
                "decision": "answer_now",
                "reason": decision.reason,
                "used_docs": [doc.doc_id for doc in initial_docs],
                "features": decision.features,
                "sufficiency_score": decision.sufficiency_score,
                "answer": answer,
            }

        expanded_docs = self.retriever.retrieve(query, top_k=self.expanded_k)
        answer = self.generator.generate(query, expanded_docs)
        return {
            "query": query.text,
            "decision": "retrieve_more",
            "reason": decision.reason,
            "used_docs": [doc.doc_id for doc in expanded_docs],
            "features": decision.features,
            "sufficiency_score": decision.sufficiency_score,
            "answer": answer,
        }


def embed_corpus_texts(
    raw_docs: List[dict],
    model_name: str = "BAAI/bge-small-en-v1.5",
) -> List[RetrievedDocument]:
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(model_name)
    texts = [doc["text"] for doc in raw_docs]
    embeddings = encoder.encode(texts, normalize_embeddings=True)

    corpus: List[RetrievedDocument] = []
    for raw_doc, embedding in zip(raw_docs, embeddings):
        corpus.append(
            RetrievedDocument(
                doc_id=raw_doc["doc_id"],
                text=raw_doc["text"],
                retrieval_score=float(raw_doc.get("retrieval_score", 0.0)),
                supportiveness_score=float(raw_doc.get("supportiveness_score", 0.5)),
                embedding=list(embedding),
            )
        )
    return corpus


def load_hotpotqa_sample(limit: int = 50) -> List[dict]:
    from datasets import load_dataset

    dataset = load_dataset("hotpot_qa", "fullwiki", split=f"validation[:{limit}]")

    raw_docs: List[dict] = []
    seen_ids = set()
    for item_idx, item in enumerate(dataset):
        contexts = item["context"]
        titles = contexts["title"]
        sentences = contexts["sentences"]
        for passage_idx, (title, sentence_list) in enumerate(zip(titles, sentences)):
            doc_id = f"{item_idx}_{passage_idx}_{title}"
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            text = " ".join(sentence_list)
            if not text.strip():
                continue
            raw_docs.append(
                {
                    "doc_id": doc_id,
                    "text": f"{title}. {text}",
                    "supportiveness_score": 0.5,
                }
            )
    return raw_docs


def load_hotpotqa_queries(limit: int = 5) -> List[Query]:
    from datasets import load_dataset

    dataset = load_dataset("hotpot_qa", "fullwiki", split=f"validation[:{limit}]")
    return [Query(text=item["question"], answer=item["answer"]) for item in dataset]


def load_nq_sample(limit: int = 50) -> List[dict]:
    from datasets import load_dataset

    dataset = load_dataset("natural_questions", split=f"validation[:{limit}]")

    raw_docs: List[dict] = []
    for item_idx, item in enumerate(dataset):
        question_text = item["question"]["text"] if isinstance(item["question"], dict) else item["question"]
        doc_text = item.get("document", {}).get("text", "") if isinstance(item.get("document"), dict) else item.get("document", "")
        if not doc_text:
            continue
        raw_docs.append(
            {
                "doc_id": f"nq_{item_idx}",
                "text": doc_text,
                "supportiveness_score": 0.5,
                "question_hint": question_text,
            }
        )
    return raw_docs


def load_nq_queries(limit: int = 5) -> List[Query]:
    from datasets import load_dataset

    dataset = load_dataset("natural_questions", split=f"validation[:{limit}]")
    queries: List[Query] = []
    for item in dataset:
        question_text = item["question"]["text"] if isinstance(item["question"], dict) else item["question"]
        answers = item.get("annotations", {}).get("short_answers", []) if isinstance(item.get("annotations"), dict) else []
        answer_text = None
        if answers:
            first_answer = answers[0]
            if isinstance(first_answer, dict):
                answer_text = first_answer.get("text")
        queries.append(Query(text=question_text, answer=answer_text))
    return queries


def build_demo_corpus() -> List[RetrievedDocument]:
    return [
        RetrievedDocument(
            doc_id="doc_1",
            text="Michael Phelps was born on June 30, 1985.",
            retrieval_score=0.92,
            supportiveness_score=0.95,
            embedding=[0.9, 0.1, 0.1],
        ),
        RetrievedDocument(
            doc_id="doc_2",
            text="Phelps is an American former competitive swimmer.",
            retrieval_score=0.89,
            supportiveness_score=0.60,
            embedding=[0.88, 0.12, 0.1],
        ),
        RetrievedDocument(
            doc_id="doc_3",
            text="Phelps won many Olympic gold medals.",
            retrieval_score=0.85,
            supportiveness_score=0.45,
            embedding=[0.87, 0.1, 0.12],
        ),
        RetrievedDocument(
            doc_id="doc_4",
            text="The birthday of Michael Phelps is June 30, 1985.",
            retrieval_score=0.84,
            supportiveness_score=0.96,
            embedding=[0.91, 0.11, 0.08],
        ),
        RetrievedDocument(
            doc_id="doc_5",
            text="Michael Phelps was born in Baltimore, Maryland.",
            retrieval_score=0.81,
            supportiveness_score=0.70,
            embedding=[0.55, 0.5, 0.2],
        ),
    ]


def build_silver_label(initial_correct: bool, expanded_correct: bool) -> int:
    if initial_correct:
        return 1
    if expanded_correct:
        return 0
    return 0
