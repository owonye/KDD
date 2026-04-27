from __future__ import annotations

import json
import os
import re
import time
import hashlib
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from statistics import mean
from typing import List, Optional, Protocol

import numpy as np

CONTENT_STOPWORDS = {
    "what", "when", "where", "who", "which", "why", "how", "is", "are", "was",
    "were", "the", "a", "an", "of", "in", "on", "for", "to", "did", "do", "does",
    "and", "or", "with", "by", "from", "as", "at",
}
SUPPORTIVENESS_LAMBDA = 0.5
GENERATOR_PROMPT_VERSION = "short_answer_v2"


@dataclass
class Query:
    text: str
    answer: Optional[str] = None
    answers: Optional[List[str]] = None
    query_id: Optional[str] = None


@dataclass
class RetrievedDocument:
    doc_id: str
    text: str
    retrieval_score: float
    embedding: List[float]


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

    def __init__(self, model: str = "gpt-4.1-mini", cache_path: Optional[str] = None) -> None:
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set.")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self._cache: dict[str, str] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_path = cache_path or os.getenv("OPENAI_CACHE_PATH", "")
        if self.cache_path:
            self._load_disk_cache()

    def _load_disk_cache(self) -> None:
        path = self._cache_file_path()
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = entry.get("key")
                value = entry.get("value")
                if isinstance(key, str) and isinstance(value, str):
                    self._cache[key] = value

    def _cache_file_path(self) -> Path:
        return Path(self.cache_path)

    def _append_disk_cache(self, key: str, value: str) -> None:
        path = self._cache_file_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")

    def get_cache_stats(self) -> dict[str, int]:
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_size": len(self._cache),
        }

    def _cache_key(self, query: Query, evidence: List[RetrievedDocument]) -> str:
        ids = "|".join(doc.doc_id for doc in evidence)
        return f"{GENERATOR_PROMPT_VERSION}::{self.model}::{query.text}::{ids}"

    def _request_with_retry(self, prompt: str) -> str:
        for attempt in range(3):
            try:
                if hasattr(self.client, "responses"):
                    response = self.client.responses.create(
                        model=self.model,
                        input=prompt,
                    )
                    return response.output_text.strip()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                return (response.choices[0].message.content or "").strip()
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1.5 * (attempt + 1))
        return ""

    def generate(self, query: Query, evidence: List[RetrievedDocument]) -> str:
        context = "\n\n".join(
            f"[{doc.doc_id}] {doc.text}" for doc in evidence
        )
        prompt = (
            "Answer the question using only the provided evidence.\n"
            "Return only the shortest answer span or phrase. Do not write a sentence unless the answer requires it.\n"
            "If the evidence does not contain the answer, return exactly: unknown\n\n"
            f"Question: {query.text}\n\n"
            f"Evidence:\n{context}\n\n"
            "Short answer:"
        )
        key = self._cache_key(query, evidence)
        cached = self._cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        self.cache_misses += 1
        output = self._request_with_retry(prompt)
        self._cache[key] = output
        if self.cache_path:
            self._append_disk_cache(key, output)
        return output


class FaissRetriever:
    """
    FAISS-based dense retriever.
    Embeddings are produced by a sentence-transformers model.
    """

    def __init__(
        self,
        corpus: List[RetrievedDocument],
        model_name: str = "BAAI/bge-small-en-v1.5",
        cache_dir: str = "",
        cache_namespace: str = "",
    ) -> None:
        import faiss
        from sentence_transformers import SentenceTransformer

        self.corpus = corpus
        self.model_name = model_name
        self.encoder = SentenceTransformer(model_name)
        self.cache_dir = cache_dir
        self.cache_namespace = cache_namespace

        matrix = np.array([doc.embedding for doc in corpus], dtype="float32")
        if len(matrix.shape) != 2:
            raise ValueError("Embeddings must be a 2D matrix.")

        dimension = matrix.shape[1]
        cached_index = self._load_cached_index(dimension)
        if cached_index is not None:
            self.index = cached_index
            return

        self.index = faiss.IndexFlatIP(dimension)
        normalized = self._normalize(matrix)
        self.index.add(normalized)
        self._save_cached_index()

    def _normalize(self, matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return matrix / norms

    def _doc_ids_digest(self) -> str:
        hasher = hashlib.sha256()
        for doc in self.corpus:
            hasher.update(doc.doc_id.encode("utf-8"))
            hasher.update(b"\n")
        return hasher.hexdigest()[:16]

    def _index_cache_key(self) -> str:
        raw = f"{self.model_name}::{self.cache_namespace}::{len(self.corpus)}::{self._doc_ids_digest()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def _index_cache_path(self) -> Optional[Path]:
        if not self.cache_dir or not self.cache_namespace:
            return None
        cache_root = Path(self.cache_dir) / "faiss"
        return cache_root / f"{self._index_cache_key()}.index"

    def _load_cached_index(self, dimension: int):
        import faiss

        cache_path = self._index_cache_path()
        if cache_path is None or not cache_path.exists():
            return None
        try:
            index = faiss.read_index(str(cache_path))
            if index.d != dimension or index.ntotal != len(self.corpus):
                return None
            return index
        except Exception:
            return None

    def _save_cached_index(self) -> None:
        import faiss

        cache_path = self._index_cache_path()
        if cache_path is None:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(cache_path))

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
    query_tokens = set(extract_content_tokens(query.text))
    doc_tokens = set(normalize_text(doc.text))
    if not query_tokens or not doc_tokens:
        return 0.0
    overlap = len(query_tokens & doc_tokens)
    return overlap / len(query_tokens)


def extract_content_tokens(text: str) -> List[str]:
    tokens = normalize_text(text)
    filtered = [token for token in tokens if token not in CONTENT_STOPWORDS]
    return filtered if filtered else tokens


def extract_question_aspects(query: Query) -> List[str]:
    tokens = normalize_text(query.text)
    filtered = extract_content_tokens(query.text)

    # Add simple bigrams to capture lightweight aspect phrases.
    bigrams = [" ".join(pair) for pair in zip(filtered, filtered[1:])]

    # Capture capitalized tokens from the original query (lightweight entity hint).
    capitalized = re.findall(r"\b[A-Z][a-zA-Z]+\b", query.text)
    # Capture simple numeric patterns as lightweight aspect cues.
    numeric_like = re.findall(r"\b\d+[a-zA-Z0-9\-]*\b", query.text)

    aspects = list(dict.fromkeys(filtered + bigrams + capitalized + numeric_like))
    if not aspects:
        return tokens[:3]
    return aspects


_ASPECT_ENCODER_BY_MODEL: dict[str, object] = {}


def get_aspect_encoder(model_name: str):
    if not model_name:
        return False
    cached = _ASPECT_ENCODER_BY_MODEL.get(model_name)
    if cached is not None:
        return cached
    if model_name not in _ASPECT_ENCODER_BY_MODEL:
        try:
            from sentence_transformers import SentenceTransformer

            _ASPECT_ENCODER_BY_MODEL[model_name] = SentenceTransformer(model_name)
        except Exception:
            _ASPECT_ENCODER_BY_MODEL[model_name] = False
    return _ASPECT_ENCODER_BY_MODEL[model_name]


def compute_aspect_document_similarity(aspect: str, doc: RetrievedDocument, encoder=None) -> float:
    if encoder is None:
        return compute_aspect_lexical_overlap(aspect, doc)

    aspect_tokens = normalize_text(aspect)
    if not aspect_tokens:
        return compute_aspect_lexical_overlap(aspect, doc)

    # Build an aspect vector from token-level embeddings and average them.
    token_vectors = encoder.encode(aspect_tokens, normalize_embeddings=True)
    if len(token_vectors) == 0:
        return compute_aspect_lexical_overlap(aspect, doc)

    aspect_vec = np.mean(np.array(token_vectors, dtype="float32"), axis=0)
    norm = np.linalg.norm(aspect_vec)
    if norm == 0.0:
        return compute_aspect_lexical_overlap(aspect, doc)
    aspect_vec = aspect_vec / norm
    doc_vec = np.array(doc.embedding)
    if len(aspect_vec) != len(doc_vec):
        return compute_aspect_lexical_overlap(aspect, doc)
    return float(np.dot(aspect_vec, doc_vec))


def compute_aspect_lexical_overlap(aspect: str, doc: RetrievedDocument) -> float:
    aspect_tokens = set(normalize_text(aspect))
    doc_tokens = set(normalize_text(doc.text))
    if not aspect_tokens or not doc_tokens:
        return 0.0
    return len(aspect_tokens & doc_tokens) / len(aspect_tokens)


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
        blended_score = SUPPORTIVENESS_LAMBDA * normalized_score + (1.0 - SUPPORTIVENESS_LAMBDA) * overlap_score
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
                "initial_doc_ids": [doc.doc_id for doc in initial_docs],
                "final_doc_ids": [doc.doc_id for doc in initial_docs],
                "expanded_doc_ids": [],
                "expansion_triggered": False,
                "final_doc_count": len(initial_docs),
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
            "initial_doc_ids": [doc.doc_id for doc in initial_docs],
            "final_doc_ids": [doc.doc_id for doc in expanded_docs],
            "expanded_doc_ids": [doc.doc_id for doc in expanded_docs],
            "expansion_triggered": True,
            "final_doc_count": len(expanded_docs),
            "features": decision.features,
            "sufficiency_score": decision.sufficiency_score,
            "answer": answer,
        }


def embed_corpus_texts(
    raw_docs: List[dict],
    model_name: str = "BAAI/bge-small-en-v1.5",
    cache_dir: str = "",
    cache_namespace: str = "",
) -> List[RetrievedDocument]:
    from sentence_transformers import SentenceTransformer

    cache_path = _embedding_cache_path(cache_dir, cache_namespace, model_name, len(raw_docs))
    if cache_path is not None:
        cached = _load_embedding_cache(cache_path)
        if cached is not None:
            return cached

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
                embedding=list(embedding),
            )
        )

    if cache_path is not None:
        _save_embedding_cache(cache_path, corpus)
    return corpus


def _embedding_cache_key(cache_namespace: str, model_name: str, size: int) -> str:
    raw = f"{cache_namespace}::{model_name}::{size}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _embedding_cache_path(
    cache_dir: str,
    cache_namespace: str,
    model_name: str,
    size: int,
) -> Optional[Path]:
    if not cache_dir or not cache_namespace:
        return None
    return Path(cache_dir) / "embeddings" / f"{_embedding_cache_key(cache_namespace, model_name, size)}.npz"


def _load_embedding_cache(path: Path) -> Optional[List[RetrievedDocument]]:
    if not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=True)
        doc_ids = data["doc_ids"]
        texts = data["texts"]
        retrieval_scores = data["retrieval_scores"]
        embeddings = data["embeddings"]
        if not (len(doc_ids) == len(texts) == len(retrieval_scores) == len(embeddings)):
            return None
        corpus: List[RetrievedDocument] = []
        for idx in range(len(doc_ids)):
            corpus.append(
                RetrievedDocument(
                    doc_id=str(doc_ids[idx]),
                    text=str(texts[idx]),
                    retrieval_score=float(retrieval_scores[idx]),
                    embedding=list(np.array(embeddings[idx], dtype="float32")),
                )
            )
        return corpus
    except Exception:
        return None


def _save_embedding_cache(path: Path, corpus: List[RetrievedDocument]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc_ids = np.array([doc.doc_id for doc in corpus], dtype=object)
    texts = np.array([doc.text for doc in corpus], dtype=object)
    retrieval_scores = np.array([doc.retrieval_score for doc in corpus], dtype="float32")
    embeddings = np.array([doc.embedding for doc in corpus], dtype="float32")
    np.savez_compressed(
        path,
        doc_ids=doc_ids,
        texts=texts,
        retrieval_scores=retrieval_scores,
        embeddings=embeddings,
    )


def load_hotpotqa_sample(start: int = 0, limit: int = 50, split: str = "validation") -> List[dict]:
    from datasets import load_dataset

    dataset = load_dataset("hotpot_qa", "fullwiki", split=f"{split}[{start}:{start + limit}]")

    raw_docs: List[dict] = []
    seen_ids = set()
    for item_idx, item in enumerate(dataset):
        absolute_item_idx = start + item_idx
        contexts = item["context"]
        titles = contexts["title"]
        sentences = contexts["sentences"]
        for passage_idx, (title, sentence_list) in enumerate(zip(titles, sentences)):
            normalized_title = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
            doc_id = f"hotpotqa::{split}::{absolute_item_idx}::{passage_idx}::{normalized_title}"
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
                }
            )
    return raw_docs


def load_hotpotqa_queries(start: int = 0, limit: int = 5, split: str = "validation") -> List[Query]:
    from datasets import load_dataset

    dataset = load_dataset("hotpot_qa", "fullwiki", split=f"{split}[{start}:{start + limit}]")
    queries: List[Query] = []
    for local_idx, item in enumerate(dataset):
        query_uid = f"hotpotqa::{split}::{start + local_idx}"
        queries.append(Query(text=item["question"], answer=item["answer"], answers=[item["answer"]], query_id=query_uid))
    return queries


def _extract_nq_document_tokens(item: dict) -> List[str]:
    document = item.get("document")
    if not isinstance(document, dict):
        return []

    tokens_field = document.get("tokens")
    if isinstance(tokens_field, dict):
        # Common schema: {"token": [...], "is_html": [...]}
        token_values = tokens_field.get("token")
        html_flags = tokens_field.get("is_html")
        if isinstance(token_values, list):
            if isinstance(html_flags, list) and len(html_flags) == len(token_values):
                return [
                    str(token)
                    for token, is_html in zip(token_values, html_flags)
                    if not is_html and str(token).strip()
                ]
            return [str(token) for token in token_values if str(token).strip()]
    elif isinstance(tokens_field, list):
        values: List[str] = []
        for entry in tokens_field:
            if isinstance(entry, dict):
                token = entry.get("token")
                if token:
                    values.append(str(token))
            elif entry:
                values.append(str(entry))
        return values

    text_field = document.get("text")
    if isinstance(text_field, str) and text_field.strip():
        return text_field.split()
    return []


def _extract_nq_short_answer_texts(item: dict) -> List[str]:
    annotations = item.get("annotations")
    candidate_answers: List[dict] = []

    if isinstance(annotations, dict):
        raw = annotations.get("short_answers", [])
        if isinstance(raw, list):
            candidate_answers.extend([answer for answer in raw if isinstance(answer, dict)])
    elif isinstance(annotations, list):
        for annotation in annotations:
            if not isinstance(annotation, dict):
                continue
            raw = annotation.get("short_answers", [])
            if isinstance(raw, list):
                candidate_answers.extend([answer for answer in raw if isinstance(answer, dict)])

    if not candidate_answers:
        return []

    outputs: List[str] = []
    for answer in candidate_answers:
        answer_text = answer.get("text")
        if isinstance(answer_text, str) and answer_text.strip():
            outputs.append(answer_text.strip())
            continue

        start_token = answer.get("start_token")
        end_token = answer.get("end_token")
        if isinstance(start_token, int) and isinstance(end_token, int) and end_token > start_token:
            tokens = _extract_nq_document_tokens(item)
            if tokens and start_token < len(tokens):
                clipped_end = min(end_token, len(tokens))
                span = tokens[start_token:clipped_end]
                span_text = " ".join(token for token in span if token).strip()
                if span_text:
                    outputs.append(span_text)

    deduped = list(dict.fromkeys(outputs))
    return deduped


def _chunk_tokens(tokens: List[str], max_tokens: int, stride: int) -> List[List[str]]:
    if max_tokens <= 0:
        return [tokens]
    if stride <= 0:
        stride = max_tokens

    chunks: List[List[str]] = []
    for chunk_start in range(0, len(tokens), stride):
        chunk = tokens[chunk_start:chunk_start + max_tokens]
        if not chunk:
            continue
        chunks.append(chunk)
        if chunk_start + max_tokens >= len(tokens):
            break
    return chunks


def _load_nq_stream_slice(start: int, limit: int, split: str):
    from datasets import load_dataset

    dataset = load_dataset("natural_questions", split=split, streaming=True)
    return islice(dataset, start, start + limit)


def load_nq_sample(
    start: int = 0,
    limit: int = 50,
    split: str = "validation",
    max_tokens: int = 220,
    stride: int = 110,
) -> List[dict]:
    raw_docs: List[dict] = []
    for item_idx, item in enumerate(_load_nq_stream_slice(start=start, limit=limit, split=split)):
        absolute_item_idx = start + item_idx
        question_text = item["question"]["text"] if isinstance(item["question"], dict) else item["question"]
        tokens = _extract_nq_document_tokens(item)
        if not tokens:
            doc_text = item.get("document", {}).get("text", "") if isinstance(item.get("document"), dict) else item.get("document", "")
            tokens = doc_text.split() if isinstance(doc_text, str) else []
        if not tokens:
            continue
        for chunk_idx, chunk_tokens in enumerate(_chunk_tokens(tokens, max_tokens=max_tokens, stride=stride)):
            doc_text = " ".join(chunk_tokens).strip()
            if not doc_text:
                continue
            raw_docs.append(
                {
                    "doc_id": f"nq::{split}::{absolute_item_idx}::chunk_{chunk_idx}",
                    "text": doc_text,
                    "question_hint": question_text,
                }
            )
    return raw_docs


def load_nq_queries(start: int = 0, limit: int = 5, split: str = "validation") -> List[Query]:
    queries: List[Query] = []
    for local_idx, item in enumerate(_load_nq_stream_slice(start=start, limit=limit, split=split)):
        question_text = item["question"]["text"] if isinstance(item["question"], dict) else item["question"]
        answers = _extract_nq_short_answer_texts(item)
        answer_text = answers[0] if answers else None
        query_uid = f"nq::{split}::{start + local_idx}"
        queries.append(Query(text=question_text, answer=answer_text, answers=answers, query_id=query_uid))
    return queries


def build_demo_corpus() -> List[RetrievedDocument]:
    return [
        RetrievedDocument(
            doc_id="doc_1",
            text="Michael Phelps was born on June 30, 1985.",
            retrieval_score=0.92,
            embedding=[0.9, 0.1, 0.1],
        ),
        RetrievedDocument(
            doc_id="doc_2",
            text="Phelps is an American former competitive swimmer.",
            retrieval_score=0.89,
            embedding=[0.88, 0.12, 0.1],
        ),
        RetrievedDocument(
            doc_id="doc_3",
            text="Phelps won many Olympic gold medals.",
            retrieval_score=0.85,
            embedding=[0.87, 0.1, 0.12],
        ),
        RetrievedDocument(
            doc_id="doc_4",
            text="The birthday of Michael Phelps is June 30, 1985.",
            retrieval_score=0.84,
            embedding=[0.91, 0.11, 0.08],
        ),
        RetrievedDocument(
            doc_id="doc_5",
            text="Michael Phelps was born in Baltimore, Maryland.",
            retrieval_score=0.81,
            embedding=[0.55, 0.5, 0.2],
        ),
    ]


def build_silver_label(initial_correct: bool, expanded_correct: bool) -> Optional[int]:
    # Sufficient if weak support is already present in D.
    if initial_correct:
        return 1
    # Insufficient if support appears only after expansion.
    if expanded_correct:
        return 0
    # Unknown when support is absent in both D and D+.
    return None
