"""
Гибридный поиск: BM25 + Dense → RRF Fusion.
Возвращает топ-N кандидатов для каждой декларации.
"""

import math
from pathlib import Path
from typing import Any

import numpy as np
from pymystem3 import Mystem
from sentence_transformers import SentenceTransformer

from config import PROJECT_ROOT, load_config

# ─── Загрузка конфига ───────────────────────────────────────────────────────

config = load_config()

EMBEDDING_MODEL_NAME = config["models"]["embedding_model"]
RERANKER_MODEL_NAME = config["models"]["reranker_model"]

BM25_TOP_N = config["search"]["bm25_top_n"]
DENSE_TOP_N = config["search"]["dense_top_n"]
RRF_K = config["search"]["rrf_k"]
CANDIDATES_TOP_N = config["search"]["candidates_top_n"]

RERANK_BATCH_SIZE = config["reranking"]["batch_size"]
RERANK_MAX_LENGTH = config["reranking"]["max_length"]

OUTPUT_TOP_N = config["output"]["top_n"]


# ─── BM25 ───────────────────────────────────────────────────────────────────

class BM25:
    """Простая реализация BM25 с лемматизацией."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_len = [len(doc) for doc in corpus]
        self.avg_doc_len = sum(self.doc_len) / max(len(corpus), 1)

        self.doc_freqs: list[dict[str, int]] = []
        self.inverted_index: dict[str, set[int]] = {}

        for doc_id, doc in enumerate(corpus):
            freqs: dict[str, int] = {}
            for term in doc:
                freqs[term] = freqs.get(term, 0) + 1
            self.doc_freqs.append(freqs)

            for term in freqs:
                if term not in self.inverted_index:
                    self.inverted_index[term] = set()
                self.inverted_index[term].add(doc_id)

        self.num_docs = len(corpus)

    def _idf(self, term: str) -> float:
        """Inverse Document Frequency."""
        df = len(self.inverted_index.get(term, set()))
        if df == 0:
            return 0.0
        return math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1)

    def score(self, query: list[str], doc_id: int) -> float:
        """Скор BM25 для одного документа."""
        score = 0.0
        doc_freq = self.doc_freqs[doc_id]
        doc_len = self.doc_len[doc_id]

        for term in query:
            tf = doc_freq.get(term, 0)
            if tf == 0:
                continue

            idf = self._idf(term)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
            score += idf * (numerator / denominator)

        return score

    def search(self, query: list[str], top_n: int) -> list[tuple[int, float]]:
        """Возвращает список (doc_id, score) длиной top_n."""
        scores = [(doc_id, self.score(query, doc_id)) for doc_id in range(self.num_docs)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]


# ─── Лемматизация ───────────────────────────────────────────────────────────

class Lemmatizer:
    """Лемматизатор на основе Mystem."""

    def __init__(self):
        self._mystem = Mystem()

    def lemmatize(self, text: str) -> list[str]:
        """Приводит текст к списку лемм (без пунктуации и стоп-символов)."""
        if not text:
            return []
        lemmas = self._mystem.lemmatize(text.lower())
        return [
            lemma.strip()
            for lemma in lemmas
            if lemma.strip() and not lemma.isspace()
        ]


# ─── Dense Retrieval ────────────────────────────────────────────────────────

class DenseRetriever:
    """Семантический поиск на эмбеддингах."""

    def __init__(self, model_name: str):
        print(f"  Загрузка эмбеддинг-модели: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.reg_vectors: np.ndarray | None = None

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Кодирует тексты в векторы."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def build_index(self, regulations: list[dict]) -> None:
        """Кодирует все регуляции один раз."""
        texts = [reg["description"] for reg in regulations]
        self.reg_vectors = self.encode(texts)

    def search(self, query_text: str, top_n: int) -> list[tuple[int, float]]:
        """Возвращает (doc_id, cosine_sim) для топ-N похожих регуляций."""
        query_vector = self.encode([query_text])[0]
        similarities = np.dot(self.reg_vectors, query_vector)
        top_indices = np.argsort(similarities)[::-1][:top_n]
        return [(int(idx), float(similarities[idx])) for idx in top_indices]


# ─── RRF Fusion ─────────────────────────────────────────────────────────────

def rrf_fusion(
    bm25_results: list[tuple[int, float]],
    dense_results: list[tuple[int, float]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Объединяет два ранжированных списка через Reciprocal Rank Fusion."""
    rrf_scores: dict[int, float] = {}

    for rank, (doc_id, _) in enumerate(bm25_results):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    for rank, (doc_id, _) in enumerate(dense_results):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return fused


# ─── Построение поискового индекса ─────────────────────────────────────────

def build_search_index(regulations: list[dict]) -> tuple[BM25, DenseRetriever, Lemmatizer]:
    """Строит BM25-индекс и dense-индекс по регуляциям."""
    print("Построение поискового индекса...")

    lemmatizer = Lemmatizer()

    # 1. Лемматизируем описания регуляций для BM25
    print("  Лемматизация регуляций...")
    reg_texts = [reg["description"] for reg in regulations]
    reg_lemmas = [lemmatizer.lemmatize(text) for text in reg_texts]

    # 2. Строим BM25
    print("  Построение BM25-индекса...")
    bm25 = BM25(reg_lemmas)

    # 3. Строим Dense-индекс
    print("  Построение Dense-индекса...")
    dense = DenseRetriever(EMBEDDING_MODEL_NAME)
    dense.build_index(regulations)

    return bm25, dense, lemmatizer


# ─── Основная функция поиска ────────────────────────────────────────────────

def search_candidates(
    declaration_text: str,
    bm25: BM25,
    dense: DenseRetriever,
    lemmatizer: Lemmatizer,
) -> list[dict]:
    """
    Гибридный поиск кандидатов.
    Возвращает список из CANDIDATES_TOP_N словарей:
    [
        {
            "regulation_id": str,
            "rrf_score": float,
            "rank_bm25": int | None,
            "rank_dense": int | None,
        },
        ...
    ]
    """
    # 1. Лемматизация запроса
    query_lemmas = lemmatizer.lemmatize(declaration_text)

    # 2. BM25 поиск
    bm25_results = bm25.search(query_lemmas, BM25_TOP_N)

    # 3. Dense поиск
    dense_results = dense.search(declaration_text, DENSE_TOP_N)

    # 4. RRF Fusion
    fused = rrf_fusion(bm25_results, dense_results, k=RRF_K)

    # 5. Топ-N кандидатов
    top_candidates = fused[:CANDIDATES_TOP_N]

    # 6. Формируем результат
    bm25_rank_map = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(bm25_results)}
    dense_rank_map = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(dense_results)}

    candidates = []
    for doc_id, rrf_score in top_candidates:
        candidates.append({
            "regulation_id": None,  # будет заполнено в main.py, чтобы не тащить сюда данные
            "reg_index": doc_id,
            "rrf_score": rrf_score,
            "rank_bm25": bm25_rank_map.get(doc_id),
            "rank_dense": dense_rank_map.get(doc_id),
        })

    return candidates


# ─── Прогресс-бар ───────────────────────────────────────────────────────────

def log_progress(current: int, total: int, step: int = 10) -> None:
    """Пишет прогресс в консоль каждые step обработанных деклараций."""
    if current % step == 0 or current == total:
        print(f"  Обработано {current} из {total}")