"""
Cross-Encoder реранкинг.
Принимает декларацию и список кандидатов, возвращает финальный скор 0..1.
"""

from typing import Any

from sentence_transformers import CrossEncoder

from config import load_config

# ─── Загрузка конфига ───────────────────────────────────────────────────────

config = load_config()

RERANKER_MODEL_NAME = config["models"]["reranker_model"]
RERANK_BATCH_SIZE = config["reranking"]["batch_size"]
RERANK_MAX_LENGTH = config["reranking"]["max_length"]


# ─── Класс реранкера ────────────────────────────────────────────────────────

class Reranker:
    """Обёртка над CrossEncoder для финального скоринга пар."""

    def __init__(self, model_name: str = RERANKER_MODEL_NAME, max_length: int = RERANK_MAX_LENGTH):
        print(f"  Загрузка реранкер-модели: {model_name}")
        self.model = CrossEncoder(
            model_name,
            max_length=max_length,
        )

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        """
        Принимает список пар (текст_декларации, текст_регуляции).
        Возвращает список скоров 0..1.
        """
        if not pairs:
            return []

        scores = self.model.predict(
            pairs,
            batch_size=RERANK_BATCH_SIZE,
            show_progress_bar=False,
        )

        # CrossEncoder может возвращать как один скор, так и массив
        # Приводим к списку float
        if isinstance(scores, float):
            scores = [scores]

        return [float(s) for s in scores]


# ─── Функция реранкинга кандидатов ─────────────────────────────────────────

def rerank_candidates(
    declaration_text: str,
    candidates: list[dict],
    regulations: list[dict],
    reranker: Reranker,
) -> list[dict]:
    """
    Прогоняет кандидатов через кросс-энкодер и добавляет финальный скор.

    candidates — список из search_candidates:
    [
        {"reg_index": int, "rrf_score": float, ...},
        ...
    ]

    Возвращает список с добавленным полем "score":
    [
        {"reg_index": int, "rrf_score": float, "score": float, ...},
        ...
    ]
    """
    # Формируем пары (декларация, описание регуляции)
    pairs: list[tuple[str, str]] = []
    for cand in candidates:
        reg_text = regulations[cand["reg_index"]]["description"]
        pairs.append((declaration_text, reg_text))

    # Считаем скоры
    scores = reranker.score_pairs(pairs)

    # Добавляем скор к кандидатам
    for cand, score in zip(candidates, scores):
        cand["score"] = score

    # Сортируем по убыванию скора
    candidates.sort(key=lambda x: x["score"], reverse=True)

    return candidates


# ─── Утилита для подготовки текста декларации ──────────────────────────────

def build_declaration_text(declaration: dict) -> str:
    """Склеивает основное и расширенное описание декларации."""
    text = declaration.get("G31_1", "")
    ext = declaration.get("desc_extention")
    if ext:
        text += " " + ext
    return text.strip()


# ─── Основная точка входа (для теста модуля) ───────────────────────────────

if __name__ == "__main__":
    # Пример использования
    sample_decl = "СВИНИНА МОРОЖЕНАЯ: РЕБЕРНЫЕ ОТРУБЫ ДОМАШНИХ СВИНЕЙ НА КОСТИ"

    sample_candidates = [
        {"reg_index": 0, "rrf_score": 0.045, "rank_bm25": 1, "rank_dense": 2},
        {"reg_index": 1, "rrf_score": 0.032, "rank_bm25": 4, "rank_dense": 1},
        {"reg_index": 2, "rrf_score": 0.021, "rank_bm25": 7, "rank_dense": 5},
    ]

    sample_regs = [
        {"regulation_id": "R0001", "description": "Свинина мороженая..."},
        {"regulation_id": "R0002", "description": "Мясо и пищевые субпродукты..."},
        {"regulation_id": "R0003", "description": "Лекарственные средства..."},
    ]

    reranker = Reranker()
    result = rerank_candidates(sample_decl, sample_candidates, sample_regs, reranker)

    for item in result:
        print(f"  {item['reg_index']}: {item['score']:.4f}")