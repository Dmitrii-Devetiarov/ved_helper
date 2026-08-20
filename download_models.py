"""
Предварительное скачивание моделей.
Запускается ДО основного run.py:

    python download_models.py

Модели сохраняются в локальный кэш Hugging Face:
    ~/.cache/huggingface/hub/

После скачивания run.py работает офлайн.
"""

import sys
from pathlib import Path

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from sentence_transformers import SentenceTransformer, CrossEncoder

from config import load_config

# ─── Загрузка конфига ───────────────────────────────────────────────────────

config = load_config()

EMBEDDING_MODEL_NAME = config["models"]["embedding_model"]
RERANKER_MODEL_NAME = config["models"]["reranker_model"]


# ─── Основная функция ───────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("DOWNLOAD MODELS")
    print("=" * 60)

    # 1. Эмбеддинг-модель
    print(f"\n[1/2] Скачивание эмбеддинг-модели: {EMBEDDING_MODEL_NAME}")
    print("  (может занять несколько минут при первом запуске)")
    SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("  Готово.")

    # 2. Реранкер
    print(f"\n[2/2] Скачивание реранкер-модели: {RERANKER_MODEL_NAME}")
    print("  (может занять несколько минут при первом запуске)")
    CrossEncoder(RERANKER_MODEL_NAME)
    print("  Готово.")

    print("\n" + "=" * 60)
    print("Все модели скачаны. Можно запускать run.py офлайн.")
    print("=" * 60)


if __name__ == "__main__":
    main()