"""
Точка входа: python run.py --data ./data --out ./out

Пайплайн:
1. Data Check (проверка входных JSONL)
2. Построение поискового индекса (BM25 + Dense)
3. Для каждой декларации:
   a. Гибридный поиск кандидатов (BM25 + Dense → RRF)
   b. Cross-Encoder реранкинг
   c. Топ-10 результатов
4. Сохранение в predictions.csv
"""

import argparse
import csv
import sys
from pathlib import Path

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from config import PROJECT_ROOT, load_config
from data_check import main as data_check_main
from reranker import Reranker, rerank_candidates, build_declaration_text
from search import build_search_index, search_candidates, log_progress

# ─── Загрузка конфига ───────────────────────────────────────────────────────

config = load_config()

DEFAULT_DATA_DIR = PROJECT_ROOT / config["paths"]["data_dir"]
DEFAULT_OUT_DIR = PROJECT_ROOT / config["paths"]["out_dir"]

OUTPUT_TOP_N = config["output"]["top_n"]
PROGRESS_STEP = config.get("progress", {}).get("step", 10)


# ─── Аргументы командной строки ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Разбирает аргументы: --data и --out (переопределяют конфиг)."""
    parser = argparse.ArgumentParser(
        description="Сопоставление деклараций и регуляций"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help=f"Путь к папке с данными (по умолчанию: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help=f"Путь к папке для результатов (по умолчанию: {DEFAULT_OUT_DIR})",
    )
    return parser.parse_args()


# ─── Утилиты ─────────────────────────────────────────────────────────────────

def save_predictions_csv(results: list[dict], output_path: Path) -> None:
    """
    Сохраняет результаты в CSV.
    Колонки: declaration_id, rank, regulation_id, score
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(["declaration_id", "rank", "regulation_id", "score"])

        for row in results:
            writer.writerow([
                row["declaration_id"],
                row["rank"],
                row["regulation_id"],
                f"{row['score']:.6f}",
            ])

    print(f"  Результаты сохранены: {output_path}")


# ─── Основной пайплайн ─────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out)

    print("=" * 60)
    print("VED HELPER: Сопоставление деклараций и регуляций")
    print("=" * 60)
    print(f"  Папка с данными: {data_dir}")
    print(f"  Папка результатов: {out_dir}")

    # ── Этап 1: Data Check ─────────────────────────────────────────────
    print("\n[Этап 1/4] Проверка данных...")
    clean = data_check_main(data_dir=data_dir)

    declarations = clean.get("declarations", [])
    regulations = clean.get("regulations", [])

    if not declarations:
        print("  [ERROR] Нет чистых деклараций. Завершение.")
        return
    if not regulations:
        print("  [ERROR] Нет чистых регуляций. Завершение.")
        return

    # ── Этап 2: Построение индекса ─────────────────────────────────────
    print("\n[Этап 2/4] Построение поискового индекса...")
    bm25, dense, lemmatizer = build_search_index(regulations)
    reranker = Reranker()

    # ── Этап 3: Обработка деклараций ───────────────────────────────────
    print(f"\n[Этап 3/4] Обработка деклараций...")
    total = len(declarations)
    all_results: list[dict] = []

    for i, declaration in enumerate(declarations, start=1):
        decl_id = declaration["declaration_id"]
        decl_text = build_declaration_text(declaration)

        # 3a. Гибридный поиск
        candidates = search_candidates(decl_text, bm25, dense, lemmatizer)

        # 3b. Подставляем regulation_id
        for cand in candidates:
            cand["regulation_id"] = regulations[cand["reg_index"]]["regulation_id"]

        # 3c. Реранкинг
        ranked = rerank_candidates(decl_text, candidates, regulations, reranker)

        # 3d. Топ-N
        for rank, cand in enumerate(ranked[:OUTPUT_TOP_N], start=1):
            all_results.append({
                "declaration_id": decl_id,
                "rank": rank,
                "regulation_id": cand["regulation_id"],
                "score": cand["score"],
            })

        # Прогресс
        log_progress(i, total, PROGRESS_STEP)

    # ── Этап 4: Сохранение ─────────────────────────────────────────────
    print(f"\n[Этап 4/4] Сохранение результатов...")
    output_file = out_dir / "predictions.csv"
    save_predictions_csv(all_results, output_file)

    print("\n" + "=" * 60)
    print("ГОТОВО")
    print("=" * 60)


if __name__ == "__main__":
    main()