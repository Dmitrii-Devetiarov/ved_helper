"""
Поиск топ-10 деклараций с наихудшим скором на первом ранге.
Показывает реально проблемные случаи, без усреднения.
"""

import csv
import sys
from pathlib import Path

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import PROJECT_ROOT, load_config

# ─── Загрузка конфига ───────────────────────────────────────────────────────

config = load_config()

DATA_DIR = PROJECT_ROOT / config["paths"]["data_dir"]
OUT_DIR = PROJECT_ROOT / config["paths"]["out_dir"]
VAL_OUT_DIR = PROJECT_ROOT / config["paths"]["val_out_dir"]

DECLARATIONS_FILE = DATA_DIR / "declarations.jsonl"
RESULTS_FILE = OUT_DIR / "results.csv"

OUTPUT_CSV = VAL_OUT_DIR / "worst_declarations.csv"

TOP_WORST_N = 10


# ─── Чтение данных ─────────────────────────────────────────────────────────

def read_jsonl(path: Path) -> list[dict]:
    """Читает JSONL-файл."""
    import json
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_results_csv(path: Path) -> dict[str, list[dict]]:
    """
    Читает results.csv.
    Возвращает: {declaration_id: [ {rank, regulation_id, score}, ... ]}
    """
    results: dict[str, list[dict]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            decl_id = row["declaration_id"]
            if decl_id not in results:
                results[decl_id] = []
            results[decl_id].append({
                "rank": int(row["rank"]),
                "regulation_id": row["regulation_id"],
                "score": float(row["score"]),
            })
    return results


def find_declaration(declarations: list[dict], decl_id: str) -> dict | None:
    """Ищет декларацию по ID."""
    for d in declarations:
        if d["declaration_id"] == decl_id:
            return d
    return None


# ─── Поиск худших ───────────────────────────────────────────────────────────

def find_worst_declarations(
    declarations: list[dict],
    results: dict[str, list[dict]],
    top_n: int = 10,
) -> list[dict]:
    """
    Находит топ-N деклараций с наименьшим скором на первом ранге.
    Возвращает список словарей:
    [
        {
            "declaration_id": str,
            "top1_score": float,
            "top1_regulation_id": str,
            "description": str,
        },
        ...
    ]
    """
    worst: list[dict] = []

    for decl in declarations:
        decl_id = decl["declaration_id"]

        if decl_id not in results:
            worst.append({
                "declaration_id": decl_id,
                "top1_score": 0.0,
                "top1_regulation_id": "",
                "description": decl.get("G31_1", ""),
            })
            continue

        top_regs = sorted(results[decl_id], key=lambda x: x["rank"])
        if not top_regs:
            worst.append({
                "declaration_id": decl_id,
                "top1_score": 0.0,
                "top1_regulation_id": "",
                "description": decl.get("G31_1", ""),
            })
            continue

        rank1 = top_regs[0]
        worst.append({
            "declaration_id": decl_id,
            "top1_score": rank1["score"],
            "top1_regulation_id": rank1["regulation_id"],
            "description": decl.get("G31_1", "") + " " + (decl.get("desc_extention") or ""),
        })

    # Сортируем по возрастанию top1_score
    worst.sort(key=lambda x: x["top1_score"])

    return worst[:top_n]


def save_worst_csv(worst: list[dict], output_path: Path) -> None:
    """Сохраняет худшие декларации в CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(["declaration_id", "top1_score", "top1_regulation_id", "description"])
        for row in worst:
            writer.writerow([
                row["declaration_id"],
                f"{row['top1_score']:.6f}",
                row["top1_regulation_id"],
                row["description"],
            ])

    print(f"  Результаты сохранены: {output_path}")


# ─── Основная функция ───────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("WORST CASES: Худшие декларации по топ-1")
    print("=" * 60)

    # 1. Читаем данные
    print("\n[1/3] Чтение данных...")
    declarations = read_jsonl(DECLARATIONS_FILE)
    results = read_results_csv(RESULTS_FILE)

    print(f"  Деклараций: {len(declarations)}")
    print(f"  Результатов (деклараций в CSV): {len(results)}")

    # 2. Ищем худших
    print(f"\n[2/3] Поиск топ-{TOP_WORST_N} худших...")
    worst = find_worst_declarations(declarations, results, TOP_WORST_N)

    # 3. Сохраняем
    print(f"\n[3/3] Сохранение результатов...")
    save_worst_csv(worst, OUTPUT_CSV)

    # Вывод в консоль
    print("\n" + "=" * 60)
    print(f"ТОП-{TOP_WORST_N} ХУДШИХ ДЕКЛАРАЦИЙ")
    print("=" * 60)

    for i, row in enumerate(worst, start=1):
        print(f"\n  {i}. {row['declaration_id']} (top1_score={row['top1_score']:.4f})")
        print(f"     Top-1 регуляция: {row['top1_regulation_id']}")
        print(f"     Описание: {row['description'][:150]}")


if __name__ == "__main__":
    main()