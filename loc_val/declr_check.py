"""
Быстрая проверка результатов для конкретной декларации.
Выводит описание декларации и список регуляций из results.csv
в порядке убывания скора.
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

DECLARATIONS_FILE = DATA_DIR / "declarations.jsonl"
REGULATIONS_FILE = DATA_DIR / "regulations.jsonl"
RESULTS_FILE = OUT_DIR / "results.csv"


# ─── Утилиты чтения ─────────────────────────────────────────────────────────

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
    Возвращает словарь: {declaration_id: [ {rank, regulation_id, score}, ... ]}
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


# ─── Поиск данных ───────────────────────────────────────────────────────────

def find_declaration(declarations: list[dict], decl_id: str) -> dict | None:
    """Ищет декларацию по ID."""
    for d in declarations:
        if d["declaration_id"] == decl_id:
            return d
    return None


def find_regulation(regulations: list[dict], reg_id: str) -> dict | None:
    """Ищет регуляцию по ID."""
    for r in regulations:
        if r["regulation_id"] == reg_id:
            return r
    return None


# ─── Вывод ──────────────────────────────────────────────────────────────────

def print_declaration(decl_id: str, declarations: list[dict], regulations: list[dict], results: dict[str, list[dict]]) -> None:
    """Выводит информацию по декларации."""
    print("=" * 80)
    print(f"ДЕКЛАРАЦИЯ: {decl_id}")
    print("=" * 80)

    # 1. Декларация
    decl = find_declaration(declarations, decl_id)
    if decl is None:
        print(f"  [ERROR] Декларация {decl_id} не найдена в исходных данных.")
        return

    print("\n--- ОПИСАНИЕ ДЕКЛАРАЦИИ ---")
    print(f"  G31_1: {decl.get('G31_1', '')}")
    if decl.get("desc_extention"):
        print(f"  desc_extention: {decl['desc_extention']}")
    print(f"  Страна (G34): {decl.get('G34', 'N/A')}")

    # 2. Результаты
    if decl_id not in results:
        print(f"\n  [WARN] Для декларации {decl_id} нет результатов в {RESULTS_FILE.name}")
        return

    top_regs = results[decl_id]
    top_regs.sort(key=lambda x: x["rank"])

    print(f"\n--- ТОП-{len(top_regs)} РЕГУЛЯЦИИ ---")
    for item in top_regs:
        reg = find_regulation(regulations, item["regulation_id"])
        if reg is None:
            print(f"\n  Ранг {item['rank']}: {item['regulation_id']} (score={item['score']:.4f})")
            print(f"    [WARN] Регуляция не найдена в исходных данных.")
            continue

        print(f"\n  Ранг {item['rank']}: {item['regulation_id']} (score={item['score']:.4f})")
        print(f"    Код ТН ВЭД: {reg.get('code', 'N/A')}")
        print(f"    Описание: {reg.get('description', '')[:300]}")
        if reg.get("explanation"):
            print(f"    Пояснение: {reg['explanation'][:200]}")


# ─── Основная функция ───────────────────────────────────────────────────────

def main() -> None:
    print("Быстрая проверка результата по декларации")

    # Читаем данные
    declarations = read_jsonl(DECLARATIONS_FILE)
    regulations = read_jsonl(REGULATIONS_FILE)
    results = read_results_csv(RESULTS_FILE)

    decl_ids = [d["declaration_id"] for d in declarations]

    print(f"Доступные ID деклараций ({len(decl_ids)}):")
    for did in decl_ids:
        print(f"  {did}")

    # Запрос ID
    while True:
        print()
        decl_id = input("Введите declaration_id (или 'exit' для выхода): ").strip()

        if decl_id.lower() == "exit":
            print("Выход.")
            break

        if decl_id not in decl_ids:
            print(f"  [WARN] Декларация {decl_id} не найдена. Попробуйте ещё раз.")
            continue

        print_declaration(decl_id, declarations, regulations, results)

if __name__ == "__main__":
    main()