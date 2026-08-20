"""
Data Check Script
Проверяет входные JSONL-файлы (декларации и регуляции) на:
- пропуски в обязательных полях
- дубликаты ID
- некорректные типы

Результат:
- val_out/declarations_issues.jsonl
- val_out/regulations_issues.jsonl
- возвращает чистые данные (исходники не изменяются)
"""

import json
from pathlib import Path
from typing import Any

from config import PROJECT_ROOT, load_config

# ─── Загрузка конфига ───────────────────────────────────────────────────────

config = load_config()

# Обязательные поля для каждого типа
DECL_REQUIRED_FIELDS = ["declaration_id", "G31_1"]
REG_REQUIRED_FIELDS = ["regulation_id", "description"]


# ─── Утилиты ─────────────────────────────────────────────────────────────────

def read_jsonl(path: Path) -> list[dict]:
    """Читает JSONL-файл. Пропускает пустые строки."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [WARN] {path.name}:{line_num} — битый JSON, строка пропущена: {e}")
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Записывает список словарей в JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def find_duplicates(rows: list[dict], key: str) -> set[str]:
    """Возвращает множество ID, которые встречаются больше одного раза."""
    seen: dict[str, int] = {}
    for row in rows:
        val = row.get(key)
        if val is None:
            continue
        seen[val] = seen.get(val, 0) + 1
    return {val for val, count in seen.items() if count > 1}


def check_declarations(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Проверяет декларации.
    Возвращает (чистые, проблемные).
    """
    issues: list[dict] = []
    clean: list[dict] = []
    duplicate_ids = find_duplicates(rows, "declaration_id")

    for row in rows:
        decl_id = row.get("declaration_id")
        problems: list[str] = []

        # 1. Проверка дубликатов
        if decl_id in duplicate_ids:
            problems.append("duplicate_declaration_id")

        # 2. Проверка пропусков
        for field in DECL_REQUIRED_FIELDS:
            val = row.get(field)
            if val is None or (isinstance(val, str) and not val.strip()):
                problems.append(f"missing_{field}")

        # 3. Проверка типов
        if decl_id is not None and not isinstance(decl_id, str):
            problems.append("wrong_type_declaration_id")
        if row.get("G31_1") is not None and not isinstance(row.get("G31_1"), str):
            problems.append("wrong_type_G31_1")
        if "desc_extention" in row and row["desc_extention"] is not None and not isinstance(row["desc_extention"], str):
            problems.append("wrong_type_desc_extention")

        if problems:
            issues.append({
                "declaration_id": decl_id,
                "issues": problems,
                "row": row
            })
        else:
            clean.append(row)

    return clean, issues


def check_regulations(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Проверяет регуляции.
    Возвращает (чистые, проблемные).
    """
    issues: list[dict] = []
    clean: list[dict] = []
    duplicate_ids = find_duplicates(rows, "regulation_id")

    for row in rows:
        reg_id = row.get("regulation_id")
        problems: list[str] = []

        if reg_id in duplicate_ids:
            problems.append("duplicate_regulation_id")

        for field in REG_REQUIRED_FIELDS:
            val = row.get(field)
            if val is None or (isinstance(val, str) and not val.strip()):
                problems.append(f"missing_{field}")

        if reg_id is not None and not isinstance(reg_id, str):
            problems.append("wrong_type_regulation_id")
        if row.get("description") is not None and not isinstance(row.get("description"), str):
            problems.append("wrong_type_description")
        if "code" in row and row["code"] is not None and not isinstance(row["code"], str):
            problems.append("wrong_type_code")

        if problems:
            issues.append({
                "regulation_id": reg_id,
                "issues": problems,
                "row": row
            })
        else:
            clean.append(row)

    return clean, issues


# ─── Основной блок ───────────────────────────────────────────────────────────

def main(
    data_dir: Path | None = None,
    val_out_dir: Path | None = None,
) -> dict[str, list[dict]]:
    """
    Проверяет данные и возвращает чистые декларации и регуляции.

    Args:
        data_dir: Путь к папке с входными данными.
                  Если None — берётся из конфига.
        val_out_dir: Путь к папке для отчётов.
                     Если None — берётся из конфига.

    Returns:
        {"declarations": [...], "regulations": [...]}
    """

    # Определяем пути
    if data_dir is None:
        _data_dir = PROJECT_ROOT / config["paths"]["data_dir"]
    else:
        _data_dir = Path(data_dir)

    if val_out_dir is None:
        _val_out_dir = PROJECT_ROOT / config["paths"]["val_out_dir"]
    else:
        _val_out_dir = Path(val_out_dir)

    declarations_file = _data_dir / "declarations.jsonl"
    regulations_file = _data_dir / "regulations.jsonl"
    decl_issues_file = _val_out_dir / "declarations_issues.jsonl"
    reg_issues_file = _val_out_dir / "regulations_issues.jsonl"

    print("=" * 60)
    print("DATA CHECK")
    print("=" * 60)
    print(f"  Папка данных: {_data_dir}")
    print(f"  Папка отчётов: {_val_out_dir}")

    # 1. Читаем данные
    print(f"\n[1/4] Чтение данных...")
    if not declarations_file.exists():
        print(f"  [ERROR] Файл не найден: {declarations_file}")
        return {"declarations": [], "regulations": []}
    if not regulations_file.exists():
        print(f"  [ERROR] Файл не найден: {regulations_file}")
        return {"declarations": [], "regulations": []}

    declarations_raw = read_jsonl(declarations_file)
    regulations_raw = read_jsonl(regulations_file)

    print(f"  Деклараций загружено: {len(declarations_raw)}")
    print(f"  Регуляций загружено: {len(regulations_raw)}")

    # 2. Проверка
    print(f"\n[2/4] Проверка данных...")
    clean_decls, issue_decls = check_declarations(declarations_raw)
    clean_regs, issue_regs = check_regulations(regulations_raw)

    print(f"  Чистых деклараций: {len(clean_decls)}")
    print(f"  Проблемных деклараций: {len(issue_decls)}")
    print(f"  Чистых регуляций: {len(clean_regs)}")
    print(f"  Проблемных регуляций: {len(issue_regs)}")

    # 3. Сохраняем отчёты о проблемах
    print(f"\n[3/4] Сохранение отчётов...")

    decl_issues_short = [
        {"declaration_id": item["declaration_id"], "issues": item["issues"]}
        for item in issue_decls
    ]
    reg_issues_short = [
        {"regulation_id": item["regulation_id"], "issues": item["issues"]}
        for item in issue_regs
    ]

    write_jsonl(decl_issues_file, decl_issues_short)
    write_jsonl(reg_issues_file, reg_issues_short)

    print(f"  Отчёт по декларациям: {decl_issues_file}")
    print(f"  Отчёт по регуляциям: {reg_issues_file}")

    # 4. Итог
    print(f"\n[4/4] Готово.")
    print(f"  Чистые данные готовы к использованию в памяти.")
    print(f"  Исходные файлы НЕ изменены.")

    return {
        "declarations": clean_decls,
        "regulations": clean_regs
    }


if __name__ == "__main__":
    result = main()