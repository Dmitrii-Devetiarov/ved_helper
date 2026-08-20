"""Загрузка конфигурации проекта."""

import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = PROJECT_ROOT / "config.yaml"


def load_config(path: Path = CONFIG_FILE) -> dict:
    """Читает YAML-конфиг."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)