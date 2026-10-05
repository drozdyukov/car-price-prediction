"""Чтение конфигурации и разрешение путей относительно TOML-файла."""

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.toml"


@dataclass(frozen=True)
class Config:
    data_path: Path
    model_path: Path
    metrics_path: Path
    reference_year: int
    test_size: float
    random_state: int
    model_params: dict[str, Any]


def load_config(path: str | Path = DEFAULT_CONFIG) -> Config:
    config_path = Path(path).resolve()
    with config_path.open("rb") as source:
        values = tomllib.load(source)
    base = config_path.parent
    config = Config(
        data_path=(base / values["data"]["path"]).resolve(),
        model_path=(base / values["artifacts"]["model_path"]).resolve(),
        metrics_path=(base / values["artifacts"]["metrics_path"]).resolve(),
        reference_year=int(values["data"]["reference_year"]),
        test_size=float(values["training"]["test_size"]),
        random_state=int(values["training"]["random_state"]),
        model_params=dict(values["model"]),
    )
    if not 0 < config.test_size < 1:
        raise ValueError("Доля тестовой выборки должна находиться между 0 и 1.")
    if config.reference_year < 1900:
        raise ValueError("Базовый год должен быть не меньше 1900.")
    if config.model_path == config.metrics_path:
        raise ValueError("Модель и метрики должны сохраняться в разные файлы.")
    if config.data_path in (config.model_path, config.metrics_path):
        raise ValueError("Артефакты не должны перезаписывать исходные данные.")
    return config
