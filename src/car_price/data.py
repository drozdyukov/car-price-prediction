"""Загрузка датасета с проверкой признаков и целевой цены."""

from pathlib import Path

import numpy as np
import pandas as pd

from car_price.features import validate_features


def load_dataset(path: Path, reference_year: int) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(path)
    features = validate_features(data, reference_year)
    if "Selling_Price" not in data:
        raise ValueError("В датасете отсутствует целевой столбец Selling_Price.")
    target = pd.to_numeric(data["Selling_Price"], errors="raise")
    if not np.isfinite(target.to_numpy(dtype=float)).all() or (target < 0).any():
        raise ValueError("Цена продажи должна быть конечной и неотрицательной.")
    if len(data) < 10:
        raise ValueError("Для обучения нужно не менее 10 строк.")
    return features, target
