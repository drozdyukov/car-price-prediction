"""Загрузка обученного Pipeline и расчёт цены."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def load_model(path: Path) -> Pipeline:
    model = joblib.load(path)
    if not isinstance(model, Pipeline) or "features" not in model.named_steps:
        raise ValueError("Некорректный формат модели. Выполните car-train для переобучения.")
    return model


def predict_prices(model: Pipeline, data: pd.DataFrame) -> np.ndarray:
    predictions = np.asarray(model.predict(data), dtype=float)
    if not np.isfinite(predictions).all() or (predictions < 0).any():
        raise ValueError("Модель вернула некорректные цены.")
    return predictions
