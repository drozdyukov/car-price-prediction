"""Воспроизводимое обучение и сохранение артефактов."""

import json
import logging
from typing import Any

import joblib
import sklearn
from sklearn.model_selection import train_test_split

from car_price.config import Config
from car_price.data import load_dataset
from car_price.evaluate import regression_metrics
from car_price.model import build_pipeline

LOGGER = logging.getLogger(__name__)


def train(config: Config) -> dict[str, Any]:
    features, target = load_dataset(config.data_path, config.reference_year)
    LOGGER.info("Загружено %s строк из %s", len(features), config.data_path)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=config.test_size,
        random_state=config.random_state,
    )
    if len(x_test) < 2:
        raise ValueError("Для расчёта R² тестовая выборка должна содержать минимум две строки.")
    pipeline = build_pipeline(config)
    pipeline.fit(x_train, y_train)
    metrics: dict[str, Any] = regression_metrics(y_test, pipeline.predict(x_test))
    metrics.update(
        {
            "train_rows": len(x_train),
            "test_rows": len(x_test),
            "reference_year": config.reference_year,
            "random_state": config.random_state,
            "sklearn_version": sklearn.__version__,
            "model_params": config.model_params,
        }
    )
    config.model_path.parent.mkdir(parents=True, exist_ok=True)
    config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    # Сначала полностью записать файлы, затем заменить предыдущие артефакты.
    temporary_model = config.model_path.with_suffix(".joblib.tmp")
    temporary_metrics = config.metrics_path.with_suffix(".json.tmp")
    joblib.dump(pipeline, temporary_model)
    temporary_metrics.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_model.replace(config.model_path)
    temporary_metrics.replace(config.metrics_path)
    LOGGER.info("Модель сохранена: %s", config.model_path)
    LOGGER.info("MAE=%.3f; RMSE=%.3f; R²=%.3f", metrics["MAE"], metrics["RMSE"], metrics["R2"])
    return metrics
