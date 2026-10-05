"""Метрики регрессии на отложенной выборке."""

from typing import Any

from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def regression_metrics(target: Any, predictions: Any) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(target, predictions)),
        "RMSE": float(root_mean_squared_error(target, predictions)),
        "R2": float(r2_score(target, predictions)),
    }
