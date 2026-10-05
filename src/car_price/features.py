"""Единая проверка и обработка входных признаков для обучения и предсказаний."""

import numpy as np
import pandas as pd

RAW_FEATURES = ["Present_Price", "Year", "Owner", "Fuel_Type", "Seller_Type", "Transmission"]
MODEL_FEATURES = [
    "Present_Price",
    "Owner",
    "Years_old",
    "Fuel_Type_Diesel",
    "Seller_Type_Individual",
    "Transmission_Manual",
]
CATEGORIES = {
    "Fuel_Type": {"Petrol", "Diesel", "CNG"},
    "Seller_Type": {"Dealer", "Individual"},
    "Transmission": {"Manual", "Automatic"},
}


def validate_features(data: pd.DataFrame, reference_year: int) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Входные данные должны быть таблицей pandas DataFrame.")
    if not data.columns.is_unique:
        raise ValueError("Названия столбцов должны быть уникальными.")
    missing = sorted(set(RAW_FEATURES) - set(data.columns))
    if missing:
        raise ValueError(f"Отсутствуют обязательные столбцы: {', '.join(missing)}.")
    result = data.loc[:, RAW_FEATURES].copy()
    if result.empty or result.isna().any().any():
        raise ValueError("Данные не должны быть пустыми или содержать пропуски.")
    for column in ("Present_Price", "Year", "Owner"):
        result[column] = pd.to_numeric(result[column], errors="raise")
        if not np.isfinite(result[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Столбец {column} содержит бесконечные значения.")
    if (result["Present_Price"] <= 0).any():
        raise ValueError("Цена нового автомобиля должна быть больше нуля.")
    if not result["Year"].between(1900, reference_year).all():
        raise ValueError(f"Год должен находиться между 1900 и {reference_year}.")
    if (result["Year"] % 1 != 0).any():
        raise ValueError("Год должен быть целым числом.")
    if ((result["Owner"] < 0) | (result["Owner"] % 1 != 0)).any():
        raise ValueError("Количество владельцев должно быть целым неотрицательным числом.")
    for column, categories in CATEGORIES.items():
        if not result[column].isin(categories).all():
            raise ValueError(f"Недопустимые значения в {column}: {sorted(categories)}.")
    return result


def prepare_features(data: pd.DataFrame, reference_year: int = 2020) -> pd.DataFrame:
    raw = validate_features(data, reference_year)
    features = pd.DataFrame(index=raw.index)
    features["Present_Price"] = raw["Present_Price"]
    features["Owner"] = raw["Owner"]
    features["Years_old"] = reference_year - raw["Year"]
    features["Fuel_Type_Diesel"] = (raw["Fuel_Type"] == "Diesel").astype(int)
    features["Seller_Type_Individual"] = (raw["Seller_Type"] == "Individual").astype(int)
    features["Transmission_Manual"] = (raw["Transmission"] == "Manual").astype(int)
    return features.loc[:, MODEL_FEATURES].astype(float)
