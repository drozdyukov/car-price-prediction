"""Команды обучения и пакетного предсказания."""

import argparse
import logging
from pathlib import Path

import pandas as pd

from car_price.config import DEFAULT_CONFIG, load_config
from car_price.predict import load_model, predict_prices
from car_price.train import train


def train_main() -> None:
    parser = argparse.ArgumentParser(description="Обучить модель стоимости автомобиля")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="TOML-конфигурация")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        train(load_config(args.config))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Ошибка обучения: {error}\n")


def predict_main() -> None:
    parser = argparse.ArgumentParser(description="Рассчитать цены автомобилей из CSV")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="TOML-конфигурация")
    parser.add_argument("--input", type=Path, required=True, help="Входной CSV")
    parser.add_argument("--output", type=Path, required=True, help="CSV с прогнозами")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.exit(1, "Входной и выходной CSV должны быть разными файлами.\n")
    try:
        config = load_config(args.config)
        data = pd.read_csv(args.input)
        data["Predicted_Price"] = predict_prices(load_model(config.model_path), data)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(args.output, index=False)
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        parser.exit(1, f"Ошибка предсказания: {error}\n")
    print(f"Прогнозы сохранены: {args.output}")
