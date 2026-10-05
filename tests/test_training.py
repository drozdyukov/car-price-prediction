import json
from dataclasses import replace

import numpy as np
import pytest

from car_price.config import load_config
from car_price.data import load_dataset
from car_price.predict import load_model, predict_prices
from car_price.train import train


def test_pipeline_serialization_and_reproducibility(small_config, cars):
    metrics = train(small_config)
    model = load_model(small_config.model_path)
    prediction = predict_prices(model, cars)
    assert prediction.shape == (2,)
    assert np.isfinite(prediction).all()
    assert (prediction >= 0).all()
    assert model.named_steps["regressor"].n_features_in_ == 6
    saved_metrics = json.loads(small_config.metrics_path.read_text())
    assert saved_metrics == metrics
    assert metrics["train_rows"] + metrics["test_rows"] == 301
    # Повторное обучение на тех же данных и конфигурации воспроизводимо.
    train(small_config)
    np.testing.assert_array_equal(
        prediction, predict_prices(load_model(small_config.model_path), cars)
    )


def test_missing_target_is_rejected(tmp_path, small_config, cars):
    path = tmp_path / "cars.csv"
    cars.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Selling_Price"):
        load_dataset(path, small_config.reference_year)


def test_config_paths_do_not_depend_on_working_directory(tmp_path, monkeypatch):
    original = load_config()
    monkeypatch.chdir(tmp_path)
    assert load_config().data_path == original.data_path


def test_too_small_holdout_is_rejected(small_config):
    with pytest.raises(ValueError, match="минимум две строки"):
        train(replace(small_config, test_size=0.001))
