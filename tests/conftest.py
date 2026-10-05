from dataclasses import replace

import pandas as pd
import pytest

from car_price.config import load_config


@pytest.fixture
def cars():
    return pd.DataFrame(
        [
            {
                "Present_Price": 5.59,
                "Year": 2014,
                "Owner": 0,
                "Fuel_Type": "Petrol",
                "Seller_Type": "Dealer",
                "Transmission": "Manual",
            },
            {
                "Present_Price": 9.54,
                "Year": 2013,
                "Owner": 1,
                "Fuel_Type": "Diesel",
                "Seller_Type": "Individual",
                "Transmission": "Automatic",
            },
        ]
    )


@pytest.fixture
def small_config(tmp_path):
    config = load_config()
    return replace(
        config,
        model_path=tmp_path / "model.joblib",
        metrics_path=tmp_path / "metrics.json",
        model_params={**config.model_params, "n_estimators": 20, "n_jobs": 1},
    )
