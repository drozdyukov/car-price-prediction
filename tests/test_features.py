import numpy as np
import pytest

from car_price.features import MODEL_FEATURES, prepare_features


def test_feature_order_and_category_encoding(cars):
    prepared = prepare_features(cars)
    assert prepared.columns.tolist() == MODEL_FEATURES
    np.testing.assert_array_equal(prepared.iloc[0], [5.59, 0, 6, 0, 0, 1])
    np.testing.assert_array_equal(prepared.iloc[1], [9.54, 1, 7, 1, 1, 0])


@pytest.mark.parametrize(
    "column,value",
    [
        ("Year", 2021),
        ("Year", 2014.5),
        ("Owner", -1),
        ("Owner", 0.5),
        ("Present_Price", 0),
        ("Present_Price", float("inf")),
        ("Fuel_Type", "Electric"),
        ("Transmission", "Mannual"),
        ("Seller_Type", "Unknown"),
        ("Present_Price", None),
    ],
)
def test_invalid_inputs_are_rejected(cars, column, value):
    cars[column] = value
    with pytest.raises(ValueError):
        prepare_features(cars)


def test_missing_column_is_rejected(cars):
    with pytest.raises(ValueError, match="Owner"):
        prepare_features(cars.drop(columns="Owner"))
