"""Pipeline сохраняет подготовку признаков вместе с регрессором."""

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from car_price.config import Config
from car_price.features import prepare_features


def build_pipeline(config: Config) -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                FunctionTransformer(
                    prepare_features,
                    kw_args={"reference_year": config.reference_year},
                ),
            ),
            (
                "regressor",
                RandomForestRegressor(
                    **config.model_params,
                    random_state=config.random_state,
                ),
            ),
        ]
    )
