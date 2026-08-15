"""Leakage-safe validation, feature engineering, and preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.config import (
    BINARY_FEATURES,
    CONTINUOUS_FEATURES,
    ENGINEERED_CATEGORICAL,
    ENGINEERED_NUMERIC,
    ORDINAL_FEATURES,
    RAW_FEATURES,
    VALID_RANGES,
)


class DomainFeatureEngineer(BaseEstimator, TransformerMixin):
    """Validate input ranges and add deterministic, domain-informed features.

    Invalid values become missing so that imputers fitted only on the training
    fold handle them. Extreme but valid BMI values are retained; RobustScaler
    limits their influence without erasing clinically important observations.
    """

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = np.asarray(RAW_FEATURES, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = pd.DataFrame(X).copy()
        missing = [name for name in RAW_FEATURES if name not in frame.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        frame = frame[RAW_FEATURES].apply(pd.to_numeric, errors="coerce")

        for column, (lower, upper) in VALID_RANGES.items():
            frame.loc[~frame[column].between(lower, upper), column] = np.nan

        frame["HealthBurden"] = frame["MentHlth"] + frame["PhysHlth"]
        frame["CardioBurden"] = frame[
            ["HighBP", "HighChol", "Stroke", "HeartDiseaseorAttack"]
        ].sum(axis=1, min_count=1)
        frame["LifestyleScore"] = (
            frame["PhysActivity"] + frame["Fruits"] + frame["Veggies"]
            + (1 - frame["Smoker"]) + (1 - frame["HvyAlcoholConsump"])
        )
        frame["AccessBarrier"] = np.maximum(
            1 - frame["AnyHealthcare"], frame["NoDocbcCost"]
        )
        frame["BMIAgeRisk"] = frame["BMI"] * frame["Age"]
        frame["BMIGroup"] = pd.cut(
            frame["BMI"],
            bins=[-np.inf, 18.5, 25, 30, 35, 40, np.inf],
            labels=["underweight", "healthy", "overweight", "obesity_1", "obesity_2", "obesity_3"],
        ).astype(object)
        return frame

    def get_feature_names_out(self, input_features=None):
        return np.asarray(
            RAW_FEATURES + ENGINEERED_NUMERIC + ENGINEERED_CATEGORICAL,
            dtype=object,
        )


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", RobustScaler()),
        ]
    )
    ordinal = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent", add_indicator=True)),
            ("scale", RobustScaler()),
        ]
    )
    encoded = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OneHotEncoder(
                    drop="if_binary", handle_unknown="ignore", sparse_output=False
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("continuous", numeric, CONTINUOUS_FEATURES + ENGINEERED_NUMERIC),
            ("ordinal", ordinal, ORDINAL_FEATURES),
            ("binary", encoded, BINARY_FEATURES),
            ("bmi_group", encoded, ENGINEERED_CATEGORICAL),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def invalid_value_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Return counts outside the published/source-coded ranges."""
    return {
        column: int((~frame[column].between(lower, upper)).sum())
        for column, (lower, upper) in VALID_RANGES.items()
    }

