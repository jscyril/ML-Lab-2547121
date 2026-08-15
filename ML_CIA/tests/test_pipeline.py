import numpy as np
import pandas as pd

from src.config import RAW_FEATURES
from src.modeling import baseline_pipeline
from src.preprocessing import DomainFeatureEngineer


def sample_frame():
    base = {
        "HighBP": 1, "HighChol": 1, "CholCheck": 1, "BMI": 34,
        "Smoker": 0, "Stroke": 0, "HeartDiseaseorAttack": 0,
        "PhysActivity": 0, "Fruits": 1, "Veggies": 1,
        "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0,
        "GenHlth": 3, "MentHlth": 2, "PhysHlth": 5, "DiffWalk": 0,
        "Sex": 0, "Age": 8, "Education": 5, "Income": 5,
    }
    return pd.DataFrame([base, {**base, "HighBP": 0, "BMI": 23, "Age": 3}])


def test_feature_engineering_preserves_rows_and_adds_features():
    result = DomainFeatureEngineer().fit_transform(sample_frame())
    assert len(result) == 2
    assert {"BMIGroup", "CardioBurden", "LifestyleScore", "BMIAgeRisk"} <= set(result)
    assert result.loc[0, "CardioBurden"] == 2


def test_invalid_values_become_missing_for_training_fitted_imputation():
    frame = sample_frame()
    frame.loc[0, "MentHlth"] = 90
    result = DomainFeatureEngineer().fit_transform(frame)
    assert np.isnan(result.loc[0, "MentHlth"])


def test_pipeline_predicts_probabilities():
    X = pd.concat([sample_frame()] * 6, ignore_index=True)
    y = np.array([1, 0] * 6)
    model = baseline_pipeline().fit(X[RAW_FEATURES], y)
    probability = model.predict_proba(X[RAW_FEATURES])[:, 1]
    assert probability.shape == (12,)
    assert np.all((0 <= probability) & (probability <= 1))

