"""Model definitions and tuning spaces."""

from __future__ import annotations

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.config import SEED
from src.preprocessing import DomainFeatureEngineer, build_preprocessor


def make_pipeline(model) -> Pipeline:
    return Pipeline(
        [
            ("features", DomainFeatureEngineer()),
            ("preprocess", build_preprocessor()),
            ("model", model),
        ]
    )


def baseline_pipeline() -> Pipeline:
    return make_pipeline(
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1500,
            random_state=SEED,
        )
    )


def random_forest_pipeline() -> Pipeline:
    return make_pipeline(
        RandomForestClassifier(
            n_estimators=160,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=1,
        )
    )


def adaboost_pipeline() -> Pipeline:
    return make_pipeline(
        AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=SEED),
            n_estimators=120,
            learning_rate=0.1,
            random_state=SEED,
        )
    )


def voting_pipeline(rf_params: dict, ada_params: dict) -> Pipeline:
    """Soft vote across heterogeneous tuned model families.

    Voting has no learned meta-model, so it cannot train on in-sample base
    predictions. Base hyperparameters come only from cross-validation on the
    training partition.
    """
    logistic = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=1500, random_state=SEED
    )
    forest = RandomForestClassifier(
        n_estimators=rf_params["model__n_estimators"],
        max_depth=rf_params["model__max_depth"],
        min_samples_leaf=rf_params["model__min_samples_leaf"],
        max_features=rf_params["model__max_features"],
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=1,
    )
    boost = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(
            max_depth=ada_params["model__estimator__max_depth"], random_state=SEED
        ),
        n_estimators=ada_params["model__n_estimators"],
        learning_rate=ada_params["model__learning_rate"],
        random_state=SEED,
    )
    return make_pipeline(
        VotingClassifier(
            estimators=[("logistic", logistic), ("forest", forest), ("adaboost", boost)],
            voting="soft",
            weights=[1, 2, 1],
            n_jobs=1,
            flatten_transform=True,
        )
    )


RF_GRID = [
    {"model__n_estimators": [120], "model__max_depth": [10], "model__min_samples_leaf": [3], "model__max_features": ["sqrt"]},
    {"model__n_estimators": [180], "model__max_depth": [14], "model__min_samples_leaf": [2], "model__max_features": ["sqrt"]},
    {"model__n_estimators": [180], "model__max_depth": [None], "model__min_samples_leaf": [5], "model__max_features": [0.7]},
    {"model__n_estimators": [240], "model__max_depth": [18], "model__min_samples_leaf": [5], "model__max_features": [0.7]},
]

ADA_GRID = [
    {"model__n_estimators": [80], "model__learning_rate": [0.05], "model__estimator__max_depth": [1]},
    {"model__n_estimators": [140], "model__learning_rate": [0.05], "model__estimator__max_depth": [2]},
    {"model__n_estimators": [100], "model__learning_rate": [0.10], "model__estimator__max_depth": [2]},
    {"model__n_estimators": [160], "model__learning_rate": [0.10], "model__estimator__max_depth": [1]},
]

