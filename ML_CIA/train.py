#!/usr/bin/env python3
"""Train, tune, compare, explain, and persist the Mission Health solution."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import sklearn
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from src.config import (
    ARTIFACT_DIR,
    DATA_PATH,
    FIGURE_DIR,
    RAW_FEATURES,
    SEED,
    TABLE_DIR,
    TARGET,
)
from src.evaluation import choose_f2_threshold, metrics_at_threshold, subgroup_metrics
from src.modeling import (
    ADA_GRID,
    RF_GRID,
    adaboost_pipeline,
    baseline_pipeline,
    random_forest_pipeline,
    voting_pipeline,
)
from src.preprocessing import invalid_value_counts

sns.set_theme(style="whitegrid", context="notebook")


DEMO_RECORD = {
    "HighBP": 1, "HighChol": 1, "CholCheck": 1, "BMI": 34,
    "Smoker": 0, "Stroke": 0, "HeartDiseaseorAttack": 0,
    "PhysActivity": 0, "Fruits": 1, "Veggies": 1,
    "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0,
    "GenHlth": 3, "MentHlth": 2, "PhysHlth": 5, "DiffWalk": 0,
    "Sex": 0, "Age": 8, "Education": 5, "Income": 5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a stratified 60,000-row development sample for a faster smoke run.",
    )
    parser.add_argument(
        "--skip-shap", action="store_true", help="Skip model-agnostic SHAP generation."
    )
    return parser.parse_args()


def ensure_directories() -> None:
    for directory in (ARTIFACT_DIR, FIGURE_DIR, TABLE_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_audit(path: Path, quick: bool = False):
    raw = pd.read_csv(path)
    if TARGET not in raw or set(RAW_FEATURES) - set(raw.columns):
        raise ValueError("Dataset columns do not match the expected UCI binary file schema.")
    raw = raw[[TARGET] + RAW_FEATURES]
    audit = {
        "dataset": str(path),
        "sha256": file_sha256(path),
        "raw_rows": len(raw),
        "columns": len(raw.columns),
        "missing_cells": int(raw.isna().sum().sum()),
        "exact_duplicate_rows": int(raw.duplicated().sum()),
        "invalid_values_by_feature": invalid_value_counts(raw),
        "target_counts_before_deduplication": {
            str(int(key)): int(value) for key, value in raw[TARGET].value_counts().items()
        },
    }
    # Without respondent identifiers, exact copies can cross partitions. Keeping
    # one copy is the conservative anti-leakage choice; the limitation is reported.
    clean = raw.drop_duplicates().reset_index(drop=True)
    audit["rows_after_exact_deduplication"] = len(clean)
    audit["rows_removed"] = len(raw) - len(clean)
    audit["positive_rate_after_deduplication"] = float(clean[TARGET].mean())

    if quick and len(clean) > 60_000:
        sample, _ = train_test_split(
            clean, train_size=60_000, stratify=clean[TARGET], random_state=SEED
        )
        clean = sample.reset_index(drop=True)
        audit["quick_mode_rows"] = len(clean)
    return clean, audit


def split_data(frame: pd.DataFrame):
    X = frame[RAW_FEATURES]
    y = frame[TARGET].astype(int)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )
    return X_train, X_validation, X_test, y_train, y_validation, y_test


def save_eda(frame: pd.DataFrame) -> None:
    class_counts = frame[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(["No diabetes", "Diabetes"], class_counts.values, color=["#3B82F6", "#EF4444"])
    ax.bar_label(bars, labels=[f"{value:,}" for value in class_counts.values], padding=4)
    ax.set(title="Class distribution after exact-deduplication", ylabel="Respondents")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "01_class_balance.png", dpi=180)
    plt.close(fig)

    working = frame.copy()
    working["BMI group"] = pd.cut(
        working["BMI"], [0, 18.5, 25, 30, 35, 40, np.inf],
        labels=["Under", "Healthy", "Over", "Obesity I", "Obesity II", "Obesity III"],
    )
    panels = [("Age", "Age category"), ("GenHlth", "General health (1–5)"), ("BMI group", "BMI group")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (column, title) in zip(axes, panels):
        rates = working.groupby(column, observed=True)[TARGET].agg(["mean", "count"])
        ax.plot(range(len(rates)), rates["mean"], marker="o", color="#7C3AED")
        ax.set_xticks(range(len(rates)), rates.index.astype(str), rotation=35 if column == "BMI group" else 0)
        ax.set(title=title, ylabel="Observed diabetes proportion", ylim=(0, max(0.5, rates["mean"].max() * 1.1)))
    fig.suptitle("EDA: observed outcome rates across clinically relevant factors")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "02_eda_risk_patterns.png", dpi=180)
    plt.close(fig)

    selected = ["BMI", "Age", "GenHlth", "HighBP", "HighChol", "PhysHlth", "DiffWalk", TARGET]
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(frame[selected].corr(method="spearman"), cmap="vlag", center=0, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Spearman correlation (association, not causation)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "03_correlation_heatmap.png", dpi=180)
    plt.close(fig)


def tune_model(name, estimator, grid, X_train, y_train, cv):
    print(f"Tuning {name} ...", flush=True)
    search = GridSearchCV(
        estimator,
        grid,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
        verbose=1,
    )
    fit_params = {}
    if name == "AdaBoost":
        fit_params["model__sample_weight"] = compute_sample_weight("balanced", y_train)
    started = time.perf_counter()
    search.fit(X_train, y_train, **fit_params)
    elapsed = time.perf_counter() - started
    result = pd.DataFrame(search.cv_results_)
    result.insert(0, "model", name)
    result.to_csv(TABLE_DIR / f"cv_{name.lower()}.csv", index=False)
    print(f"{name}: best CV PR-AUC={search.best_score_:.4f} in {elapsed:.1f}s")
    return search.best_estimator_, search.best_params_, float(search.best_score_), elapsed


def predict_scores(models: dict, X):
    return {name: model.predict_proba(X)[:, 1] for name, model in models.items()}


def save_model_plots(y_test, probabilities: dict, comparison: pd.DataFrame, best_name: str, threshold: float):
    order = comparison.sort_values("average_precision", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    y_pos = np.arange(len(order))
    ax.barh(y_pos - 0.18, order["average_precision"], height=0.35, label="PR-AUC", color="#7C3AED")
    ax.barh(y_pos + 0.18, order["roc_auc"], height=0.35, label="ROC-AUC", color="#0891B2")
    ax.set_yticks(y_pos, order["model"])
    ax.set(xlim=(0, 1), xlabel="Score", title="Untouched test-set model comparison (threshold-independent)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "04_model_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, probability in probabilities.items():
        RocCurveDisplay.from_predictions(y_test, probability, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_title("ROC curves — untouched test set")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "05_roc_curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, probability in probabilities.items():
        PrecisionRecallDisplay.from_predictions(y_test, probability, name=name, ax=ax)
    ax.axhline(np.mean(y_test), linestyle="--", color="gray", label="Prevalence")
    ax.set_title("Precision–recall curves — untouched test set")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "06_precision_recall_curves.png", dpi=180)
    plt.close(fig)

    prediction = (probabilities[best_name] >= threshold).astype(int)
    matrix = confusion_matrix(y_test, prediction, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.8, 5))
    sns.heatmap(matrix, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax)
    ax.set(xlabel="Predicted", ylabel="Actual", title=f"{best_name} confusion matrix at threshold {threshold:.3f}")
    ax.set_xticklabels(["No diabetes", "Diabetes"])
    ax.set_yticklabels(["No diabetes", "Diabetes"], rotation=0)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "07_best_confusion_matrix.png", dpi=180)
    plt.close(fig)


def fairness_audit(X_test, y_test, probability, threshold):
    age_group = pd.cut(
        X_test["Age"], bins=[0, 5, 9, 13], labels=["18–44", "45–64", "65+"], include_lowest=True
    )
    income_group = pd.cut(
        X_test["Income"], bins=[0, 4, 8], labels=["Lower income codes 1–4", "Higher income codes 5–8"]
    )
    audits = []
    for attribute, groups in [
        ("Sex", X_test["Sex"].map({0: "Female", 1: "Male"})),
        ("Age band", age_group),
        ("Income band", income_group),
    ]:
        result = subgroup_metrics(y_test, probability, groups, threshold)
        result.insert(0, "attribute", attribute)
        audits.append(result)
    fairness = pd.concat(audits, ignore_index=True)
    fairness.to_csv(TABLE_DIR / "fairness_subgroups.csv", index=False)

    plot_data = fairness[fairness["attribute"].isin(["Sex", "Age band"])].copy()
    labels = plot_data["attribute"] + ": " + plot_data["group"]
    fig, ax = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(plot_data))
    ax.bar(positions - 0.18, plot_data["recall_tpr"], width=0.36, label="Recall / TPR", color="#2563EB")
    ax.bar(positions + 0.18, plot_data["false_positive_rate"], width=0.36, label="False-positive rate", color="#F97316")
    ax.set_xticks(positions, labels, rotation=25, ha="right")
    ax.set(ylim=(0, 1), ylabel="Rate", title="Subgroup audit (descriptive; not proof of fairness)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "08_fairness_audit.png", dpi=180)
    plt.close(fig)
    return fairness


def generate_explanations(best_model, X_train, X_test, seed=SEED):
    rng = np.random.default_rng(seed)
    background = X_train.iloc[rng.choice(len(X_train), size=min(40, len(X_train)), replace=False)].copy()
    explain_rows = X_test.iloc[rng.choice(len(X_test), size=min(50, len(X_test)), replace=False)].copy()
    demo = pd.DataFrame([DEMO_RECORD], columns=RAW_FEATURES)

    def predict_positive(values):
        frame = pd.DataFrame(values, columns=RAW_FEATURES)
        return best_model.predict_proba(frame)[:, 1]

    masker = shap.maskers.Independent(background, max_samples=len(background))
    explainer = shap.Explainer(
        predict_positive, masker, algorithm="permutation", feature_names=RAW_FEATURES
    )
    values = explainer(
        explain_rows,
        max_evals=2 * len(RAW_FEATURES) + 1,
        batch_size=128,
        silent=True,
    )
    local = explainer(
        demo,
        max_evals=2 * len(RAW_FEATURES) + 1,
        batch_size=128,
        silent=True,
    )

    global_table = pd.DataFrame(
        {"feature": RAW_FEATURES, "mean_absolute_shap": np.abs(values.values).mean(axis=0)}
    ).sort_values("mean_absolute_shap", ascending=False)
    global_table.to_csv(TABLE_DIR / "shap_global_importance.csv", index=False)
    local_table = pd.DataFrame(
        {
            "feature": RAW_FEATURES,
            "value": demo.iloc[0].values,
            "shap_contribution": local.values[0],
        }
    ).sort_values("shap_contribution", key=np.abs, ascending=False)
    local_table.to_csv(TABLE_DIR / "shap_demo_local.csv", index=False)

    plt.figure(figsize=(9, 6))
    shap.plots.bar(values, max_display=12, show=False)
    plt.title("Global SHAP importance — mean absolute contribution")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "09_shap_global.png", dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 6))
    shap.plots.waterfall(local[0], max_display=12, show=False)
    plt.title("Local SHAP explanation — synthetic demo respondent")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "10_shap_local_demo.png", dpi=180, bbox_inches="tight")
    plt.close()
    return global_table, local_table


def main() -> None:
    args = parse_args()
    ensure_directories()
    started = time.perf_counter()
    frame, audit = load_and_audit(args.data, args.quick)
    save_eda(frame)
    X_train, X_validation, X_test, y_train, y_validation, y_test = split_data(frame)
    audit["split"] = {
        "strategy": "stratified 70/15/15, random_state=42",
        "train_rows": len(X_train),
        "validation_rows": len(X_validation),
        "test_rows": len(X_test),
        "train_positive_rate": float(y_train.mean()),
        "validation_positive_rate": float(y_validation.mean()),
        "test_positive_rate": float(y_test.mean()),
    }
    (TABLE_DIR / "data_audit.json").write_text(json.dumps(audit, indent=2))

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    models = {}
    tuning = {}

    print("Fitting logistic-regression baseline ...", flush=True)
    baseline = baseline_pipeline().fit(X_train, y_train)
    models["Logistic baseline"] = baseline

    forest, rf_params, rf_cv, rf_seconds = tune_model(
        "RandomForest", random_forest_pipeline(), RF_GRID, X_train, y_train, cv
    )
    models["Random Forest"] = forest
    tuning["Random Forest"] = {"best_params": rf_params, "cv_average_precision": rf_cv, "seconds": rf_seconds}

    boost, ada_params, ada_cv, ada_seconds = tune_model(
        "AdaBoost", adaboost_pipeline(), ADA_GRID, X_train, y_train, cv
    )
    models["AdaBoost"] = boost
    tuning["AdaBoost"] = {"best_params": ada_params, "cv_average_precision": ada_cv, "seconds": ada_seconds}

    print("Fitting heterogeneous soft-voting ensemble ...", flush=True)
    vote = voting_pipeline(rf_params, ada_params)
    vote.fit(X_train, y_train, model__sample_weight=compute_sample_weight("balanced", y_train))
    models["Soft Voting"] = vote

    validation_probabilities = predict_scores(models, X_validation)
    validation_rows = []
    for name, probability in validation_probabilities.items():
        validation_rows.append({"model": name, **metrics_at_threshold(y_validation, probability)})
    validation_comparison = pd.DataFrame(validation_rows).sort_values("average_precision", ascending=False)
    validation_comparison.to_csv(TABLE_DIR / "validation_model_comparison.csv", index=False)
    best_name = str(validation_comparison.iloc[0]["model"])
    best_model = models[best_name]
    best_threshold, validation_f2 = choose_f2_threshold(
        y_validation, validation_probabilities[best_name]
    )

    test_probabilities = predict_scores(models, X_test)
    test_rows = []
    for name, probability in test_probabilities.items():
        test_rows.append({"model": name, **metrics_at_threshold(y_test, probability)})
    comparison = pd.DataFrame(test_rows).sort_values("average_precision", ascending=False)
    comparison.to_csv(TABLE_DIR / "test_model_comparison.csv", index=False)
    operating_metrics = metrics_at_threshold(
        y_test, test_probabilities[best_name], best_threshold
    )
    pd.DataFrame([{"model": best_name, **operating_metrics}]).to_csv(
        TABLE_DIR / "best_model_operating_point.csv", index=False
    )
    save_model_plots(y_test, test_probabilities, comparison, best_name, best_threshold)
    fairness_audit(X_test, y_test, test_probabilities[best_name], best_threshold)

    if not args.skip_shap:
        print("Generating model-agnostic SHAP explanations ...", flush=True)
        generate_explanations(best_model, X_train, X_test)

    demo = pd.DataFrame([DEMO_RECORD], columns=RAW_FEATURES)
    demo_score = float(best_model.predict_proba(demo)[0, 1])
    demo_result = {
        "input": DEMO_RECORD,
        "risk_score": demo_score,
        "screen_positive": demo_score >= best_threshold,
        "threshold": best_threshold,
        "disclaimer": "Screening support only; this score is not a diagnosis or a calibrated future-risk probability.",
    }
    (ARTIFACT_DIR / "demo_prediction.json").write_text(json.dumps(demo_result, indent=2))
    joblib.dump(best_model, ARTIFACT_DIR / "diabetes_screening_pipeline.joblib", compress=3)

    baseline_ap = float(comparison.loc[comparison["model"] == "Logistic baseline", "average_precision"].iloc[0])
    best_ap = float(comparison.loc[comparison["model"] == best_name, "average_precision"].iloc[0])
    metadata = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "random_seed": SEED,
        "quick_mode": args.quick,
        "selected_model": best_name,
        "selection_rule": "highest validation average precision",
        "operating_threshold": best_threshold,
        "threshold_rule": "maximum validation F2 (recall weighted twice precision)",
        "validation_f2_at_threshold": validation_f2,
        "test_metrics_at_operating_threshold": operating_metrics,
        "test_average_precision_gain_over_baseline": best_ap - baseline_ap,
        "tuning": tuning,
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "shap": shap.__version__,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    (ARTIFACT_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    print("\nUntouched test-set comparison (default threshold 0.50):")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSelected {best_name}; validation-only F2 threshold={best_threshold:.3f}")
    print(f"Demo risk score={demo_score:.3f}; screen_positive={demo_score >= best_threshold}")
    print(f"Artifacts written to {ARTIFACT_DIR}, {TABLE_DIR}, and {FIGURE_DIR}")


if __name__ == "__main__":
    main()
