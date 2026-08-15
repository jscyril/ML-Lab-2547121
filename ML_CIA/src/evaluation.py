"""Metrics, thresholding, and subgroup audits."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def metrics_at_threshold(y_true, probability, threshold: float = 0.5) -> dict:
    prediction = (np.asarray(probability) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "roc_auc": roc_auc_score(y_true, probability),
        "average_precision": average_precision_score(y_true, probability),
        "f1": f1_score(y_true, prediction, zero_division=0),
        "f2": fbeta_score(y_true, prediction, beta=2, zero_division=0),
        "precision": precision_score(y_true, prediction, zero_division=0),
        "recall_sensitivity": recall_score(y_true, prediction, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "balanced_accuracy": balanced_accuracy_score(y_true, prediction),
        "brier_score": brier_score_loss(y_true, probability),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def choose_f2_threshold(y_true, probability) -> tuple[float, float]:
    candidates = np.linspace(0.05, 0.95, 181)
    scores = [
        fbeta_score(y_true, np.asarray(probability) >= value, beta=2, zero_division=0)
        for value in candidates
    ]
    index = int(np.argmax(scores))
    return float(candidates[index]), float(scores[index])


def subgroup_metrics(y_true, probability, groups, threshold: float) -> pd.DataFrame:
    work = pd.DataFrame(
        {"target": np.asarray(y_true), "probability": probability, "group": np.asarray(groups)}
    )
    rows = []
    for group, part in work.groupby("group", observed=True):
        if len(part) < 30 or part["target"].nunique() < 2:
            continue
        result = metrics_at_threshold(part["target"], part["probability"], threshold)
        rows.append(
            {
                "group": str(group),
                "n": len(part),
                "prevalence": part["target"].mean(),
                "selection_rate": (part["probability"] >= threshold).mean(),
                "roc_auc": result["roc_auc"],
                "precision": result["precision"],
                "recall_tpr": result["recall_sensitivity"],
                "false_positive_rate": result["fp"] / (result["fp"] + result["tn"]),
            }
        )
    return pd.DataFrame(rows)

