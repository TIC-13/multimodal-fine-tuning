from __future__ import annotations

from typing import Any

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from vlm_ft.evaluation.labels import UNKNOWN_LABEL


def _classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    *,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    if not y_true:
        return {"support": 0}

    all_labels = sorted(set(y_true) | set(y_pred))
    if labels is None:
        labels = all_labels

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        label: {
            "precision": report[label]["precision"],
            "recall": report[label]["recall"],
            "f1": report[label]["f1-score"],
            "support": int(report[label]["support"]),
        }
        for label in labels
        if label in report
    }

    return {
        "num_samples": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "weighted_precision": precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "weighted_recall": recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        },
    }


def compute_metrics(
    gold_diagnosis: list[str],
    pred_diagnosis: list[str],
    gold_malignancy: list[str | None],
    pred_malignancy: list[str | None],
) -> dict[str, Any]:
    num_unparsed = sum(1 for label in pred_diagnosis if label == UNKNOWN_LABEL)

    diagnosis_labels = sorted(set(gold_diagnosis) | set(pred_diagnosis))
    diagnosis = _classification_metrics(gold_diagnosis, pred_diagnosis, labels=diagnosis_labels)
    diagnosis["num_unparsed"] = num_unparsed

    malignancy_pairs = [
        (gold, pred)
        for gold, pred in zip(gold_malignancy, pred_malignancy, strict=True)
        if gold is not None
    ]
    malignancy: dict[str, Any] | None = None
    if malignancy_pairs:
        y_true, y_pred = zip(*malignancy_pairs, strict=True)
        malignancy_labels = sorted(set(y_true) | set(y_pred))
        malignancy = _classification_metrics(list(y_true), list(y_pred), labels=malignancy_labels)

    return {
        "num_samples": len(gold_diagnosis),
        "num_unparsed": num_unparsed,
        "accuracy": diagnosis["accuracy"],
        "macro_f1": diagnosis["macro_f1"],
        "weighted_f1": diagnosis["weighted_f1"],
        "diagnosis": diagnosis,
        "malignancy": malignancy,
    }
