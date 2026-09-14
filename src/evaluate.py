"""Shared metrics, confusion matrices, model comparison, and saved-model evaluation."""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.config import settings
from src.data_loader import load_prepared_splits
from src.utils import ensure_directories, load_json, save_json


def classification_metrics(y_true: Sequence[Any], y_pred: Sequence[Any]) -> Dict[str, float]:
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
    }


def save_confusion_matrix(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str], path: Path, title: str
) -> np.ndarray:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    figure_size = max(10, min(20, len(labels) * 0.65))
    plt.figure(figsize=(figure_size, figure_size * 0.85))
    sns.heatmap(matrix, cmap="Blues", xticklabels=labels, yticklabels=labels, annot=False)
    plt.title(title)
    plt.xlabel("Predicted intent")
    plt.ylabel("Actual intent")
    plt.xticks(rotation=65, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return matrix


def evaluate_predictions(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    model_name: str,
    labels: Optional[Sequence[str]] = None,
    extra_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Calculate and persist all requested classification diagnostics."""
    ensure_directories()
    label_order = list(labels or sorted(set(y_true) | set(y_pred)))
    metrics: Dict[str, Any] = classification_metrics(y_true, y_pred)
    if extra_metrics:
        metrics.update(extra_metrics)

    report = classification_report(
        y_true,
        y_pred,
        labels=label_order,
        target_names=label_order,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        settings.metrics_dir / f"{model_name}_classification_report.csv"
    )
    save_json(metrics, settings.metrics_dir / f"{model_name}_metrics.json")
    save_json(report, settings.metrics_dir / f"{model_name}_classification_report.json")
    save_confusion_matrix(
        y_true,
        y_pred,
        label_order,
        settings.figures_dir / f"{model_name}_confusion_matrix.png",
        f"{model_name.replace('_', ' ').title()} confusion matrix",
    )
    return metrics


def save_error_analysis(
    texts: Sequence[str],
    y_true: Sequence[str],
    y_pred: Sequence[str],
    confidences: Sequence[float],
    model_name: str,
) -> pd.DataFrame:
    results = pd.DataFrame(
        {
            "ticket_text": list(texts),
            "actual_intent": list(y_true),
            "predicted_intent": list(y_pred),
            "confidence": list(confidences),
        }
    )
    errors = results[results.actual_intent != results.predicted_intent].copy()
    errors = errors.sort_values("confidence", ascending=False)
    errors.to_csv(settings.metrics_dir / f"{model_name}_error_analysis.csv", index=False)

    confused = (
        errors.groupby(["actual_intent", "predicted_intent"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    confused.to_csv(settings.metrics_dir / f"{model_name}_most_confused_classes.csv", index=False)
    summary_lines = [
        f"# {model_name.replace('_', ' ').title()} error analysis",
        "",
        f"The model made {len(errors):,} mistakes among {len(results):,} test tickets.",
        "",
        "## Most frequent confusion directions",
        "",
    ]
    if confused.empty:
        summary_lines.append("No incorrect predictions were found.")
    else:
        for row in confused.head(10).itertuples(index=False):
            summary_lines.append(
                f"- Actual `{row.actual_intent}` predicted as `{row.predicted_intent}`: {row.count}"
            )
    summary_lines.extend(
        [
            "",
            "## Interpretation prompts",
            "",
            "Review the corresponding rows in the error CSV before drawing conclusions. Common causes "
            "to test include overlapping intent definitions, short tickets without enough context, "
            "uncommon wording or spelling, entity placeholders, and annotation ambiguity. Highly "
            "confident errors deserve priority because a confidence threshold will not catch them. "
            "These are hypotheses, not measured causes; validate each against the actual examples.",
        ]
    )
    (settings.metrics_dir / f"{model_name}_error_analysis.md").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )
    return errors


def update_model_comparison() -> Optional[pd.DataFrame]:
    """Create a comparison only from metric files produced by real model runs."""
    rows = []
    for key, display_name in (
        ("baseline", "TF-IDF + Logistic Regression"),
        ("distilbert", "DistilBERT"),
    ):
        path = settings.metrics_dir / f"{key}_metrics.json"
        if path.exists():
            values = load_json(path)
            rows.append(
                {
                    "model": display_name,
                    "accuracy": values.get("accuracy"),
                    "macro_f1": values.get("f1_macro"),
                    "weighted_f1": values.get("f1_weighted"),
                    "training_time_seconds": values.get("training_time_seconds"),
                    "test_inference_time_seconds": values.get("test_inference_time_seconds"),
                    "milliseconds_per_ticket": values.get("milliseconds_per_ticket"),
                }
            )
    if not rows:
        return None
    comparison = pd.DataFrame(rows)
    comparison.to_csv(settings.metrics_dir / "model_comparison.csv", index=False)
    return comparison


def evaluate_saved_model(model_type: str) -> Dict[str, Any]:
    test = load_prepared_splits()["test"]
    texts = test[settings.text_column].tolist()
    truth = test[settings.label_column].tolist()

    if model_type == "baseline":
        import joblib

        if not settings.baseline_model_path.exists():
            raise FileNotFoundError("Train the baseline first: python -m src.train_baseline")
        model = joblib.load(settings.baseline_model_path)
        started = time.perf_counter()
        predictions = model.predict(texts)
        probabilities = model.predict_proba(texts)
        elapsed = time.perf_counter() - started
        confidences = probabilities.max(axis=1)
    else:
        from src.inference import TicketClassifier

        classifier = TicketClassifier(settings.transformer_output_dir)
        started = time.perf_counter()
        results = classifier.predict_batch(texts)
        elapsed = time.perf_counter() - started
        predictions = [item["intent"] for item in results]
        confidences = [item["confidence"] for item in results]

    previous_metrics_path = settings.metrics_dir / f"{model_type}_metrics.json"
    previous_metrics = load_json(previous_metrics_path) if previous_metrics_path.exists() else {}
    extras = {
        "training_time_seconds": previous_metrics.get("training_time_seconds"),
        "test_inference_time_seconds": elapsed,
        "milliseconds_per_ticket": elapsed * 1000 / len(texts),
    }
    metrics = evaluate_predictions(truth, predictions, model_type, extra_metrics=extras)
    save_error_analysis(texts, truth, predictions, confidences, model_type)
    update_model_comparison()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved classifier on the fixed test split.")
    parser.add_argument("--model", choices=("baseline", "distilbert"), required=True)
    args = parser.parse_args()
    metrics = evaluate_saved_model(args.model)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
