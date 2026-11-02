"""Evaluate saved Spark predictions and produce dashboard/report artifacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

LOGGER = logging.getLogger(__name__)
METRIC_EXPLANATIONS = {
    "accuracy": "Share of all test sessions whose purchase status was classified correctly.",
    "precision": "Share of sessions predicted as purchases that were actual purchases.",
    "recall": "Share of actual purchase sessions that the model correctly identified.",
    "f1": "Harmonic mean of purchase precision and recall.",
    "roc_auc": "Ability to rank purchase sessions above non-purchase sessions across thresholds.",
}


def _calculate_metrics(predictions) -> dict[str, float]:
    multiclass = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
    )
    metrics = {
        "accuracy": multiclass.setMetricName("accuracy").evaluate(predictions),
        "precision": multiclass.setMetricName("precisionByLabel").setMetricLabel(1.0).evaluate(
            predictions
        ),
        "recall": multiclass.setMetricName("recallByLabel").setMetricLabel(1.0).evaluate(
            predictions
        ),
        "f1": multiclass.setMetricName("fMeasureByLabel").setMetricLabel(1.0).evaluate(
            predictions
        ),
        "roc_auc": BinaryClassificationEvaluator(
            labelCol="label",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC",
        ).evaluate(predictions),
    }
    return {name: float(value) for name, value in metrics.items()}


def _prediction_rows(predictions) -> pd.DataFrame:
    return predictions.select("label", "prediction", "probability").toPandas().assign(
        purchase_probability=lambda frame: frame["probability"].map(lambda vector: float(vector[1]))
    )


def _save_plots(
    results: dict[str, dict[str, Any]],
    metric_results: dict[str, dict[str, float]],
    best_model: str,
    output_directory: Path,
) -> None:
    best_rows = _prediction_rows(results[best_model]["predictions"])
    best_rows["label"] = best_rows["label"].astype(int)
    best_rows["prediction"] = best_rows["prediction"].astype(int)

    confusion = pd.crosstab(
        best_rows["label"],
        best_rows["prediction"],
        dropna=False,
    ).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    figure, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No purchase", "Purchase"],
        yticklabels=["No purchase", "Purchase"],
        ax=axis,
    )
    axis.set(title=f"Confusion Matrix - {best_model}", xlabel="Predicted", ylabel="Actual")
    figure.tight_layout()
    figure.savefig(output_directory / "confusion_matrix.png", dpi=160)
    plt.close(figure)

    from sklearn.metrics import roc_curve

    false_positive_rate, true_positive_rate, _ = roc_curve(
        best_rows["label"],
        best_rows["purchase_probability"],
    )
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(false_positive_rate, true_positive_rate, label=f"{best_model} (AUC {metric_results[best_model]['roc_auc']:.3f})")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random classifier")
    axis.set(title="ROC Curve", xlabel="False positive rate", ylabel="True positive rate")
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_directory / "roc_curve.png", dpi=160)
    plt.close(figure)

    metric_frame = pd.DataFrame(metric_results).T
    axis = metric_frame.plot(kind="bar", figsize=(10, 6), ylim=(0, 1), rot=0)
    axis.set(title="Model Performance Comparison", ylabel="Score")
    axis.legend(loc="lower right")
    axis.figure.tight_layout()
    axis.figure.savefig(output_directory / "model_comparison.png", dpi=160)
    plt.close(axis.figure)

    actual_predicted = (
        best_rows.groupby(["label", "prediction"]).size().unstack(fill_value=0).reindex(
            index=[0, 1], columns=[0, 1], fill_value=0
        )
    )
    axis = actual_predicted.plot(kind="bar", figsize=(7, 5), rot=0)
    axis.set(
        title=f"Actual vs Predicted Sessions - {best_model}",
        xlabel="Actual outcome (0 = no purchase, 1 = purchase)",
        ylabel="Number of sessions",
    )
    axis.legend(["Predicted no purchase", "Predicted purchase"])
    axis.figure.tight_layout()
    axis.figure.savefig(output_directory / "actual_vs_predicted.png", dpi=160)
    plt.close(axis.figure)


def evaluate_models(
    results: dict[str, dict[str, Any]],
    output_directory: str | Path,
) -> dict[str, Any]:
    """Calculate all requested metrics and save JSON plus diagnostic plots."""
    if not results:
        raise ValueError("At least one trained model is required for evaluation")
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)

    metrics = {
        model_name: _calculate_metrics(result["predictions"])
        for model_name, result in results.items()
    }
    best_model = max(
        metrics,
        key=lambda name: (metrics[name]["f1"], metrics[name]["roc_auc"], metrics[name]["accuracy"]),
    )
    _save_plots(results, metrics, best_model, destination)
    test_rows = results[best_model]["predictions"].count()
    report = {
        "dataset": "UCI Online Shoppers Purchasing Intention Dataset",
        "test_rows": test_rows,
        "positive_label": 1,
        "best_model": best_model,
        "selection_metric": "F1-score for the purchase class; ROC-AUC and accuracy break ties",
        "models": metrics,
        "metric_explanations": METRIC_EXPLANATIONS,
        "artifacts": {
            "confusion_matrix": "confusion_matrix.png",
            "roc_curve": "roc_curve.png",
            "model_comparison": "model_comparison.png",
            "actual_vs_predicted": "actual_vs_predicted.png",
        },
    }
    metrics_path = destination / "model_metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info("Best model: %s; metrics saved to %s", best_model, metrics_path)
    return report
