"""Fit and persist the required Spark binary classifiers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.pipeline import PipelineModel

from src.feature_engineering import SCALED_FEATURES_COLUMN, build_pipeline, prepare_model_frame

LOGGER = logging.getLogger(__name__)
MODEL_DIRECTORY = Path(__file__).resolve().parent.parent / "models"


def train_models(
    training_data,
    test_data,
    models_directory: str | Path = MODEL_DIRECTORY,
    random_seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """Train Logistic Regression and GBT pipelines using training data only."""
    model_directory = Path(models_directory)
    model_directory.mkdir(parents=True, exist_ok=True)
    train_frame = prepare_model_frame(training_data)
    test_frame = prepare_model_frame(test_data)

    classifiers = {
        "Logistic Regression": LogisticRegression(
            labelCol="label",
            featuresCol=SCALED_FEATURES_COLUMN,
            maxIter=100,
            regParam=0.01,
            elasticNetParam=0.0,
        ),
        "Gradient-Boosted Trees": GBTClassifier(
            labelCol="label",
            featuresCol=SCALED_FEATURES_COLUMN,
            maxIter=30,
            maxDepth=5,
            seed=random_seed,
        ),
    }

    results: dict[str, dict[str, Any]] = {}
    for model_name, classifier in classifiers.items():
        LOGGER.info("Training %s", model_name)
        fitted_pipeline = build_pipeline(classifier).fit(train_frame)
        predictions = fitted_pipeline.transform(test_frame).cache()
        predictions.count()
        model_path = model_directory / model_name.lower().replace(" ", "_").replace("-", "")
        fitted_pipeline.write().overwrite().save(str(model_path))
        results[model_name] = {
            "model": fitted_pipeline,
            "predictions": predictions,
            "model_path": str(model_path),
        }
        LOGGER.info("Saved %s pipeline to %s", model_name, model_path)
    return results


def load_saved_model(path: str | Path) -> PipelineModel:
    """Load a persisted feature-and-classification pipeline."""
    return PipelineModel.load(str(path))
