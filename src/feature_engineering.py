"""Shared Spark ML feature transformations used for training and prediction."""

from __future__ import annotations

from pyspark.ml import Pipeline
from pyspark.ml.base import Estimator, Transformer
from pyspark.ml.feature import OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler

from src.data_loader import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, TARGET_COLUMN

LABEL_COLUMN = "label"
FEATURES_COLUMN = "features"
SCALED_FEATURES_COLUMN = "scaled_features"


def build_feature_stages() -> list[Estimator | Transformer]:
    """Build fit-on-training transformations; saved pipelines reuse them unchanged."""
    index_columns = [f"{column}_index" for column in CATEGORICAL_COLUMNS]
    encoded_columns = [f"{column}_encoded" for column in CATEGORICAL_COLUMNS]
    stages: list[Estimator | Transformer] = [
        StringIndexer(
            inputCol=column,
            outputCol=index_column,
            handleInvalid="keep",
        )
        for column, index_column in zip(CATEGORICAL_COLUMNS, index_columns)
    ]
    stages.extend(
        [
            OneHotEncoder(
                inputCols=index_columns,
                outputCols=encoded_columns,
                handleInvalid="keep",
                dropLast=False,
            ),
            VectorAssembler(
                inputCols=[*NUMERIC_COLUMNS, *encoded_columns],
                outputCol=FEATURES_COLUMN,
                handleInvalid="keep",
            ),
            StandardScaler(
                inputCol=FEATURES_COLUMN,
                outputCol=SCALED_FEATURES_COLUMN,
                withMean=True,
                withStd=True,
            ),
        ]
    )
    return stages


def prepare_model_frame(frame):
    """Alias the project target to Spark ML's conventional label column."""
    return frame.withColumnRenamed(TARGET_COLUMN, LABEL_COLUMN)


def build_pipeline(classifier: Estimator) -> Pipeline:
    """Create one complete preprocessing-and-classification pipeline."""
    return Pipeline(stages=[*build_feature_stages(), classifier])
