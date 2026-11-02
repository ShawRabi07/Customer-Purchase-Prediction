"""Load, validate, split, and impute the UCI data with Spark."""

from __future__ import annotations

import logging
from typing import Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType

from src.data_loader import (
    CATEGORICAL_COLUMNS,
    DATASET_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
    ensure_dataset,
)

LOGGER = logging.getLogger(__name__)
ID_COLUMN = "session_id"


def load_dataset(spark: Any, path: str | None = None) -> DataFrame:
    """Load the project CSV as a Spark DataFrame."""
    dataset_path = ensure_dataset(path) if path is not None else ensure_dataset()
    frame = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .option("mode", "FAILFAST")
        .csv(str(dataset_path))
    )
    missing_columns = sorted(set(DATASET_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing_columns))
    LOGGER.info("Loaded dataset from %s", dataset_path)
    return frame.select(*DATASET_COLUMNS)


def _clean_frame(frame: DataFrame) -> DataFrame:
    cleaned = frame.select(
        *[F.col(column).cast(DoubleType()).alias(column) for column in NUMERIC_COLUMNS],
        *[F.col(column).cast(StringType()).alias(column) for column in CATEGORICAL_COLUMNS],
        F.when(
            F.lower(F.col(TARGET_COLUMN).cast("string")).isin("true", "1"),
            F.lit(1.0),
        )
        .when(F.lower(F.col(TARGET_COLUMN).cast("string")).isin("false", "0"), F.lit(0.0))
        .otherwise(F.col(TARGET_COLUMN).cast(DoubleType()))
        .alias(TARGET_COLUMN),
    )
    cleaned = cleaned.dropDuplicates(DATASET_COLUMNS)

    nonnegative_columns = (
        "administrative",
        "administrative_duration",
        "informational",
        "informational_duration",
        "product_related",
        "product_related_duration",
        "page_values",
    )
    for column in nonnegative_columns:
        cleaned = cleaned.filter(F.col(column).isNull() | (F.col(column) >= 0))
    for column in ("bounce_rates", "exit_rates", "special_day"):
        cleaned = cleaned.filter(
            F.col(column).isNull() | F.col(column).between(0.0, 1.0)
        )
    cleaned = cleaned.filter(F.col(TARGET_COLUMN).isin(0.0, 1.0))
    return cleaned


def _impute_from_training(training: DataFrame, testing: DataFrame) -> tuple[DataFrame, DataFrame]:
    numeric_medians = training.agg(
        *[F.percentile_approx(F.col(column), 0.5).alias(column) for column in NUMERIC_COLUMNS]
    ).first()
    medians = {
        column: float(numeric_medians[column]) if numeric_medians[column] is not None else 0.0
        for column in NUMERIC_COLUMNS
    }

    categorical_modes: dict[str, str] = {}
    for column in CATEGORICAL_COLUMNS:
        mode = (
            training.filter(F.col(column).isNotNull() & (F.trim(F.col(column)) != ""))
            .groupBy(column)
            .count()
            .orderBy(F.desc("count"), F.asc(column))
            .first()
        )
        categorical_modes[column] = str(mode[column]) if mode is not None else "Unknown"

    def apply_imputation(data: DataFrame) -> DataFrame:
        result = data
        for column, median in medians.items():
            result = result.withColumn(
                column,
                F.when(
                    F.col(column).isNull() | F.isnan(F.col(column)),
                    F.lit(median),
                ).otherwise(F.col(column)),
            )
        for column, mode in categorical_modes.items():
            result = result.withColumn(
                column,
                F.when(
                    F.col(column).isNull() | (F.trim(F.col(column)) == ""),
                    F.lit(mode),
                ).otherwise(F.col(column)),
            )
        return result

    return apply_imputation(training), apply_imputation(testing)


def preprocess_data(
    frame: DataFrame,
    train_ratio: float = 0.8,
    random_seed: int = 42,
) -> tuple[DataFrame, DataFrame, dict[str, Any]]:
    """Clean and split the dataset; fit imputation values on training data only."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be strictly between 0 and 1")
    cleaned = _clean_frame(frame).cache()
    cleaned_count = cleaned.count()
    if cleaned_count < 2:
        raise ValueError("At least two valid dataset rows are required")

    # UCI provides one session per user but no record identifier; this is a generated row ID.
    ordered = cleaned.withColumn(ID_COLUMN, F.monotonically_increasing_id()).cache()
    ordered.count()
    train, test = ordered.randomSplit([train_ratio, 1 - train_ratio], seed=random_seed)
    train, test = train.cache(), test.cache()
    train_count, test_count = train.count(), test.count()
    if train_count == 0 or test_count == 0:
        raise ValueError("The train/test split produced an empty partition")
    train, test = _impute_from_training(train, test)
    train, test = train.cache(), test.cache()
    train_count, test_count = train.count(), test.count()

    null_counts = {
        column: frame.filter(F.col(column).isNull()).count()
        for column in frame.columns
    }
    class_distribution = {
        str(row[TARGET_COLUMN]): int(row["count"])
        for row in cleaned.groupBy(TARGET_COLUMN).count().collect()
    }
    summary = {
        "rows_before_cleaning": frame.count(),
        "rows_after_cleaning": cleaned_count,
        "columns": len(frame.columns),
        "column_names": list(frame.columns),
        "null_counts": null_counts,
        "class_distribution": class_distribution,
        "train_rows": train_count,
        "test_rows": test_count,
        "train_ratio": train_ratio,
        "random_seed": random_seed,
    }
    LOGGER.info(
        "Prepared %s rows (%s train, %s test)",
        cleaned_count,
        summary["train_rows"],
        summary["test_rows"],
    )
    return train, test, summary
