"""Export holdout predictions and support single-session inference."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.data_loader import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from src.data_preprocessing import ID_COLUMN
from src.feature_engineering import prepare_model_frame
from src.model_training import load_saved_model

LOGGER = logging.getLogger(__name__)
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "outputs" / "purchase_predictions.csv"


def export_predictions(
    model_path: str | Path,
    test_data,
    output_path: str | Path = OUTPUT_PATH,
) -> Path:
    """Transform holdout sessions with the persisted pipeline and save predictions."""
    model = load_saved_model(model_path)
    transformed = model.transform(prepare_model_frame(test_data))
    selected_columns = [
        ID_COLUMN,
        *NUMERIC_COLUMNS,
        *CATEGORICAL_COLUMNS,
        "label",
        "prediction",
        "probability",
    ]
    records = transformed.select(*selected_columns).toPandas()
    records["purchase_probability"] = records["probability"].map(
        lambda value: float(value[1])
    )
    records["customer_id"] = records[ID_COLUMN].map(
        lambda value: f"session-{int(value):06d}"
    )
    records["actual_purchase"] = records["label"].astype(int)
    records["predicted_purchase"] = records["prediction"].astype(int)
    records["correct_prediction"] = (
        records["actual_purchase"] == records["predicted_purchase"]
    )
    records = records[
        [
            "customer_id",
            *NUMERIC_COLUMNS,
            *CATEGORICAL_COLUMNS,
            "actual_purchase",
            "predicted_purchase",
            "purchase_probability",
            "correct_prediction",
        ]
    ]
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    records.to_csv(destination, index=False)
    LOGGER.info("Saved %s test predictions to %s", len(records), destination)
    return destination


def predict_session(model, spark, features: dict[str, Any]) -> dict[str, Any]:
    """Predict a single session using the same persisted feature pipeline."""
    expected = {*NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS}
    missing = expected - features.keys()
    extra = features.keys() - expected
    if missing or extra:
        raise ValueError(
            f"Feature fields do not match model schema; missing={sorted(missing)}, "
            f"unexpected={sorted(extra)}"
        )
    row = {
        column: (
            str(features[column])
            if column in CATEGORICAL_COLUMNS
            else float(features[column])
        )
        for column in [*NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS]
    }
    prediction = model.transform(spark.createDataFrame([row])).first()
    return {
        "predicted_purchase": int(prediction["prediction"]),
        "purchase_probability": float(prediction["probability"][1]),
    }
