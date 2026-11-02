"""Run the complete local Spark training, evaluation, and reporting workflow."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.data_loader import DATASET_PATH, DATASET_URL, UCI_DATASET_PAGE, ensure_dataset
from src.data_preprocessing import load_dataset, preprocess_data
from src.model_evaluation import evaluate_models
from src.model_training import train_models
from src.prediction import export_predictions
from src.spark_session import create_spark_session

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs"
MODEL_DIRECTORY = PROJECT_ROOT / "models"
REPORT_PATH = PROJECT_ROOT / "report" / "project_report.pdf"


def run_pipeline(dataset_path: Path = DATASET_PATH, refresh_data: bool = False) -> dict:
    """Execute ingestion through PDF report generation and return run artifacts."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)

    local_dataset = ensure_dataset(dataset_path, overwrite=refresh_data)
    spark = create_spark_session()
    try:
        raw_data = load_dataset(spark, str(local_dataset))
        print("\nLoaded Spark DataFrame schema:")
        raw_data.printSchema()
        print("\nFirst 10 sessions:")
        raw_data.show(10, truncate=False)
        print(f"Rows: {raw_data.count():,} | Columns: {len(raw_data.columns)}")

        training_data, test_data, preprocessing_summary = preprocess_data(raw_data)
        summary_path = OUTPUT_DIRECTORY / "preprocessing_summary.json"
        summary_path.write_text(json.dumps(preprocessing_summary, indent=2), encoding="utf-8")
        print("\nNull values before cleaning:")
        print(json.dumps(preprocessing_summary["null_counts"], indent=2))
        print("\nPurchase class distribution:")
        print(json.dumps(preprocessing_summary["class_distribution"], indent=2))
        print("\nPreprocessed training sample:")
        training_data.show(5, truncate=False)

        trained_models = train_models(
            training_data,
            test_data,
            models_directory=MODEL_DIRECTORY,
            random_seed=42,
        )
        evaluation = evaluate_models(trained_models, OUTPUT_DIRECTORY)
        predictions_path = export_predictions(
            trained_models[evaluation["best_model"]]["model_path"],
            test_data,
        )

        from report.generate_report import generate_report

        report_path = generate_report(
            preprocessing_summary=preprocessing_summary,
            evaluation=evaluation,
            dataset_path=local_dataset,
            output_path=REPORT_PATH,
        )
        return {
            "dataset": str(local_dataset),
            "dataset_source": DATASET_URL,
            "dataset_page": UCI_DATASET_PAGE,
            "metrics": str(OUTPUT_DIRECTORY / "model_metrics.json"),
            "predictions": str(predictions_path),
            "report": str(report_path),
            "best_model": evaluation["best_model"],
        }
    finally:
        spark.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DATASET_PATH,
        help="Path to the UCI CSV (downloaded there if it does not exist).",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Download and validate the official UCI archive again.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    artifacts = run_pipeline(**vars(parse_args()))
    print("\nPipeline completed successfully:")
    print(json.dumps(artifacts, indent=2))
