"""Download the public UCI Online Shoppers Purchasing Intention dataset."""

from __future__ import annotations

import logging
import re
import zipfile
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

LOGGER = logging.getLogger(__name__)

DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "customer_behaviour.csv"
DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/468/"
    "online+shoppers+purchasing+intention+dataset.zip"
)
UCI_DATASET_PAGE = (
    "https://archive.ics.uci.edu/dataset/468/"
    "online+shoppers+purchasing+intention+dataset"
)

COLUMN_RENAMES = {
    "administrative": "administrative",
    "administrativeduration": "administrative_duration",
    "informational": "informational",
    "informationalduration": "informational_duration",
    "productrelated": "product_related",
    "productrelatedduration": "product_related_duration",
    "bouncerates": "bounce_rates",
    "exitrates": "exit_rates",
    "pagevalues": "page_values",
    "specialday": "special_day",
    "month": "month",
    "operatingsystems": "operating_systems",
    "browser": "browser",
    "region": "region",
    "traffictype": "traffic_type",
    "visitortype": "visitor_type",
    "weekend": "weekend",
    "revenue": "purchase",
}

NUMERIC_COLUMNS = (
    "administrative",
    "administrative_duration",
    "informational",
    "informational_duration",
    "product_related",
    "product_related_duration",
    "bounce_rates",
    "exit_rates",
    "page_values",
    "special_day",
)
CATEGORICAL_COLUMNS = (
    "month",
    "operating_systems",
    "browser",
    "region",
    "traffic_type",
    "visitor_type",
    "weekend",
)
TARGET_COLUMN = "purchase"
DATASET_COLUMNS = (*NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS, TARGET_COLUMN)


def _normalize_column_name(column: str) -> str:
    """Convert the dataset's original names into the documented project schema."""
    normalized = re.sub(r"[^a-z0-9]", "", column.lower())
    try:
        return COLUMN_RENAMES[normalized]
    except KeyError as exc:
        raise ValueError(f"Unexpected column in UCI dataset: {column!r}") from exc


def _read_dataset_archive(archive_bytes: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(BytesIO(archive_bytes)) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError("The UCI download did not contain a CSV file")
        with archive.open(csv_names[0]) as csv_file:
            frame = pd.read_csv(csv_file)
    frame.columns = [_normalize_column_name(str(column)) for column in frame.columns]

    missing_columns = sorted(set(DATASET_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(
            "The UCI dataset is missing expected columns: " + ", ".join(missing_columns)
        )
    if frame.empty:
        raise ValueError("The UCI dataset contains no records")

    frame = frame.loc[:, DATASET_COLUMNS].copy()
    frame[TARGET_COLUMN] = (
        frame[TARGET_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": 1, "false": 0, "1": 1, "0": 0})
    )
    if frame[TARGET_COLUMN].isna().any():
        raise ValueError("The UCI purchase target contains values other than True or False")

    frame["weekend"] = (
        frame["weekend"].astype(str).str.strip().str.lower().map(
            {"true": "True", "false": "False", "1": "True", "0": "False"}
        )
    )
    if frame["weekend"].isna().any():
        raise ValueError("The UCI weekend column contains unexpected values")

    return frame


def ensure_dataset(
    path: str | Path = DATASET_PATH,
    *,
    source_url: str = DATASET_URL,
    overwrite: bool = False,
    timeout: int = 60,
) -> Path:
    """Ensure the official UCI CSV exists locally and return its path.

    Existing files are reused unless ``overwrite`` is true. Downloads are
    validated before the destination is replaced.
    """
    destination = Path(path)
    if destination.is_file() and not overwrite:
        LOGGER.info("Using existing dataset at %s", destination)
        return destination

    request = Request(source_url, headers={"User-Agent": "customer-purchase-prediction/1.0"})
    with urlopen(request, timeout=timeout) as response:
        frame = _read_dataset_archive(response.read())

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    try:
        frame.to_csv(temporary_path, index=False)
        temporary_path.replace(destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    LOGGER.info("Downloaded %s rows from %s to %s", len(frame), source_url, destination)
    return destination
