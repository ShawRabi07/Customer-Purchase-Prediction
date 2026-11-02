import csv
import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from src.data_loader import DATASET_COLUMNS, _read_dataset_archive, ensure_dataset


def build_archive(purchase_value: str = "True") -> bytes:
    headers = [
        "Administrative",
        "Administrative_Duration",
        "Informational",
        "Informational_Duration",
        "ProductRelated",
        "ProductRelated_Duration",
        "BounceRates",
        "ExitRates",
        "PageValues",
        "SpecialDay",
        "Month",
        "OperatingSystems",
        "Browser",
        "Region",
        "TrafficType",
        "VisitorType",
        "Weekend",
        "Revenue",
    ]
    row = [
        "1",
        "10.5",
        "0",
        "0",
        "3",
        "120.0",
        "0.02",
        "0.04",
        "0.0",
        "0.0",
        "Feb",
        "1",
        "2",
        "1",
        "3",
        "Returning_Visitor",
        "False",
        purchase_value,
    ]
    text = io.StringIO()
    writer = csv.writer(text)
    writer.writerow(headers)
    writer.writerow(row)
    result = io.BytesIO()
    with zipfile.ZipFile(result, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("online_shoppers_intention.csv", text.getvalue())
    return result.getvalue()


class DatasetLoaderTests(unittest.TestCase):
    def test_reads_and_normalizes_official_dataset_schema(self):
        frame = _read_dataset_archive(build_archive())
        self.assertEqual(tuple(frame.columns), DATASET_COLUMNS)
        self.assertEqual(frame.loc[0, "purchase"], 1)
        self.assertEqual(frame.loc[0, "weekend"], "False")
        self.assertEqual(frame.loc[0, "visitor_type"], "Returning_Visitor")

    def test_rejects_unrecognized_target_values(self):
        with self.assertRaisesRegex(ValueError, "purchase target"):
            _read_dataset_archive(build_archive("sometimes"))

    def test_downloads_validated_archive_and_reuses_existing_file(self):
        response = io.BytesIO(build_archive())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "customer_behaviour.csv"
            with patch("src.data_loader.urlopen", return_value=response):
                self.assertEqual(
                    ensure_dataset(path, source_url="https://example.test/data.zip"),
                    path,
                )
            self.assertTrue(path.is_file())
            original = path.read_bytes()
            with patch("src.data_loader.urlopen", side_effect=AssertionError("unexpected download")):
                ensure_dataset(path, source_url="https://example.test/data.zip")
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
