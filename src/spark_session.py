"""Shared configuration for the project's local PySpark session."""

from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path

from pyspark.sql import SparkSession

LOGGER = logging.getLogger(__name__)


def _configure_windows_hadoop() -> None:
    hadoop_home = os.environ.get("HADOOP_HOME")
    if hadoop_home and (Path(hadoop_home) / "bin" / "winutils.exe").is_file():
        return

    winutils = shutil.which("winutils.exe")
    if winutils:
        hadoop_home = str(Path(winutils).resolve().parent.parent)
        os.environ["HADOOP_HOME"] = hadoop_home
        os.environ["hadoop.home.dir"] = hadoop_home
        LOGGER.info("Using winutils from %s", winutils)
        return

    raise RuntimeError(
        "Windows Spark model persistence requires winutils.exe. Install Hadoop "
        "winutils matching the Hadoop version bundled with PySpark, set HADOOP_HOME "
        "to its parent directory, and add HADOOP_HOME\\bin to PATH."
    )


def create_spark_session(app_name: str = "CustomerPurchasePrediction") -> SparkSession:
    """Create or reuse a Windows-compatible local Spark session."""
    if os.name == "nt":
        _configure_windows_hadoop()
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    builder = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.session.timeZone", "UTC")
    )
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    LOGGER.info("Started Spark %s in %s mode", spark.version, spark.sparkContext.master)
    return spark
