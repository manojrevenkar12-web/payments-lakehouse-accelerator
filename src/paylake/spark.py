"""SparkSession factory: reuse the cluster/serverless session on Databricks, build a tuned local one elsewhere."""
from __future__ import annotations

import os

from pyspark.sql import SparkSession

from .config import Settings


def get_spark(settings: Settings | None = None, app_name: str = "paylake") -> SparkSession:
    settings = settings or Settings()
    if settings.is_databricks:
        return SparkSession.builder.getOrCreate()

    os.environ.setdefault("PYARROW_IGNORE_TIMEZONE", "1")
    builder = (
        SparkSession.builder.master(os.environ.get("PAYLAKE_SPARK_MASTER", "local[*]"))
        .appName(app_name)
        .config("spark.sql.session.timeZone", "UTC")
        # small local box: keep shuffles narrow and let AQE coalesce
        .config("spark.sql.shuffle.partitions", os.environ.get("PAYLAKE_SHUFFLE_PARTITIONS", "8"))
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.driver.memory", os.environ.get("PAYLAKE_DRIVER_MEMORY", "3g"))
        .config("spark.ui.enabled", "false")
    )
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark
