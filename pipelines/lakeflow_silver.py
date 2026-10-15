# Databricks notebook source
# Lakeflow Declarative Pipelines (formerly DLT) version of bronze -> silver.
#
# The job in resources/payments_job.yml is the primary path. This file shows the same layer
# declaratively for customers standardised on Lakeflow: Auto Loader for ingestion and the
# *same* expectation registry (paylake.expectations.RULES) enforced by the pipeline engine,
# so DQ rules are written once and behave identically in both paths.
# Not exercised in local CI (the `dlt` module only exists inside a pipeline).

import dlt  # noqa: E402
from pyspark.sql import functions as F

from paylake.config import Settings
from paylake.expectations import lakeflow_expectations
from paylake.schemas import ROW_TAGS, SCHEMAS
from paylake.silver import flatten_pacs008

settings = Settings()
LANDING = spark.conf.get("paylake.landing", f"/Volumes/{settings.catalog}/bronze/landing")  # noqa: F821
WARN, DROP = lakeflow_expectations()


@dlt.table(name="pacs008_raw", comment="Raw pacs.008 messages via Auto Loader")
def pacs008_raw():
    return (
        spark.readStream.format("cloudFiles")  # noqa: F821
        .option("cloudFiles.format", "xml")
        .option("rowTag", ROW_TAGS["pacs008"])
        .schema(SCHEMAS["pacs008"])
        .load(f"{LANDING}/pacs008")
        .withColumn("_source_file", F.col("_metadata.file_path"))
        .withColumn("_ingested_at", F.current_timestamp())
        .withColumn("_batch_id", F.lit(0).cast("long"))
    )


@dlt.view(name="payments_candidates")
def payments_candidates():
    return flatten_pacs008(dlt.read_stream("pacs008_raw"), settings)


@dlt.table(name="payments_quarantine", comment="Rows failing a quarantine-level rule")
def payments_quarantine():
    failing = " OR ".join(f"NOT coalesce({c}, false)" for c in DROP.values())
    return dlt.read_stream("payments_candidates").where(failing)


@dlt.table(name="payments_validated", comment="Rows passing all quarantine-level rules; warnings tracked in event log")
@dlt.expect_all(WARN)
@dlt.expect_all_or_drop(DROP)
def payments_validated():
    # re-sent files: drop repeated UETRs arriving within three days of the first copy
    return (
        dlt.read_stream("payments_candidates")
        .withWatermark("accepted_at", "3 days")
        .dropDuplicatesWithinWatermark(["uetr"])
    )
