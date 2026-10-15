"""Bronze: land raw ISO 20022 messages as-is, incrementally and idempotently.

Locally a file manifest (bronze.ingest_log) gives exactly-once file pickup. On Databricks the
same contract is provided by Auto Loader (see `autoloader_reader`), which tracks files in its
checkpoint and scales to millions of files.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from .lake import Lake
from .schemas import INGEST_LOG, ROW_TAGS, SCHEMAS


@dataclass
class BronzeResult:
    batch_id: int
    files: dict
    records: dict
    corrupt: dict


def read_xml(spark: SparkSession, files: list[str], message_type: str) -> DataFrame:
    return (
        spark.read.format("xml")
        .schema(SCHEMAS[message_type])
        .option("rowTag", ROW_TAGS[message_type])
        .option("mode", "PERMISSIVE")
        .option("columnNameOfCorruptRecord", "_corrupt_record")
        .load(files)
        .withColumn("_source_file", F.regexp_replace(F.col("_metadata.file_path"), r"^file:/*", "/"))
    )


def autoloader_reader(spark: SparkSession, path: str, message_type: str, schema_location: str) -> DataFrame:  # pragma: no cover
    """Databricks Auto Loader equivalent of the local manifest (streaming, run with trigger(availableNow=True))."""
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "xml")
        .option("rowTag", ROW_TAGS[message_type])
        .option("cloudFiles.schemaLocation", schema_location)
        .schema(SCHEMAS[message_type])
        .load(path)
        .withColumn("_source_file", F.col("_metadata.file_path"))
    )


def _discover(landing: Path, message_type: str, as_of: date | None) -> list[Path]:
    files = []
    for day_dir in sorted((landing / message_type).glob("*")):
        try:
            day = date.fromisoformat(day_dir.name)
        except ValueError:
            continue
        if as_of is None or day <= as_of:
            files += sorted(day_dir.glob("*.xml"))
    return files


def ingest(spark: SparkSession, lake: Lake, landing: Path, as_of: date | None = None) -> BronzeResult:
    log = lake.read_or_empty("bronze.ingest_log", INGEST_LOG)
    seen = {(r.file_path, r.file_size) for r in log.select("file_path", "file_size").collect()}
    last = log.agg(F.max("batch_id")).first()[0]
    batch_id = int(last or 0) + 1
    now = datetime.now(timezone.utc)

    result = BronzeResult(batch_id, {}, {}, {})
    log_rows = []
    for mt in ("pacs008", "pacs002"):
        new = [p for p in _discover(landing, mt, as_of) if (p.resolve().as_posix(), p.stat().st_size) not in seen]
        result.files[mt] = len(new)
        if not new:
            result.records[mt] = result.corrupt[mt] = 0
            continue
        df = (
            read_xml(spark, [p.resolve().as_posix() for p in new], mt)
            .withColumn("_ingested_at", F.lit(now))
            .withColumn("_batch_id", F.lit(batch_id).cast("long"))
        ).cache()
        per_file = {r["_source_file"]: (r["n"], r["bad"]) for r in df.groupBy("_source_file").agg(
            F.count("*").alias("n"), F.sum(F.col("_corrupt_record").isNotNull().cast("int")).alias("bad")).collect()}
        lake.append(df, f"bronze.{mt}")
        result.records[mt] = sum(v[0] for v in per_file.values())
        result.corrupt[mt] = sum(v[1] for v in per_file.values())
        for p in new:
            key = p.resolve().as_posix()
            n, bad = per_file.get(key, (0, 0))
            log_rows.append((key, p.stat().st_size, mt, batch_id, now, int(n), int(bad)))
        df.unpersist()
    if log_rows:
        lake.append(spark.createDataFrame(log_rows, INGEST_LOG), "bronze.ingest_log")
    return result
