"""Silver: conformed, validated, de-duplicated payments with clearing status applied.

Contract of silver.payments
  * one row per UETR (MERGE insert-only, so re-sent files and re-runs never duplicate)
  * typed: DECIMAL(18,2) amounts, UTC timestamps, partitioned by settlement_date
  * quarantine-level DQ failures never reach it (they go to silver.payments_quarantine)
  * warning-level DQ failures are kept and listed in dq_warnings
  * status / status_reason reflect the latest pacs.002 report for that UETR
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import LongType, StringType, StructField, StructType, TimestampType

from . import expectations
from .config import Settings
from .lake import Lake
from .privacy import iban_checksum_ok, iban_country, pseudonymise
from .reference import BIC_DIRECTORY_COLUMNS, bic_directory_rows

WATERMARKS = StructType([
    StructField("stage", StringType()), StructField("last_batch_id", LongType()), StructField("updated_at", TimestampType()),
])


@dataclass
class SilverResult:
    bronze_batches: list
    messages: int
    corrupt_messages: int
    transactions: int
    quarantined: int
    duplicates_skipped: int
    inserted: int
    status_updates: int


def _watermark(lake: Lake, stage: str) -> int:
    if not lake.exists("ops.watermarks"):
        return 0
    row = lake.read("ops.watermarks").filter(F.col("stage") == stage).first()
    return int(row["last_batch_id"]) if row else 0


def _set_watermark(spark: SparkSession, lake: Lake, stage: str, batch_id: int) -> None:
    df = spark.createDataFrame([(stage, batch_id, datetime.now(timezone.utc))], WATERMARKS)
    lake.merge(df, "ops.watermarks", ["stage"], update_cols=["last_batch_id", "updated_at"])


def flatten_pacs008(bronze: DataFrame, settings: Settings) -> DataFrame:
    tx = bronze.filter(F.col("_corrupt_record").isNull()).select(
        F.col("GrpHdr.MsgId").alias("msg_id"),
        F.col("GrpHdr.CreDtTm").alias("msg_created_at"),
        F.explode("CdtTrfTxInf").alias("t"),
        "_source_file", "_batch_id", "_ingested_at",
    )
    flat = tx.select(
        F.lower(F.col("t.PmtId.UETR")).alias("uetr"),
        F.col("t.PmtId.EndToEndId").alias("end_to_end_id"),
        F.col("t.PmtId.InstrId").alias("instr_id"),
        "msg_id", "msg_created_at",
        F.col("t.PmtTpInf.SvcLvl.Cd").alias("service_level"),
        F.col("t.AccptncDtTm").alias("accepted_at"),
        F.col("t.IntrBkSttlmDt").alias("settlement_date"),
        F.col("t.IntrBkSttlmAmt._VALUE").alias("amount"),
        F.upper(F.col("t.IntrBkSttlmAmt._Ccy")).alias("currency"),
        F.col("t.ChrgBr").alias("charge_bearer"),
        F.trim(F.col("t.Dbtr.Nm")).alias("debtor_name"),
        F.upper(F.regexp_replace("t.DbtrAcct.Id.IBAN", r"\s", "")).alias("debtor_iban"),
        F.upper(F.col("t.DbtrAgt.FinInstnId.BICFI")).alias("debtor_bic"),
        F.trim(F.col("t.Cdtr.Nm")).alias("creditor_name"),
        F.upper(F.regexp_replace("t.CdtrAcct.Id.IBAN", r"\s", "")).alias("creditor_iban"),
        F.upper(F.col("t.CdtrAgt.FinInstnId.BICFI")).alias("creditor_bic"),
        F.col("t.RmtInf.Ustrd").alias("remittance_info"),
        "_source_file", "_batch_id", "_ingested_at",
    )
    spark = bronze.sparkSession
    bic_dir = spark.createDataFrame(bic_directory_rows(), BIC_DIRECTORY_COLUMNS).select(
        F.col("bic").alias("creditor_bic"), F.col("bank_name").alias("creditor_bank_name"))
    return (
        flat.join(F.broadcast(bic_dir), "creditor_bic", "left")
        .withColumn("debtor_country", iban_country(F.col("debtor_iban")))
        .withColumn("creditor_country", iban_country(F.col("creditor_iban")))
        .withColumn("debtor_iban_checksum_ok", iban_checksum_ok(F.col("debtor_iban")))
        .withColumn("creditor_iban_checksum_ok", iban_checksum_ok(F.col("creditor_iban")))
        .withColumn("debtor_ref", pseudonymise(F.col("debtor_iban"), settings.hmac_key))
        .withColumn("creditor_ref", pseudonymise(F.col("creditor_iban"), settings.hmac_key))
    )


SILVER_COLUMNS = [
    "uetr", "end_to_end_id", "instr_id", "msg_id", "msg_created_at", "service_level", "accepted_at", "settlement_date",
    "amount", "currency", "charge_bearer", "debtor_name", "debtor_iban", "debtor_bic", "debtor_country", "debtor_ref",
    "creditor_name", "creditor_iban", "creditor_bic", "creditor_country", "creditor_bank_name", "creditor_ref",
    "remittance_info", "dq_warnings", "status", "status_reason", "status_at", "_source_file", "_batch_id", "_ingested_at",
]


def build(spark: SparkSession, lake: Lake, settings: Settings) -> SilverResult:
    last = _watermark(lake, "silver.payments")
    bronze = lake.read("bronze.pacs008").filter(F.col("_batch_id") > last)
    batches = sorted(r[0] for r in bronze.select("_batch_id").distinct().collect())
    if not batches:
        upd = apply_statuses(spark, lake)
        return SilverResult([], 0, 0, 0, 0, 0, 0, upd)
    batch_id = max(batches)

    bronze = bronze.cache()
    messages = bronze.count()
    corrupt = bronze.filter(F.col("_corrupt_record").isNotNull()).count()

    flat = flatten_pacs008(bronze, settings)
    clean, quarantined, metrics = expectations.apply(flat, batch_id)
    metrics.append((batch_id, "well_formed_xml", "quarantine", corrupt, messages, metrics[0][5]))

    clean = (
        clean.dropDuplicates(["uetr"])
        .withColumn("status", F.lit("PENDING"))
        .withColumn("status_reason", F.lit(None).cast("string"))
        .withColumn("status_at", F.lit(None).cast("timestamp"))
        .select(*SILVER_COLUMNS)
    ).cache()
    n_tx = flat.count()
    n_quarantined = quarantined.count()
    n_clean = clean.count()

    m = lake.merge(clean, "silver.payments", ["uetr"], update_cols=None, insert=True, partition_by=["settlement_date"])
    inserted = int(m["num_target_rows_inserted"])
    if n_quarantined:
        lake.append(quarantined.select(*[c for c in SILVER_COLUMNS if c not in ("status", "status_reason", "status_at", "dq_warnings")],
                                       "dq_warnings", "dq_failed", "quarantined_at"), "silver.payments_quarantine")
    duplicates = (n_tx - n_quarantined) - inserted
    metrics.append((batch_id, "duplicate_uetr", "skip", duplicates, n_tx - n_quarantined, metrics[0][5]))
    lake.append(spark.createDataFrame(metrics, expectations.METRICS_SCHEMA), "silver.dq_metrics")

    _set_watermark(spark, lake, "silver.payments", batch_id)
    upd = apply_statuses(spark, lake)
    bronze.unpersist()
    clean.unpersist()
    return SilverResult(batches, messages, corrupt, n_tx, n_quarantined, duplicates, inserted, upd)


def apply_statuses(spark: SparkSession, lake: Lake) -> int:
    """Rebuild status events from bronze pacs.002 and MERGE the latest status per UETR.

    Recomputing from all events makes out-of-order arrival harmless: a status report that
    lands before its payment is applied on the next run, once the payment exists.
    """
    if not lake.exists("bronze.pacs002") or not lake.exists("silver.payments"):
        return 0
    events = (
        lake.read("bronze.pacs002").filter(F.col("_corrupt_record").isNull())
        .select(F.col("GrpHdr.CreDtTm").alias("report_at"), F.explode("TxInfAndSts").alias("s"))
        .select(
            F.lower(F.col("s.OrgnlUETR")).alias("uetr"),
            F.col("s.TxSts").alias("status"),
            F.col("s.StsRsnInf.Rsn.Cd").alias("status_reason"),
            F.col("report_at").alias("status_at"),
        )
        .filter(F.col("uetr").isNotNull())
    )
    w = Window.partitionBy("uetr").orderBy(F.col("status_at").desc())
    latest = events.withColumn("_rn", F.row_number().over(w)).filter("_rn = 1").drop("_rn")
    current = lake.read("silver.payments").select("uetr", F.col("status").alias("_cur"), F.col("status_at").alias("_cur_at"))
    changed = (
        latest.join(current, "uetr", "inner")
        .filter((F.col("_cur") != F.col("status")) | F.col("_cur_at").isNull() | (F.col("_cur_at") < F.col("status_at")))
        .select("uetr", "status", "status_reason", "status_at")
    )
    if changed.isEmpty():
        return 0
    m = lake.merge(changed, "silver.payments", ["uetr"], update_cols=["status", "status_reason", "status_at"], insert=False)
    return int(m["num_target_rows_updated"])
