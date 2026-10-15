"""Gold: business-level tables for analysts, the fraud model and the copilot.

Nothing in gold carries raw PII: parties appear only as HMAC tokens (debtor_ref / creditor_ref).

gold.payment_features  - point-in-time behavioural features per payment (no look-ahead)
gold.payments_analytics- PII-free payment fact table for analysts and the copilot
gold.daily_kpis        - volume, value and reject rate by day and creditor country
gold.reject_reasons    - pacs.002 reject reasons by day
gold.dq_summary        - data-quality rule failure rates across all batches
"""
from __future__ import annotations

from dataclasses import dataclass

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from .lake import Lake

H, D = 3600, 86400

FEATURE_COLUMNS = [
    "log_amount", "amount_z", "thin_history", "hour_local", "is_night", "is_weekend",
    "is_new_payee", "prior_payments_to_payee", "d_cnt_1h", "d_cnt_24h", "d_sum_24h_log",
    "d_new_payee_cnt_24h", "d_near_threshold_cnt_24h", "near_threshold", "cross_border",
    "c_cnt_24h", "c_distinct_debtors_24h", "creditor_age_days", "c_new_inbound_share_24h",
]


@dataclass
class GoldResult:
    features: int
    kpi_rows: int


def payment_features(silver: DataFrame) -> DataFrame:
    base = (
        silver.select("uetr", "accepted_at", "settlement_date", "amount", "currency", "debtor_ref", "creditor_ref",
                      "debtor_country", "creditor_country")
        .filter(F.col("debtor_ref").isNotNull() & F.col("creditor_ref").isNotNull())
        .withColumn("ts", F.col("accepted_at").cast("long"))
        .withColumn("amt", F.col("amount").cast("double"))
        .withColumn("log_amount", F.log1p("amt"))
    )

    def rng(part, lo, hi):
        return Window.partitionBy(*part).orderBy("ts").rangeBetween(lo, hi)

    d1h, d24h = rng(["debtor_ref"], -H, 0), rng(["debtor_ref"], -D, 0)
    d_hist = rng(["debtor_ref"], -30 * D, -1)          # strictly before this payment
    pair_all = Window.partitionBy("debtor_ref", "creditor_ref").orderBy("ts").rowsBetween(Window.unboundedPreceding, -1)
    c24h = rng(["creditor_ref"], -D, 0)
    c_all = Window.partitionBy("creditor_ref").orderBy("ts").rangeBetween(Window.unboundedPreceding, 0)

    f = (
        base
        .withColumn("prior_payments_to_payee", F.coalesce(F.count("*").over(pair_all), F.lit(0)))
        .withColumn("is_new_payee", (F.col("prior_payments_to_payee") == 0).cast("int"))
        .withColumn("d_cnt_1h", F.count("*").over(d1h))
        .withColumn("d_cnt_24h", F.count("*").over(d24h))
        .withColumn("d_sum_24h_log", F.log1p(F.sum("amt").over(d24h)))
        .withColumn("d_new_payee_cnt_24h", F.sum("is_new_payee").over(d24h))
        .withColumn("near_threshold", ((F.col("amt") >= 9000) & (F.col("amt") < 10000)).cast("int"))
        .withColumn("d_near_threshold_cnt_24h", F.sum("near_threshold").over(d24h))
        .withColumn("_h_n", F.count("*").over(d_hist))
        .withColumn("_h_mean", F.avg("log_amount").over(d_hist))
        .withColumn("_h_std", F.stddev_samp("log_amount").over(d_hist))
        .withColumn("thin_history", (F.col("_h_n") < 3).cast("int"))
        .withColumn(
            "amount_z",
            F.when(F.col("_h_n") >= 3,
                   (F.col("log_amount") - F.col("_h_mean")) / F.greatest(F.coalesce(F.col("_h_std"), F.lit(0.0)), F.lit(0.25)))
            .otherwise(F.lit(0.0)),
        )
        .withColumn("c_cnt_24h", F.count("*").over(c24h))
        .withColumn("c_distinct_debtors_24h", F.size(F.collect_set("debtor_ref").over(c24h)))
        .withColumn("c_new_inbound_share_24h", F.avg("is_new_payee").over(c24h))
        .withColumn("creditor_age_days", (F.col("ts") - F.min("ts").over(c_all)) / D)
        .withColumn("_local", F.from_utc_timestamp("accepted_at", "Europe/Berlin"))
        .withColumn("hour_local", F.hour("_local"))
        .withColumn("is_night", (F.col("hour_local") < 6).cast("int"))
        .withColumn("is_weekend", F.dayofweek("_local").isin(1, 7).cast("int"))
        .withColumn("cross_border", (F.col("debtor_country") != F.col("creditor_country")).cast("int"))
    )
    return f.select("uetr", "accepted_at", "settlement_date", "debtor_ref", "creditor_ref", "amount", "currency",
                    *FEATURE_COLUMNS)


def payments_analytics(silver: DataFrame) -> DataFrame:
    return silver.select(
        "uetr", "accepted_at", "settlement_date", "amount", "currency", "debtor_ref", "creditor_ref",
        "debtor_country", "creditor_country", "creditor_bic", "creditor_bank_name", "status", "status_reason",
        F.size("dq_warnings").alias("dq_warning_count"),
    )


def daily_kpis(silver: DataFrame) -> DataFrame:
    return (
        silver.groupBy("settlement_date", "creditor_country")
        .agg(
            F.count("*").alias("payments"),
            F.sum(F.when(F.col("currency") == "EUR", F.col("amount"))).alias("value_eur"),
            F.sum((F.col("status") == "RJCT").cast("int")).alias("rejected"),
            F.sum((F.col("status") == "PENDING").cast("int")).alias("pending"),
        )
        .withColumn("reject_rate", F.round(F.col("rejected") / F.col("payments"), 5))
    )


def reject_reasons(silver: DataFrame) -> DataFrame:
    return (
        silver.filter(F.col("status") == "RJCT")
        .groupBy("settlement_date", "status_reason").agg(F.count("*").alias("rejections"))
    )


def dq_summary(metrics: DataFrame) -> DataFrame:
    return (
        metrics.groupBy("rule", "action")
        .agg(F.sum("failed").alias("failed"), F.sum("evaluated").alias("evaluated"), F.max("run_at").alias("last_run_at"))
        .withColumn("failure_rate", F.round(F.col("failed") / F.col("evaluated"), 6))
    )


def build(spark: SparkSession, lake: Lake) -> GoldResult:
    silver = lake.read("silver.payments").cache()
    feats = payment_features(silver)
    lake.overwrite(feats, "gold.payment_features", partition_by=["settlement_date"])
    lake.overwrite(payments_analytics(silver), "gold.payments_analytics", partition_by=["settlement_date"])
    kpis = daily_kpis(silver)
    lake.overwrite(kpis, "gold.daily_kpis")
    lake.overwrite(reject_reasons(silver), "gold.reject_reasons")
    lake.overwrite(dq_summary(lake.read("silver.dq_metrics")), "gold.dq_summary")
    n_feat = lake.read("gold.payment_features").count()
    n_kpi = lake.read("gold.daily_kpis").count()
    silver.unpersist()
    return GoldResult(n_feat, n_kpi)
