"""Data-quality expectations, defined once and executed in two places.

* Locally / in Databricks jobs: `apply()` splits a DataFrame into clean and quarantined rows,
  keeps warning flags on clean rows and returns per-rule metrics.
* In Lakeflow Declarative Pipelines (formerly DLT): `lakeflow_expectations()` returns the
  same rules as the dicts that `@dlt.expect_all` / `@dlt.expect_all_or_drop` accept.

Every condition is a Spark SQL boolean expression that must be TRUE for a valid row.
NULL is treated as a failure.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import LongType, StringType, StructField, StructType, TimestampType

UUID_V4 = r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
BIC = r"^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?$"


@dataclass(frozen=True)
class Expectation:
    name: str
    condition: str
    action: str  # "quarantine" | "warn"
    description: str


RULES: list[Expectation] = [
    Expectation("uetr_valid", f"uetr IS NOT NULL AND uetr RLIKE '{UUID_V4}'", "quarantine",
                "UETR present and a valid UUIDv4 (required for gpi/ISO 20022 end-to-end tracking)"),
    Expectation("accepted_at_present", "accepted_at IS NOT NULL", "quarantine", "Acceptance timestamp present"),
    Expectation("amount_positive", "amount > 0", "quarantine", "Settlement amount strictly positive"),
    Expectation("debtor_iban_valid", "debtor_iban_checksum_ok", "quarantine", "Debtor IBAN passes ISO 13616 mod-97 and length"),
    Expectation("settlement_date_plausible",
                "settlement_date BETWEEN to_date(accepted_at) AND date_add(to_date(accepted_at), 2)", "quarantine",
                "Settlement date on or up to two days after acceptance (SEPA SCT D+1 cycle)"),
    Expectation("creditor_iban_valid", "creditor_iban_checksum_ok", "warn",
                "Creditor IBAN passes mod-97; failures are expected to come back RJCT AC01"),
    Expectation("creditor_agent_bic_valid", f"creditor_bic IS NOT NULL AND creditor_bic RLIKE '{BIC}'", "warn",
                "Creditor agent BIC present and well-formed (ISO 9362)"),
    Expectation("creditor_bic_reachable", "creditor_bank_name IS NOT NULL", "warn",
                "Creditor BIC found in the reachability directory"),
    Expectation("sepa_currency_eur", "currency = 'EUR'", "warn", "SEPA credit transfers settle in EUR"),
]

METRICS_SCHEMA = StructType([
    StructField("batch_id", LongType()), StructField("rule", StringType()), StructField("action", StringType()),
    StructField("failed", LongType()), StructField("evaluated", LongType()), StructField("run_at", TimestampType()),
])


def _failed_array(rules: list[Expectation]):
    return F.filter(
        F.array(*[F.when(~F.coalesce(F.expr(r.condition), F.lit(False)), F.lit(r.name)) for r in rules]),
        lambda x: x.isNotNull(),
    )


def apply(df: DataFrame, batch_id: int, rules: list[Expectation] = RULES) -> tuple[DataFrame, DataFrame, list[tuple]]:
    q_rules = [r for r in rules if r.action == "quarantine"]
    w_rules = [r for r in rules if r.action == "warn"]
    tagged = df.withColumn("_dq_failed", _failed_array(q_rules)).withColumn("dq_warnings", _failed_array(w_rules)).cache()

    agg = tagged.agg(
        F.count("*").alias("_n"),
        *[F.sum(F.array_contains("_dq_failed" if r.action == "quarantine" else "dq_warnings", r.name).cast("long")).alias(r.name)
          for r in rules],
    ).first()
    now = datetime.now(timezone.utc)
    metrics = [(batch_id, r.name, r.action, int(agg[r.name] or 0), int(agg["_n"]), now) for r in rules]

    clean = tagged.filter(F.size("_dq_failed") == 0).drop("_dq_failed")
    quarantined = (
        tagged.filter(F.size("_dq_failed") > 0)
        .withColumnRenamed("_dq_failed", "dq_failed")
        .withColumn("quarantined_at", F.lit(now))
    )
    return clean, quarantined, metrics


def lakeflow_expectations() -> tuple[dict, dict]:
    """(warn_rules, drop_rules) in the shape `@dlt.expect_all(...)` / `@dlt.expect_all_or_drop(...)` take."""
    warn = {r.name: r.condition for r in RULES if r.action == "warn"}
    drop = {r.name: r.condition for r in RULES if r.action == "quarantine"}
    return warn, drop
