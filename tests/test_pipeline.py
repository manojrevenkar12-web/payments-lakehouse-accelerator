"""Integration tests over the shared small run (see conftest.built)."""
from datetime import date

from pyspark.sql import functions as F

from paylake import silver
from paylake.expectations import RULES, lakeflow_expectations
from paylake.privacy import PII_COLUMNS, token


def test_bronze_is_incremental_and_idempotent(built):
    b1, b2, b3 = built["bronze"]
    m = built["manifest"]
    assert b1.files["pacs008"] > 0 and b2.files["pacs008"] > 0
    assert b1.files["pacs008"] + b2.files["pacs008"] == m["files_pacs008"] + m["resent_files"] + m["truncated_files"]
    assert b3.files == {"pacs008": 0, "pacs002": 0}           # nothing new -> nothing ingested
    assert b1.corrupt["pacs008"] + b2.corrupt["pacs008"] == m["truncated_files"]
    assert [b1.batch_id, b2.batch_id, b3.batch_id] == [1, 2, 3]


def test_silver_has_one_row_per_uetr_and_rerun_is_noop(built):
    lake = built["lake"]
    p = lake.read("silver.payments")
    assert p.count() == p.select("uetr").distinct().count()
    s3 = built["silver"][2]
    assert s3.inserted == 0 and s3.transactions == 0
    # re-sent files are absorbed as duplicates, never as new rows
    assert sum(s.duplicates_skipped for s in built["silver"]) > 0


def test_quarantine_rules_never_reach_silver(built):
    lake = built["lake"]
    p = lake.read("silver.payments")
    assert p.filter("amount <= 0").count() == 0
    assert p.filter("settlement_date > date_add(to_date(accepted_at), 2)").count() == 0
    q = lake.read("silver.payments_quarantine")
    assert q.count() == sum(s.quarantined for s in built["silver"]) > 0
    assert q.filter(F.size("dq_failed") == 0).count() == 0


def test_dq_metrics_match_injected_defects(built):
    m = built["manifest"]["defects"]
    lake = built["lake"]
    dq = {r["rule"]: r["failed"] for r in lake.read("gold.dq_summary").collect()}
    # ">=" throughout: copies inside re-sent files are evaluated (and counted) again
    assert dq["amount_positive"] >= m.get("nonpositive", 0) > 0
    assert dq["sepa_currency_eur"] >= m.get("usd", 0)            # >= because re-sent copies are evaluated too
    assert dq["creditor_iban_valid"] >= m.get("bad_iban", 0)
    assert dq["well_formed_xml"] == 1


def test_clearing_status_applied(built):
    p = built["lake"].read("silver.payments")
    bad = p.filter(F.array_contains("dq_warnings", "creditor_iban_valid"))
    assert bad.count() > 0
    # every payment with a bad creditor IBAN that has a status report was rejected AC01
    reported = bad.filter("status != 'PENDING'")
    assert reported.filter("status = 'RJCT' AND status_reason = 'AC01'").count() == reported.count()
    settled_or_rejected = p.filter("status != 'PENDING'")
    assert p.filter("status = 'ACSC'").count() > 0.97 * settled_or_rejected.count()
    # only the newest files can still be waiting for their status report
    last_full_day = date(2026, 9, 10)
    assert p.filter("status = 'PENDING'").filter(F.col("settlement_date") < last_full_day).count() == 0


def test_status_merge_is_idempotent(built, spark):
    assert silver.apply_statuses(spark, built["lake"]) == 0


def test_gold_and_alerts_carry_no_raw_pii(built):
    lake = built["lake"]
    for t in ("gold.payment_features", "gold.payments_analytics", "gold.alerts", "gold.daily_kpis", "gold.reject_reasons"):
        assert not set(lake.read(t).columns) & set(PII_COLUMNS), t


def test_tokens_are_keyed_hmac(built):
    lake, settings = built["lake"], built["settings"]
    row = lake.read("silver.payments").select("debtor_iban", "debtor_ref").first()
    assert row["debtor_ref"] == token(row["debtor_iban"], settings.hmac_key)
    assert token(row["debtor_iban"], "another-key") != row["debtor_ref"]


def test_delta_history_records_merges(built):
    ops = [h["operation"] for h in built["lake"].history("silver.payments")]
    assert "MERGE" in ops


def test_lakeflow_expectations_cover_all_rules():
    warn, drop = lakeflow_expectations()
    assert set(warn) | set(drop) == {r.name for r in RULES}
    assert not set(warn) & set(drop)
