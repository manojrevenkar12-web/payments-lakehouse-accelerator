"""Point-in-time correctness of the behavioural features on a hand-built example."""
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from paylake.gold import payment_features

T0 = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)


def _row(uetr, minutes, amount, debtor, creditor, cc="DE"):
    ts = T0 + timedelta(minutes=minutes)
    return (uetr, ts, ts.date(), Decimal(str(amount)), "EUR", debtor, creditor, "DE", cc)


@pytest.fixture(scope="module")
def feats(spark):
    rows = [
        _row("a1", -60 * 24 * 5, 100, "D1", "C1"),
        _row("a2", -60 * 24 * 4, 110, "D1", "C1"),
        _row("a3", -60 * 24 * 3, 90, "D1", "C1"),
        _row("a4", -60 * 24 * 2, 105, "D1", "C2"),
        _row("a5", 0, 5000, "D1", "M1", cc="NL"),     # spike to a new payee
        _row("a6", 10, 4000, "D1", "M1", cc="NL"),    # second payment to the same payee 10 min later
        _row("b1", 20, 300, "D2", "M1"),
        _row("b2", 30, 300, "D3", "M1"),
    ]
    cols = ["uetr", "accepted_at", "settlement_date", "amount", "currency", "debtor_ref", "creditor_ref",
            "debtor_country", "creditor_country"]
    df = spark.createDataFrame(rows, cols)
    return {r["uetr"]: r for r in payment_features(df).collect()}


def test_new_payee_and_prior_counts(feats):
    assert feats["a5"]["is_new_payee"] == 1 and feats["a6"]["is_new_payee"] == 0
    assert feats["a3"]["prior_payments_to_payee"] == 2


def test_amount_z_uses_only_prior_history(feats):
    assert feats["a5"]["amount_z"] > 5           # far above the 90-110 history
    assert feats["a1"]["thin_history"] == 1 and feats["a1"]["amount_z"] == 0.0
    # a6 history includes a5 (a past payment) - the spike is part of history, not the future
    assert feats["a6"]["amount_z"] < feats["a5"]["amount_z"]


def test_velocity_and_fan_in_windows(feats):
    assert feats["a6"]["d_cnt_1h"] == 2
    assert feats["a5"]["c_distinct_debtors_24h"] == 1          # only counts up to and including itself
    assert feats["b2"]["c_distinct_debtors_24h"] == 3          # D1, D2, D3 within 24h
    assert feats["b2"]["creditor_age_days"] == pytest.approx(30 / 1440)
    assert feats["a5"]["cross_border"] == 1 and feats["b1"]["cross_border"] == 0
