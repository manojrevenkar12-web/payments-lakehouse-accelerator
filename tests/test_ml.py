import json

import numpy as np
import pandas as pd
from pyspark.sql import functions as F

from paylake.ml import REASONS, _at_k, evaluate


def test_at_k():
    y = np.array([1, 0, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.1, 0.0])
    assert _at_k(y, s, 2) == (0.5, 0.5)
    assert _at_k(y, s, 3) == (2 / 3, 1.0)


def test_evaluate_compares_against_rules_at_same_volume():
    test = pd.DataFrame({
        "amount": [20000, 50, 50, 50], "is_new_payee": [0, 1, 0, 0], "amount_z": [0, 0, 0, 0],
        "d_new_payee_cnt_24h": [0, 0, 0, 0], "is_night": [0, 0, 0, 0], "typology": ["none", "app_scam", "none", "none"],
    })
    y = np.array([0, 1, 0, 0])
    m = evaluate(y, np.array([0.1, 0.9, 0.2, 0.0]), test, budget=0.25)
    assert m["rules_alerts"] == 1 and m["rules_recall"] == 0.0
    assert m["model_recall_at_rules_volume"] == 1.0
    assert m["recall_at_budget__app_scam"] == 1.0


def test_model_beats_rules_at_equal_alert_volume(built):
    m = built["train"].metrics
    assert m["model_recall_at_rules_volume"] > m["rules_recall"]
    assert m["model_precision_at_rules_volume"] > m["rules_precision"]
    assert m["pr_auc"] > 0.5


def test_model_registered_and_promoted(built):
    t = built["train"]
    assert t.promoted and t.version == "1"
    report = json.loads((built["settings"].lake_root / "reports" / "model_report.json").read_text())
    assert report["pr_auc"] == t.metrics["pr_auc"]


def test_alerts_respect_daily_budget_and_reasons(built):
    lake, settings = built["lake"], built["settings"]
    alerts = lake.read("gold.alerts")
    daily = lake.read("gold.payment_features").groupBy("settlement_date").count()
    per_day = alerts.groupBy("settlement_date").agg(F.count("*").alias("a")).join(daily, "settlement_date")
    for r in per_day.collect():
        assert r["a"] <= max(1, round(r["count"] * settings.alert_budget))
    allowed = {name for name, _ in REASONS}
    codes = {c for r in alerts.select(F.explode("reason_codes").alias("c")).distinct().collect() for c in r}
    assert codes and codes <= allowed
    assert alerts.filter("risk_score < 0 OR risk_score > 1").count() == 0
