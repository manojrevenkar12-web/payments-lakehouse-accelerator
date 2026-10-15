"""Fraud / mule scoring: train, evaluate against the bank's rule set, gate promotion, batch-score in Spark.

Evaluation is built around how a fraud team actually works: analysts can review a fixed
number of alerts per day, so the headline numbers are precision and recall *at an alert
budget*, and the model is compared with the incumbent rules *at the same alert volume*.
The split is by time (train on earlier days, test on later days) to avoid leakage.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import Settings
from .gold import FEATURE_COLUMNS
from .lake import Lake

EXPERIMENT = "paylake-fraud-scoring"
MODEL_NAME = "paylake_fraud_scorer"
PARAMS = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40,
              l2_regularization=1.0, random_state=42)


class FraudScorer(mlflow.pyfunc.PythonModel):
    """pyfunc wrapper so `predict` returns a fraud probability (sklearn's predict returns a class)."""

    def __init__(self, model, features):
        self.model = model
        self.features = list(features)

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.Series:
        X = model_input[self.features].astype("float64")
        return pd.Series(self.model.predict_proba(X)[:, 1])


def rules_baseline(df: pd.DataFrame) -> pd.Series:
    """A typical incumbent rule set for outgoing SCT monitoring."""
    return (
        (df["amount"].astype(float) >= 10000)
        | ((df["is_new_payee"] == 1) & (df["amount_z"] >= 3))
        | (df["d_new_payee_cnt_24h"] >= 3)
        | ((df["is_night"] == 1) & (df["is_new_payee"] == 1) & (df["amount"].astype(float) >= 1000))
    )


def _at_k(y: np.ndarray, score: np.ndarray, k: int) -> tuple[float, float]:
    k = max(1, min(k, len(y)))
    top = np.argsort(-score, kind="stable")[:k]
    tp = y[top].sum()
    return float(tp / k), float(tp / max(1, y.sum()))


def evaluate(y: np.ndarray, score: np.ndarray, test: pd.DataFrame, budget: float) -> dict:
    rules = rules_baseline(test).to_numpy()
    k_budget = int(round(budget * len(y)))
    k_rules = int(rules.sum())
    p_b, r_b = _at_k(y, score, k_budget)
    p_m_at_rules, r_m_at_rules = _at_k(y, score, k_rules)
    rules_tp = int((rules & (y == 1)).sum())
    out = {
        "test_payments": int(len(y)),
        "test_fraud": int(y.sum()),
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "alert_budget_share": budget,
        "alerts_at_budget": k_budget,
        "precision_at_budget": p_b,
        "recall_at_budget": r_b,
        "rules_alerts": k_rules,
        "rules_precision": rules_tp / max(1, k_rules),
        "rules_recall": rules_tp / max(1, int(y.sum())),
        "model_precision_at_rules_volume": p_m_at_rules,
        "model_recall_at_rules_volume": r_m_at_rules,
    }
    # hybrid: keep the incumbent rules, spend the rest of the budget on the highest model scores
    k_total = max(k_budget, k_rules)
    hybrid = rules.copy()
    order = np.argsort(-score, kind="stable")
    hybrid[order[~rules[order]][: max(0, k_total - k_rules)]] = True
    out["hybrid_alerts"] = int(hybrid.sum())
    out["hybrid_precision"] = float((hybrid & (y == 1)).sum() / max(1, hybrid.sum()))
    out["hybrid_recall"] = float((hybrid & (y == 1)).sum() / max(1, int(y.sum())))
    out["max_recall_at_budget"] = float(min(1.0, k_budget / max(1, int(y.sum()))))

    top = np.zeros(len(y), dtype=bool)
    top[np.argsort(-score, kind="stable")[:k_budget]] = True
    for typ, grp in test.assign(_top=top, _rule=rules, _hyb=hybrid).groupby("typology"):
        if typ == "none":
            continue
        out[f"recall_at_budget__{typ}"] = float(grp["_top"].mean())
        out[f"rules_recall__{typ}"] = float(grp["_rule"].mean())
        out[f"hybrid_recall__{typ}"] = float(grp["_hyb"].mean())
    return out


def load_cases(spark: SparkSession, lake: Lake, landing: Path) -> DataFrame:
    path = landing / "cases" / "case_outcomes.csv"
    cases = (
        spark.read.option("header", True).csv(path.resolve().as_posix())
        .select(F.lower("uetr").alias("uetr"), "typology", F.to_date("confirmed_on").alias("confirmed_on"))
    )
    lake.overwrite(cases, "silver.case_outcomes")
    return lake.read("silver.case_outcomes")


@dataclass
class TrainResult:
    run_id: str
    version: str
    promoted: bool
    metrics: dict


def train(spark: SparkSession, lake: Lake, settings: Settings, landing: Path, test_share: float = 0.3) -> TrainResult:
    cases = load_cases(spark, lake, landing)
    feats = lake.read("gold.payment_features")
    labelled = (
        feats.join(cases.select("uetr", "typology"), "uetr", "left")
        .withColumn("label", F.col("typology").isNotNull().cast("int"))
        .fillna({"typology": "none"})
    )
    pdf = labelled.toPandas()
    pdf["settlement_date"] = pd.to_datetime(pdf["settlement_date"])
    days = sorted(pdf["settlement_date"].unique())
    cut = days[int(len(days) * (1 - test_share))]
    train_df, test_df = pdf[pdf["settlement_date"] < cut], pdf[pdf["settlement_date"] >= cut]

    X_tr, y_tr = train_df[FEATURE_COLUMNS].astype("float64"), train_df["label"].to_numpy()
    X_te, y_te = test_df[FEATURE_COLUMNS].astype("float64"), test_df["label"].to_numpy()
    model = HistGradientBoostingClassifier(**PARAMS).fit(X_tr, y_tr)
    score = model.predict_proba(X_te)[:, 1]
    metrics = evaluate(y_te, score, test_df.reset_index(drop=True), settings.alert_budget)
    metrics["train_payments"], metrics["train_fraud"] = int(len(y_tr)), int(y_tr.sum())

    mlflow.set_tracking_uri(settings.mlflow_uri)
    if settings.is_databricks:  # pragma: no cover
        mlflow.set_registry_uri("databricks-uc")
        model_name = f"{settings.catalog}.gold.fraud_scorer"
    else:
        model_name = MODEL_NAME
    mlflow.set_experiment(f"/Shared/{EXPERIMENT}" if settings.is_databricks else EXPERIMENT)
    with mlflow.start_run(run_name=f"hgb-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}") as run:
        mlflow.log_params({**PARAMS, "features": len(FEATURE_COLUMNS), "split_date": str(pd.Timestamp(cut).date()),
                           "test_share": test_share})
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        mlflow.log_dict(metrics, "evaluation.json")
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=FraudScorer(model, FEATURE_COLUMNS),
            input_example=X_te.head(5),
            registered_model_name=model_name,
        )
    client = mlflow.MlflowClient()
    version = str(info.registered_model_version)

    # champion/challenger gate: promote only if PR-AUC does not regress against the current champion
    promoted, champion_pr_auc = True, None
    try:
        champ = client.get_model_version_by_alias(model_name, "champion")
        champion_pr_auc = client.get_run(champ.run_id).data.metrics.get("pr_auc")
        promoted = champion_pr_auc is None or metrics["pr_auc"] >= champion_pr_auc - 0.005
    except Exception:
        pass
    if promoted:
        client.set_registered_model_alias(model_name, "champion", version)
    metrics["champion_pr_auc_before"] = champion_pr_auc
    metrics["promoted"] = promoted

    report_dir = settings.lake_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "model_report.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    return TrainResult(run.info.run_id, version, promoted, metrics)


REASONS = [
    ("AMOUNT_SPIKE", "amount_z >= 3"),
    ("NEW_PAYEE", "is_new_payee = 1"),
    ("NIGHT", "is_night = 1"),
    ("VELOCITY", "d_cnt_1h >= 3"),
    ("MANY_NEW_PAYEES", "d_new_payee_cnt_24h >= 3"),
    ("PAYEE_FAN_IN", "c_distinct_debtors_24h >= 8 AND creditor_age_days < 3"),
    ("NEAR_THRESHOLD_REPEAT", "d_near_threshold_cnt_24h >= 2"),
    ("CROSS_BORDER", "cross_border = 1"),
]


def score(spark: SparkSession, lake: Lake, settings: Settings) -> int:
    mlflow.set_tracking_uri(settings.mlflow_uri)
    if settings.is_databricks:  # pragma: no cover
        mlflow.set_registry_uri("databricks-uc")
        model_name = f"{settings.catalog}.gold.fraud_scorer"
    else:
        model_name = MODEL_NAME
    mv = mlflow.MlflowClient().get_model_version_by_alias(model_name, "champion")
    udf = mlflow.pyfunc.spark_udf(spark, f"models:/{model_name}@champion", result_type="double", env_manager="local")

    feats = lake.read("gold.payment_features")
    scored = feats.withColumn("risk_score", udf(F.struct(*[F.col(c).cast("double").alias(c) for c in FEATURE_COLUMNS])))
    w = Window.partitionBy("settlement_date").orderBy(F.col("risk_score").desc(), F.col("uetr"))
    daily_n = Window.partitionBy("settlement_date")
    reasons = F.filter(F.array(*[F.when(F.expr(cond), F.lit(name)) for name, cond in REASONS]), lambda x: x.isNotNull())
    alerts = (
        scored.withColumn("alert_rank", F.row_number().over(w))
        .withColumn("_daily", F.count("*").over(daily_n))
        .filter(F.col("alert_rank") <= F.greatest(F.lit(1), F.round(F.col("_daily") * settings.alert_budget)))
        .select(
            "uetr", "settlement_date", "accepted_at", "debtor_ref", "creditor_ref", "amount", "currency",
            F.round("risk_score", 5).alias("risk_score"), "alert_rank", reasons.alias("reason_codes"),
            F.lit(str(mv.version)).alias("model_version"), F.lit(datetime.now(timezone.utc)).alias("scored_at"),
            F.lit("OPEN").alias("review_status"),
        )
    )
    lake.overwrite(alerts, "gold.alerts", partition_by=["settlement_date"])
    return lake.read("gold.alerts").count()
