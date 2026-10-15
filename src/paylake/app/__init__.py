"""Alerts API for fraud analysts.

Reads the PII-free gold tables directly as Delta (delta-rs, no Spark needed for serving) and
records analyst decisions in ops.alert_reviews, which becomes labelled feedback for the next
training run. On Databricks the same API can run as a Databricks App reading via a SQL warehouse.
"""
from __future__ import annotations

import json
import time
from datetime import date, datetime, timezone
from importlib import resources
from typing import Literal

import pandas as pd
import pyarrow as pa
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ..config import Settings

app = FastAPI(title="Payments Lakehouse - Alerts API", version="0.1.0")
_settings = Settings()
_cache: dict[str, tuple[float, pd.DataFrame]] = {}
TTL_SECONDS = 30


def settings() -> Settings:
    return _settings


def _table(name: str) -> pd.DataFrame:
    from deltalake import DeltaTable

    hit = _cache.get(name)
    if hit and time.time() - hit[0] < TTL_SECONDS:
        return hit[1]
    layer, table = name.split(".")
    path = settings().lake_root / layer / table
    if not (path / "_delta_log").exists():
        raise HTTPException(503, f"{name} not built yet - run the pipeline first")
    df = DeltaTable(str(path)).to_pandas()
    _cache[name] = (time.time(), df)
    return df


def _records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records", date_format="iso", default_handler=str))


class Review(BaseModel):
    decision: Literal["CONFIRMED_FRAUD", "FALSE_POSITIVE", "ESCALATED"]
    analyst: str = Field(min_length=2, max_length=64)
    note: str | None = Field(default=None, max_length=500)


@app.get("/health")
def health():
    return {"status": "ok", "lake_root": str(settings().lake_root)}


@app.get("/alerts")
def alerts(day: date | None = None, min_score: float = Query(0.0, ge=0, le=1), limit: int = Query(50, ge=1, le=500)):
    df = _table("gold.alerts")
    if day is None:
        day = pd.to_datetime(df["settlement_date"]).max().date()
    df = df[(pd.to_datetime(df["settlement_date"]).dt.date == day) & (df["risk_score"] >= min_score)]
    df = df.sort_values(["alert_rank"]).head(limit)
    return {"day": day.isoformat(), "count": len(df), "alerts": _records(df)}


@app.get("/alerts/{uetr}")
def alert_detail(uetr: str):
    a = _table("gold.alerts")
    hit = a[a["uetr"] == uetr.lower()]
    if hit.empty:
        raise HTTPException(404, "alert not found")
    p = _table("gold.payments_analytics")
    pay = p[p["uetr"] == uetr.lower()]
    debtor = hit.iloc[0]["debtor_ref"]
    history = p[p["debtor_ref"] == debtor].sort_values("accepted_at").tail(20)
    return {"alert": _records(hit)[0], "payment": _records(pay)[0] if not pay.empty else None,
            "debtor_recent_payments": _records(history[["uetr", "accepted_at", "amount", "creditor_ref", "creditor_country", "status"]])}


@app.post("/alerts/{uetr}/review", status_code=201)
def review(uetr: str, body: Review):
    from deltalake import write_deltalake

    a = _table("gold.alerts")
    if a[a["uetr"] == uetr.lower()].empty:
        raise HTTPException(404, "alert not found")
    row = pa.table({
        "uetr": [uetr.lower()], "decision": [body.decision], "analyst": [body.analyst], "note": [body.note],
        "reviewed_at": pa.array([datetime.now(timezone.utc)], type=pa.timestamp("us", tz="UTC")),
    })
    path = settings().lake_root / "ops" / "alert_reviews"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_deltalake(str(path), row, mode="append")
    _cache.pop("ops.alert_reviews", None)
    return {"recorded": True}


@app.get("/kpis/daily")
def kpis():
    df = _table("gold.daily_kpis").groupby("settlement_date", as_index=False)[["payments", "value_eur", "rejected"]].sum()
    df["reject_rate"] = (df["rejected"] / df["payments"]).round(5)
    df["value_eur"] = df["value_eur"].astype(float).round(2)
    return _records(df.sort_values("settlement_date"))


@app.get("/dq")
def dq():
    return _records(_table("gold.dq_summary").sort_values("failure_rate", ascending=False))


@app.get("/model")
def model():
    path = settings().lake_root / "reports" / "model_report.json"
    if not path.exists():
        raise HTTPException(503, "model not trained yet")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/", response_class=HTMLResponse)
def index():
    return resources.files(__package__).joinpath("static/index.html").read_text(encoding="utf-8")
